"""
Enhanced promptParser.py - Complete Business Scraping
Uses BERT + OpenAI + Selenium + Google Places API + OSM for maximum business discovery
Returns: Name, Address, Website, Business Group, Business Type
"""

import os
import re
import pandas as pd
import requests
from typing import List, Dict, Tuple, Optional
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import time
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
import json
from geopy.geocoders import Nominatim
import overpy
from concurrent.futures import ThreadPoolExecutor, as_completed

# Enhanced imports
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

try:
    from fake_useragent import UserAgent
    FAKE_USERAGENT_AVAILABLE = True
except ImportError:
    FAKE_USERAGENT_AVAILABLE = False

# Configuration
GOOGLE_API_KEY = "AIzaSyDMx0L3P33m8wadNp3PAv6GT7vDApRh4qQ"
OPENAI_API_KEY = "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"


class PromptParser:
    def __init__(self, folder_name="Skill-tag csv", output_folder="Output", enable_ner_workflow=False):
        self.folder_name = folder_name
        self.output_folder = output_folder
        self.enable_ner_workflow = enable_ner_workflow
        self.drive = self.authenticate_drive()
        self.folder_id = self.get_folder_id(self.folder_name)
        self.output_folder_id = self.get_folder_id(self.output_folder)
        self.api_key = GOOGLE_API_KEY
        
        # Initialize components
        self.init_components()

    def init_components(self):
        """Initialize all scraping components"""
        print("🔄 Initializing scraping components...")
        
        # Semantic model
        try:
            self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        except:
            self.semantic_model = None
        
        # BERT model for entity extraction
        if TRANSFORMERS_AVAILABLE:
            try:
                self.ner_pipeline = pipeline("ner", 
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple")
            except:
                self.ner_pipeline = None
        else:
            self.ner_pipeline = None
        
        # Selenium drivers
        self.drivers = []
        if SELENIUM_AVAILABLE:
            try:
                for i in range(2):  # 2 drivers for parallel scraping
                    chrome_options = Options()
                    chrome_options.add_argument("--headless")
                    chrome_options.add_argument("--no-sandbox")
                    chrome_options.add_argument("--disable-dev-shm-usage")
                    chrome_options.add_argument("--disable-gpu")
                    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
                    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                    
                    if FAKE_USERAGENT_AVAILABLE:
                        ua = UserAgent()
                        chrome_options.add_argument(f"--user-agent={ua.random}")
                    
                    driver = webdriver.Chrome(options=chrome_options)
                    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                    self.drivers.append(driver)
            except:
                self.drivers = []
        
        # Request sessions
        self.sessions = []
        session1 = requests.Session()
        if FAKE_USERAGENT_AVAILABLE:
            ua = UserAgent()
            session1.headers.update({'User-Agent': ua.random})
        self.sessions.append(session1)
        
        if CLOUDSCRAPER_AVAILABLE:
            self.sessions.append(cloudscraper.create_scraper())
        
        # Geocoding
        try:
            self.geolocator = Nominatim(user_agent="business_scraper")
            self.osm_api = overpy.Overpass()
        except:
            self.geolocator = None
            self.osm_api = None
        
        print("✅ Components initialized")

    def authenticate_drive(self):
        gauth = GoogleAuth()
        gauth.LoadCredentialsFile("mycreds.txt")
        if gauth.credentials is None:
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            gauth.Refresh()
        else:
            gauth.Authorize()
        gauth.SaveCredentialsFile("mycreds.txt")
        return GoogleDrive(gauth)

    def get_folder_id(self, folder_name):
        folder_list = self.drive.ListFile({
            'q': f"mimeType='application/vnd.google-apps.folder' and title='{folder_name}' and trashed=false"
        }).GetList()
        if not folder_list:
            raise Exception(f"No folder named '{folder_name}' found.")
        return folder_list[0]['id']

    def load_business_type_map(self) -> Dict[str, List[str]]:
        root_id = self.get_folder_id("Kobeyo Business Directory")
        original_folder_id = self.folder_id
        self.folder_id = root_id
        files_in_root = self.list_business_files()
        self.folder_id = original_folder_id

        file_obj = next((f for f in files_in_root if f['title'] == "Extracted_Business_Groups_and_Types.csv"), None)
        if not file_obj:
            raise FileNotFoundError("Extracted_Business_Groups_and_Types.csv not found in root.")

        file_path = self.download_file(file_obj)
        df = pd.read_csv(file_path)
        mapping = {}

        for _, row in df.iterrows():
            group = str(row["Business Group"]).strip().lower()
            types = [x.strip() for x in str(row["Business Type"]).split(",")]
            mapping[group] = types

        self.business_group_df = df
        return mapping

    def list_business_files(self):
        return self.drive.ListFile({
            'q': f"'{self.folder_id}' in parents and trashed=false"
        }).GetList()

    def download_file(self, file_obj):
        file_path = f"/tmp/{file_obj['title']}"
        file_obj.GetContentFile(file_path)
        return file_path

    def parse_location_and_type(self, prompt: str) -> Dict[str, str]:
        match = re.search(r"(.*) in (.*)", prompt, re.IGNORECASE)
        return {"business_type": match.group(1).strip(), "location": match.group(2).strip()} if match else {}

    def identify_business_group_semantic(self, business_type: str) -> str:
        if not hasattr(self, 'business_group_df'):
            self.load_business_type_map()
        
        business_type = business_type.lower().strip()
        best_score = 0.0
        best_group = None

        for _, row in self.business_group_df.iterrows():
            types = [t.strip().lower() for t in str(row["Business Type"]).split(",") if t.strip()]
            
            if types and self.semantic_model:
                try:
                    embeddings = self.semantic_model.encode(types, convert_to_tensor=True)
                    input_embedding = self.semantic_model.encode(business_type, convert_to_tensor=True)
                    
                    scores = util.cos_sim(input_embedding, embeddings)[0]
                    max_score = scores.max().item()
                    
                    if max_score > best_score:
                        best_score = max_score
                        best_group = row["Business Group"]
                except:
                    continue

        if best_score > 0.6:
            return best_group
        else:
            available_groups = self.business_group_df["Business Group"].unique().tolist()
            return self.match_business_group(business_type, available_groups)

    def match_business_group(self, user_input: str, available_groups: List[str]) -> str:
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = (
            f"A user entered the business group: '{user_input}'.\n"
            f"Here is a list of valid business groups:\n"
            + "\n".join(f"- {g}" for g in available_groups)
            + "\n\nWhich of these is the best match for the user input? Respond with only one group name from the list above."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except:
            return user_input

    # === GOOGLE PLACES API (TRULY UNLIMITED) ===
    
    def search_all_places(self, query: str, location: str) -> List[Dict]:
        """Search ALL places from Google Places API - NO LIMITS"""
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        all_results = []
        params = {"query": f"{query} in {location}", "key": self.api_key}
        page_count = 0
        
        # Try different search variations to get more results
        search_variations = [
            f"{query} in {location}",
            f"{query} near {location}",
            f"{query} {location}",
            f"best {query} in {location}",
            f"top {query} in {location}"
        ]
        
        for search_term in search_variations:
            params = {"query": search_term, "key": self.api_key}
            page_count = 0
            
            while page_count < 10:  # More pages per search
                try:
                    response = requests.get(url, params=params).json()
                    results = response.get("results", [])
                    
                    if not results:
                        break
                    
                    all_results.extend(results)
                    
                    next_page_token = response.get("next_page_token")
                    if not next_page_token:
                        break
                    
                    time.sleep(3)  # Wait for token to become valid
                    params = {"pagetoken": next_page_token, "key": self.api_key}
                    page_count += 1
                    
                except:
                    break
            
            time.sleep(1)  # Between search variations
        
        # Remove duplicates based on place_id
        seen_ids = set()
        unique_results = []
        for result in all_results:
            place_id = result.get('place_id', '')
            if place_id and place_id not in seen_ids:
                seen_ids.add(place_id)
                unique_results.append(result)
        
        return unique_results

    def get_place_details(self, place_id: str) -> Dict:
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {
            "place_id": place_id,
            "fields": "name,website,formatted_phone_number,formatted_address",
            "key": self.api_key
        }
        try:
            response = requests.get(url, params=params).json()
            return response.get("result", {})
        except:
            return {}

    def get_place_details_batch(self, place_ids: List[str]) -> Dict[str, Dict]:
        """Get details for multiple places in parallel"""
        details_dict = {}
        
        def get_single_detail(place_id):
            return place_id, self.get_place_details(place_id)

        with ThreadPoolExecutor(max_workers=20) as executor:  # More workers
            futures = {executor.submit(get_single_detail, pid): pid for pid in place_ids}
            for future in as_completed(futures):
                place_id, details = future.result()
                details_dict[place_id] = details

        return details_dict

    # === WEB SCRAPING ===
    
    def scrape_web_directories(self, business_type: str, location: str) -> List[Dict]:
        """Scrape multiple web directories"""
        all_businesses = []
        
        scrapers = [
            self.scrape_yellowpages,
            self.scrape_yelp,
            self.scrape_google_maps
        ]
        
        for scraper in scrapers:
            try:
                businesses = scraper(business_type, location)
                all_businesses.extend(businesses)
                time.sleep(1)
            except:
                continue
        
        return self.deduplicate_simple(all_businesses)

    def scrape_yellowpages(self, business_type: str, location: str) -> List[Dict]:
        """Scrape Yellow Pages - get MORE results"""
        businesses = []
        
        if not self.drivers:
            return businesses
        
        driver = self.drivers[0]
        
        try:
            search_terms = business_type.replace(' ', '+')
            geo_terms = location.replace(' ', '+')
            
            # Try multiple pages
            for page in range(10):  # Get more pages
                url = f"https://www.yellowpages.com/search?search_terms={search_terms}&geo_location_terms={geo_terms}&page={page+1}"
                
                driver.get(url)
                time.sleep(2)
                
                # Scroll to load more
                for _ in range(5):
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(1)
                
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                listings = soup.find_all(['div'], class_=re.compile(r'result|listing|search-results'))
                
                if not listings:
                    break
                
                page_businesses = []
                for listing in listings:
                    business = self.extract_business_info(listing)
                    if business.get('name'):
                        business['source'] = 'yellowpages'
                        page_businesses.append(business)
                
                businesses.extend(page_businesses)
                
                # Stop if no new businesses found
                if not page_businesses:
                    break
                
                time.sleep(1)
            
        except:
            pass
        
        return businesses

    def scrape_yelp(self, business_type: str, location: str) -> List[Dict]:
        """Scrape Yelp - get MORE results"""
        businesses = []
        
        session = self.sessions[0] if self.sessions else requests.Session()
        
        try:
            # Try multiple pages and search variations
            for start in range(0, 200, 10):  # Get up to 20 pages
                params = {
                    'find_desc': business_type,
                    'find_loc': location,
                    'start': start
                }
                
                response = session.get("https://www.yelp.com/search", params=params, timeout=10)
                if response.status_code != 200:
                    break
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                listings = soup.find_all(['div'], class_=re.compile(r'searchResult|business|result'))
                
                if not listings:
                    break
                
                page_businesses = []
                for listing in listings:
                    business = self.extract_business_info(listing)
                    if business.get('name'):
                        business['source'] = 'yelp'
                        page_businesses.append(business)
                
                businesses.extend(page_businesses)
                
                # Stop if no new businesses found
                if not page_businesses:
                    break
                
                time.sleep(1)
        except:
            pass
        
        return businesses

    def scrape_google_maps(self, business_type: str, location: str) -> List[Dict]:
        """Scrape Google Maps - get MORE results"""
        businesses = []
        
        if not self.drivers or len(self.drivers) < 2:
            return businesses
        
        driver = self.drivers[1]
        
        try:
            search_query = f"{business_type} {location}"
            url = f"https://www.google.com/maps/search/{search_query.replace(' ', '+')}"
            
            driver.get(url)
            time.sleep(5)
            
            # Scroll extensively to load ALL results
            for _ in range(50):  # Much more scrolling
                try:
                    results_panel = driver.find_element(By.CSS_SELECTOR, "[role='main']")
                    driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", results_panel)
                    time.sleep(1)
                    
                    # Check if "Load more results" button exists and click it
                    try:
                        load_more = driver.find_element(By.XPATH, "//button[contains(text(), 'Load more')]")
                        driver.execute_script("arguments[0].click();", load_more)
                        time.sleep(3)
                    except:
                        pass
                        
                except:
                    break
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            listings = soup.find_all(['div'], class_=re.compile(r'place|result|listing'))
            
            for listing in listings:
                business = self.extract_business_info(listing)
                if business.get('name'):
                    business['source'] = 'google_maps'
                    businesses.append(business)
            
        except:
            pass
        
        return businesses

    def extract_business_info(self, element) -> Dict:
        """Extract business info using BERT and regex"""
        text = element.get_text(strip=True)
        business = {"name": "", "address": "", "phone": "", "website": ""}
        
        # Use BERT if available
        if self.ner_pipeline:
            try:
                entities = self.ner_pipeline(text)
                for entity in entities:
                    if entity['entity_group'] == 'ORG' and not business["name"]:
                        business["name"] = entity['word'].replace('##', '')
                    elif entity['entity_group'] == 'LOC' and not business["address"]:
                        business["address"] = entity['word'].replace('##', '')
            except:
                pass
        
        # Regex extractions
        if not business["name"]:
            # Try to find name in strong/heading tags
            name_elem = element.find(['h1', 'h2', 'h3', 'h4', 'strong', 'b'])
            if name_elem:
                business["name"] = name_elem.get_text(strip=True)
            else:
                lines = text.split('\n')
                if lines:
                    business["name"] = lines[0].strip()
        
        # Phone number
        phone_match = re.search(r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})', text)
        if phone_match:
            business["phone"] = phone_match.group(1)
        
        # Website
        website_match = re.search(r'https?://[^\s<>"]+', text)
        if website_match:
            business["website"] = website_match.group(0)
        
        return business

    # === OSM DATA ===
    
    def search_osm_businesses(self, business_type: str, location: str) -> List[Dict]:
        """Search OpenStreetMap data"""
        if not self.geolocator or not self.osm_api:
            return []
        
        try:
            coords = self.geolocator.geocode(location)
            if not coords:
                return []
            
            lat, lon = coords.latitude, coords.longitude
            
            # Search multiple radii
            all_businesses = []
            for radius in [2000, 10000, 25000]:
                try:
                    query = self.build_osm_query(business_type, lat, lon, radius)
                    result = self.osm_api.query(query)
                    
                    for node in result.nodes:
                        business = self.process_osm_node(node)
                        if business.get('name'):
                            all_businesses.append(business)
                    
                    for way in result.ways:
                        business = self.process_osm_way(way)
                        if business.get('name'):
                            all_businesses.append(business)
                    
                    time.sleep(1)
                except:
                    continue
            
            return self.deduplicate_osm(all_businesses)
            
        except:
            return []

    def build_osm_query(self, business_type: str, lat: float, lon: float, radius: int) -> str:
        amenity_map = {
            "restaurant": "restaurant|fast_food|cafe|bar",
            "food": "restaurant|fast_food|cafe|bar|bakery",
            "retail": "shop",
            "medical": "hospital|clinic|pharmacy|veterinary",
            "hotel": "hotel|motel|guest_house",
            "beauty": "beauty_salon|hairdresser",
            "fitness": "gym|fitness_centre",
            "animal": "veterinary|animal_shelter"
        }
        
        amenities = "shop|restaurant|office"
        for key, value in amenity_map.items():
            if key.lower() in business_type.lower():
                amenities = value
                break
        
        return f"""
        [out:json][timeout:30];
        (
          node["amenity"~"{amenities}"](around:{radius},{lat},{lon});
          way["amenity"~"{amenities}"](around:{radius},{lat},{lon});
          node["shop"](around:{radius},{lat},{lon});
          way["shop"](around:{radius},{lat},{lon});
        );
        out center meta;
        """

    def process_osm_node(self, node) -> Dict:
        tags = node.tags
        return {
            "name": tags.get("name", ""),
            "address": self.build_osm_address(tags),
            "website": tags.get("website", ""),
            "phone": tags.get("phone", ""),
            "source": "osm"
        }

    def process_osm_way(self, way) -> Dict:
        tags = way.tags
        return {
            "name": tags.get("name", ""),
            "address": self.build_osm_address(tags),
            "website": tags.get("website", ""),
            "phone": tags.get("phone", ""),
            "source": "osm"
        }

    def build_osm_address(self, tags: Dict) -> str:
        parts = []
        for key in ["addr:housenumber", "addr:street", "addr:city", "addr:state"]:
            if key in tags:
                parts.append(tags[key])
        return ", ".join(parts)

    # === BUSINESS TYPE MATCHING ===
    
    def match_business_type(self, place_name: str, type_options: List[str]) -> str:
        if not type_options or not place_name:
            return ""
        
        # Semantic matching
        if self.semantic_model:
            try:
                embeddings = self.semantic_model.encode([t.lower() for t in type_options], convert_to_tensor=True)
                input_embedding = self.semantic_model.encode(place_name.lower(), convert_to_tensor=True)
                
                scores = util.cos_sim(input_embedding, embeddings)[0]
                max_score = scores.max().item()
                
                if max_score > 0.4:
                    return type_options[scores.argmax().item()]
            except:
                pass
        
        # OpenAI fallback
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            prompt = f"Which business type best matches '{place_name}'?\nOptions: {', '.join(type_options)}\nRespond with one option only."
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except:
            return ""

    # === DEDUPLICATION ===
    
    def deduplicate_simple(self, businesses: List[Dict]) -> List[Dict]:
        """Simple deduplication based on name and address"""
        seen = set()
        unique = []
        
        for business in businesses:
            name = business.get('name', '').lower().strip()
            address = business.get('address', '').lower().strip()
            key = (name, address)
            
            if key not in seen and name:
                seen.add(key)
                unique.append(business)
        
        return unique

    def deduplicate_osm(self, businesses: List[Dict]) -> List[Dict]:
        """Deduplicate OSM businesses"""
        seen_names = set()
        unique = []
        
        for business in businesses:
            name = business.get('name', '').lower().strip()
            if name and name not in seen_names:
                seen_names.add(name)
                unique.append(business)
        
        return unique

    # === MAIN PROCESSING ===
    
    def process_files(self, user_prompt: str, use_ner_workflow: bool = False) -> List[Dict]:
        """Main processing - scrape ALL businesses and return simple data"""
        print("🔍 Starting business scraping...")
        
        parsed_info = self.parse_location_and_type(user_prompt)
        if not parsed_info:
            raise ValueError("Could not extract business type and location from prompt.")
        
        business_type = parsed_info["business_type"]
        location = parsed_info["location"]
        
        # Get business group
        type_map = self.load_business_type_map()
        matched_group = self.identify_business_group_semantic(business_type)
        business_type_options = type_map.get(matched_group.lower(), [])
        
        print(f"Searching for: {business_type} in {location}")
        print(f"Business Group: {matched_group}")
        
        # Collect from all sources
        all_businesses = []
        
        # Google Places (unlimited with search variations)
        print("📍 Google Places...")
        google_places = self.search_all_places(business_type, location)
        if google_places:
            place_ids = [p.get("place_id") for p in google_places if p.get("place_id")]
            details = self.get_place_details_batch(place_ids)
            
            for place in google_places:
                place_id = place.get("place_id", "")
                detail = details.get(place_id, {})
                
                business = {
                    "name": place.get("name", ""),
                    "address": place.get("formatted_address", ""),
                    "website": detail.get("website", ""),
                    "business group": matched_group,
                    "matched business type": self.match_business_type(place.get("name", ""), business_type_options)
                }
                all_businesses.append(business)
        
        print(f"Google Places: {len(google_places)} businesses")
        
        # Web scraping (more extensive)
        print("🌐 Web directories...")
        web_businesses = self.scrape_web_directories(business_type, location)
        for business in web_businesses:
            standardized = {
                "name": business.get("name", ""),
                "address": business.get("address", ""),
                "website": business.get("website", ""),
                "business group": matched_group,
                "matched business type": self.match_business_type(business.get("name", ""), business_type_options)
            }
            all_businesses.append(standardized)
        
        print(f"Web scraping: {len(web_businesses)} businesses")
        
        # OSM data (multiple radii)
        print("🗺️ OpenStreetMap...")
        osm_businesses = self.search_osm_businesses(business_type, location)
        for business in osm_businesses:
            standardized = {
                "name": business.get("name", ""),
                "address": business.get("address", ""),
                "website": business.get("website", ""),
                "business group": matched_group,
                "matched business type": self.match_business_type(business.get("name", ""), business_type_options)
            }
            all_businesses.append(standardized)
        
        print(f"OpenStreetMap: {len(osm_businesses)} businesses")
        
        # Final deduplication
        final_businesses = self.deduplicate_simple(all_businesses)
        
        print(f"✅ Total unique businesses: {len(final_businesses)}")
        
        # Save results to Output folder
        if final_businesses:
            self.save_results(final_businesses, user_prompt)
        
        return final_businesses

    def save_results(self, businesses: List[Dict], user_prompt: str):
        """Save results to Google Drive Output folder"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(businesses)
            
            # Generate filename based on prompt
            safe_filename = "".join(c for c in user_prompt if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_filename = safe_filename.replace(' ', '_')[:50]
            filename = f"{safe_filename}_results.csv"
            
            # Save locally first
            local_path = f"/tmp/{filename}"
            df.to_csv(local_path, index=False)
            
            # Check for existing files and delete them
            existing_files = self.drive.ListFile({
                'q': f"'{self.output_folder_id}' in parents and trashed=false and title='{filename}'"
            }).GetList()
            
            for file_obj in existing_files:
                file_obj.Delete()
            
            # Upload to Google Drive
            file_to_upload = self.drive.CreateFile({
                "title": filename,
                "parents": [{"id": self.output_folder_id}]
            })
            file_to_upload.SetContentFile(local_path)
            file_to_upload.Upload()
            
            print(f"💾 Results saved to Output folder: {filename}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")

    def cleanup_resources(self):
        """Clean up resources"""
        for driver in self.drivers:
            try:
                driver.quit()
            except:
                pass
        
        for session in self.sessions:
            try:
                session.close()
            except:
                pass


# Main execution
if __name__ == "__main__":
    parser = PromptParser()
    
    try:
        businesses = parser.process_files("restaurants in Los Angeles")
        print(f"\nFound {len(businesses)} businesses")
        
        # Show first 3 results
        for i, business in enumerate(businesses[:3], 1):
            print(f"\n{i}. {business.get('name')}")
            print(f"   Address: {business.get('address')}")
            print(f"   Website: {business.get('website')}")
        
    finally:
        parser.cleanup_resources()