"""
emailClassifier.py - Email Classification Module
Extracts emails from business websites and classifies them for HR and Sales purposes
Integrates with main.py workflow
"""

import os
import requests
import time
import re
import json
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# AI components for email classification
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
    print("LangChain loaded for email classification")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available. Email classification disabled.")

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
    print("Selenium loaded for dynamic content scraping")
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Selenium not available. Will use requests only.")

# Configuration
OPENAI_API_KEY = "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"

class EmailClassifier:
    """Email extraction and classification system"""
    
    def __init__(self, enable_ai_classification: bool = True, max_workers: int = 5):
        self.enable_ai_classification = enable_ai_classification
        self.max_workers = max_workers
        self.ai_available = False
        
        print("Initializing Email Classifier...")
        
        # Initialize AI components
        if enable_ai_classification and LANGCHAIN_AVAILABLE:
            self.init_ai_components()
        
        print(f"Email Classifier ready! AI: {'enabled' if self.ai_available else 'disabled'}")
    
    def init_ai_components(self):
        """Initialize AI components for email classification"""
        try:
            if not OPENAI_API_KEY or len(OPENAI_API_KEY) < 20:
                print("OpenAI API key not properly set")
                return
            
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            self.ai_available = True
            print("AI email classification enabled")
            
        except Exception as e:
            print(f"AI initialization failed: {e}")
    
    def extract_emails(self, text: str, soup=None) -> List[str]:
        """Extract email addresses from text and HTML"""
        pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        emails = set(re.findall(pattern, text))
        
        if soup:
            # Check mailto links
            for a in soup.find_all("a", href=True):
                href = a['href'].lower()
                if href.startswith("mailto:"):
                    email = href[7:].split("?")[0].strip()
                    if re.match(pattern, email):
                        emails.add(email)
                
                # Check visible text in links
                visible = a.get_text(strip=True)
                if re.match(pattern, visible):
                    emails.add(visible)
        
        # Filter out common false positives
        filtered_emails = []
        for email in emails:
            email_lower = email.lower()
            # Skip obvious placeholders or invalid emails
            if not any(skip in email_lower for skip in [
                'example.com', 'test.com', 'placeholder', 'domain.com',
                'yoursite.com', 'company.com', 'business.com'
            ]):
                filtered_emails.append(email)
        
        return list(filtered_emails)
    
    def extract_social_links(self, soup) -> Dict[str, List[str]]:
        """Extract social media links from HTML"""
        links = {"facebook": [], "instagram": [], "tiktok": [], "linkedin": []}
        
        if not soup:
            return links
        
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "facebook.com" in href and href not in links["facebook"]:
                links["facebook"].append(href)
            elif "instagram.com" in href and href not in links["instagram"]:
                links["instagram"].append(href)
            elif "tiktok.com" in href and href not in links["tiktok"]:
                links["tiktok"].append(href)
            elif "linkedin.com" in href and href not in links["linkedin"]:
                links["linkedin"].append(href)
        
        return links
    
    def scrape_website(self, url: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """Scrape website for emails and social links"""
        if not url:
            return [], {}
        
        try:
            # Clean and validate URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Try requests first (faster)
            try:
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                html = response.text
                soup = BeautifulSoup(html, "html.parser")
                
            except Exception as requests_error:
                # Fallback to Selenium for dynamic content
                if not SELENIUM_AVAILABLE:
                    print(f"  Requests failed and Selenium not available for {url}")
                    return [], {}
                
                print(f"  Trying Selenium for {url}")
                options = Options()
                options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                
                driver = webdriver.Chrome(options=options)
                try:
                    driver.get(url)
                    time.sleep(4)  # Wait for dynamic content
                    html = driver.page_source
                    soup = BeautifulSoup(html, "html.parser")
                finally:
                    driver.quit()
            
            # Extract data
            emails = self.extract_emails(html, soup)
            socials = self.extract_social_links(soup)
            
            return emails, socials
            
        except Exception as e:
            print(f"  Scraping failed for {url}: {e}")
            return [], {}
    
    def classify_emails(self, business_name: str, emails: List[str]) -> Dict[str, Dict[str, str]]:
        """Classify emails into HR and Sales categories using AI"""
        if not emails or not self.ai_available:
            return {}
        
        prompt_text = f"""
You are an email classification assistant.

The company is {business_name}. Here is a list of emails associated with it:
{', '.join(emails)}.

Classify each email into two categories based only on its text (ignore case):

1. HR Category:
   - BEST: Joinus@, HR@, TA@, TalentAquisition@, HumanResources@, Apply@, Jobs@, Careers@, Hiring@, Recruiting@, Recruitment@, Talent@, TalentTeam@, People@, PeopleOps@, Applications@, Submit@, CV@, Resume@, WorkWithUs@, JobsHR@, HRTeam@, Recruiters@, TalentMgmt@, HiringTeam@, TeamHR@, Opportunities@, Team@, Staffing@, Onboarding@
   - BETTER: info@, contact@, hello@, admin@, mail@, store@, office@
   - EXCLUDE: support@, sales@, billing@, press@, accessibility@, estimates@, guestservices@

2. Sales Category:
   - BEST: sales@, orders@
   - BETTER: info@, contact@, hello@, admin@, mail@, store@, office@
   - EXCLUDE: support@, invoices@, billing@, customerservice@, guestservices@, press@, feedback@, accessibility@

Return each email on a new line in the format:
email@example.com: HR_CATEGORY, SALES_CATEGORY

Use only the emails from the list.
Do not explain anything or generate new emails.
""".strip()
        
        try:
            prompt = ChatPromptTemplate.from_template(prompt_text)
            chain = prompt | self.llm
            
            result = chain.invoke({})
            classified = {}
            
            for line in result.content.strip().splitlines():
                if ":" in line:
                    email_part, label_part = line.strip().split(":", 1)
                    labels = [x.strip().upper() for x in label_part.split(",")]
                    if len(labels) == 2:
                        classified[email_part.strip()] = {
                            "hr": labels[0],
                            "sales": labels[1]
                        }
            
            return classified
            
        except Exception as e:
            print(f"  Email classification error for {business_name}: {e}")
            return {}
    
    def select_priority_email(self, classified: Dict[str, str]) -> Tuple[str, str]:
        """Select the highest priority email from classified emails"""
        if not classified:
            return "", ""
        
        # Priority: BEST > BETTER > anything else
        for email, category in classified.items():
            if category == "BEST":
                return email, "BEST"
        
        for email, category in classified.items():
            if category == "BETTER":
                return email, "BETTER"
        
        # If no BEST or BETTER, return first available
        if classified:
            first_email = next(iter(classified.keys()))
            return first_email, classified[first_email]
        
        return "", ""
    
    def process_single_business(self, business: Dict) -> Dict:
        """Process a single business for email classification"""
        try:
            business_name = business.get('name', business.get('business_name', 'Unknown'))
            website = business.get('website', '')
            
            print(f"  Processing emails for: {business_name}")
            
            # Scrape website for emails and social links
            emails, socials = self.scrape_website(website) if website else ([], {})
            
            print(f"    Found {len(emails)} emails, {sum(len(links) for links in socials.values())} social links")
            
            # Classify emails using AI
            classified = {}
            hr_email, hr_category = "", ""
            sales_email, sales_category = "", ""
            
            if emails and self.ai_available:
                classified = self.classify_emails(business_name, emails)
                print(f"    Classified {len(classified)} emails")
                
                # Extract best HR and Sales emails
                hr_classified = {e: v["hr"] for e, v in classified.items()}
                sales_classified = {e: v["sales"] for e, v in classified.items()}
                
                hr_email, hr_category = self.select_priority_email(hr_classified)
                sales_email, sales_category = self.select_priority_email(sales_classified)
            
            # Add email data to business
            enhanced_business = business.copy()
            enhanced_business.update({
                'email_list': ', '.join(emails),
                'hr_email': hr_email,
                'hr_email_category': hr_category,
                'sales_email': sales_email,
                'sales_email_category': sales_category,
                'facebook_links': ', '.join(socials.get('facebook', [])),
                'instagram_links': ', '.join(socials.get('instagram', [])),
                'tiktok_links': ', '.join(socials.get('tiktok', [])),
                'linkedin_links': ', '.join(socials.get('linkedin', [])),
                'total_emails_found': len(emails),
                'total_social_links': sum(len(links) for links in socials.values())
            })
            
            return enhanced_business
            
        except Exception as e:
            print(f"  Error processing {business.get('name', 'Unknown')}: {e}")
            # Return original business with empty email fields
            enhanced_business = business.copy()
            enhanced_business.update({
                'email_list': '',
                'hr_email': '',
                'hr_email_category': '',
                'sales_email': '',
                'sales_email_category': '',
                'facebook_links': '',
                'instagram_links': '',
                'tiktok_links': '',
                'linkedin_links': '',
                'total_emails_found': 0,
                'total_social_links': 0
            })
            return enhanced_business
    
    def add_email_classification_to_businesses(self, businesses_data: List[Dict]) -> List[Dict]:
        """
        Main function to add email classification to business data
        """
        print(f"\nAdding email classification to {len(businesses_data)} businesses...")
        
        enhanced_businesses = []
        successful_extractions = 0
        total_emails_found = 0
        
        # Process businesses in parallel for faster execution
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_business = {
                executor.submit(self.process_single_business, business): business 
                for business in businesses_data
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_business):
                try:
                    enhanced_business = future.result()
                    enhanced_businesses.append(enhanced_business)
                    
                    # Track statistics
                    emails_count = enhanced_business.get('total_emails_found', 0)
                    if emails_count > 0:
                        successful_extractions += 1
                        total_emails_found += emails_count
                        
                except Exception as e:
                    original_business = future_to_business[future]
                    print(f"  Failed to process {original_business.get('name', 'Unknown')}: {e}")
                    # Add original business with empty fields
                    enhanced_businesses.append(self.process_single_business(original_business))
        
        # Sort to maintain original order (optional)
        # enhanced_businesses.sort(key=lambda x: businesses_data.index(next(b for b in businesses_data if b.get('name') == x.get('name'))))
        
        print(f"Email classification complete!")
        print(f"Successfully extracted emails from {successful_extractions}/{len(businesses_data)} businesses")
        print(f"Total emails found: {total_emails_found}")
        
        return enhanced_businesses


# ================================
# MAIN FUNCTION FOR main.py INTEGRATION
# ================================

def add_email_classification(businesses_data: List[Dict], 
                           enable_ai: bool = True,
                           max_workers: int = 5) -> List[Dict]:
    """
    Main function called by main.py workflow
    
    Args:
        businesses_data: List of business dictionaries from previous steps
        enable_ai: Whether to enable AI-powered email classification
        max_workers: Number of parallel threads for processing
    
    Returns:
        List of enhanced business dictionaries with email classification data added
    """
    try:
        email_classifier = EmailClassifier(
            enable_ai_classification=enable_ai,
            max_workers=max_workers
        )
        
        return email_classifier.add_email_classification_to_businesses(businesses_data)
        
    except Exception as e:
        print(f"Error in email classifier: {e}")
        # Return original data with empty email fields to avoid breaking pipeline
        for business in businesses_data:
            business.update({
                'email_list': '',
                'hr_email': '',
                'hr_email_category': '',
                'sales_email': '',
                'sales_email_category': '',
                'facebook_links': '',
                'instagram_links': '',
                'tiktok_links': '',
                'linkedin_links': '',
                'total_emails_found': 0,
                'total_social_links': 0
            })
        return businesses_data


# ================================
# UTILITY FUNCTIONS
# ================================

def test_email_classifier():
    """Test function for the email classifier"""
    print("Testing Email Classifier...")
    
    # Test with sample data
    sample_businesses = [
        {
            'name': 'Test Restaurant',
            'website': 'https://example-restaurant.com',
            'address': '123 Main St, Los Angeles, CA',
            'phone number': '(555) 123-4567'
        }
    ]
    
    enhanced_businesses = add_email_classification(sample_businesses)
    
    print("\nTest Results:")
    for business in enhanced_businesses:
        print(f"Business: {business['name']}")
        print(f"  Emails found: {business.get('total_emails_found', 0)}")
        print(f"  HR Email: {business.get('hr_email', 'None')}")
        print(f"  Sales Email: {business.get('sales_email', 'None')}")
        print(f"  Social Links: {business.get('total_social_links', 0)}")


if __name__ == "__main__":
    # Run test when file is executed directly
    test_email_classifier()