#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import time
import random
import json
import pandas as pd
import requests
import threading
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Constants
GOOGLE_API_KEY = ""
TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
PLACE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
SEARCH_TAGS = ["restaurant"]
SEARCH_LOCATIONS = ["Los Angeles"]
MAX_PAGES_PER_SITE = 15
CAREERS_MAX_PAGES = 5
THREAD_POOL_WORKERS = 5
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
]


# In[38]:


# Third-party job sites to exclude
THIRD_PARTY_JOB_SITES = [
    "indeed.com",
    "ziprecruiter.com",
    "linkedin.com/jobs",
    "monster.com",
    "glassdoor.com",
    "careerbuilder.com",
    "simplyhired.com",
    "dice.com",
    "flexjobs.com",
    "upwork.com",
    "freelancer.com",
    "fiverr.com",
    "snagajob.com",
    "workday.com",
    "bamboohr.com",
    "greenhouse.io",
]

# URL Shortening Configuration
MAX_URL_LENGTH = 80
ELLIPSIS = "..."

# Setup headless Chrome for Selenium
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=chrome_options)


# In[39]:


# ================= URL PROCESSING FUNCTIONS =================


def shorten_url(url, max_length=MAX_URL_LENGTH):
    """Shorten a URL if it exceeds the maximum length"""
    if len(url) <= max_length:
        return url

    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        scheme = parsed.scheme
        base_url = f"{scheme}://{domain}"

        if len(base_url) >= max_length - len(ELLIPSIS):
            return base_url[: max_length - len(ELLIPSIS)] + ELLIPSIS

        remaining_space = max_length - len(base_url) - len(ELLIPSIS)
        if remaining_space > 0 and parsed.path:
            path_part = parsed.path[:remaining_space]
            return f"{base_url}{path_part}{ELLIPSIS}"
        else:
            return base_url

    except Exception:
        return url[: max_length - len(ELLIPSIS)] + ELLIPSIS


def check_url_status(url, timeout=5):
    """Check if a URL is working/accessible (returns True if working)"""
    if not url or url == "None found":
        return False

    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        response = requests.head(
            url, headers=headers, timeout=timeout, allow_redirects=True
        )
        return response.status_code < 400
    except:
        try:
            response = requests.get(
                url, headers=headers, timeout=timeout, allow_redirects=True
            )
            return response.status_code < 400
        except:
            return False


def process_urls(url_string):
    """Process and shorten URLs, filter out broken ones"""
    if not url_string or url_string == "None found":
        return url_string

    urls = [url.strip() for url in url_string.split(";")]
    working_urls = []

    for url in urls:
        if url and check_url_status(url):
            working_urls.append(shorten_url(url))

    return "; ".join(working_urls) if working_urls else "None found"


def is_third_party_job_site(url):
    """Check if URL is from a third-party job site"""
    return any(site in url.lower() for site in THIRD_PARTY_JOB_SITES)


# In[40]:


# ================= GOOGLE MAPS API FUNCTIONS =================


def search_places(query, location, max_results=50):
    all_place_ids = []
    all_names = []
    all_addresses = []

    params = {
        "query": f"{query} in {location}",
        "key": GOOGLE_API_KEY,
    }

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    next_page_token = None

    while len(all_place_ids) < max_results:
        if next_page_token:
            params["pagetoken"] = next_page_token
            time.sleep(2)  # Required delay for next_page_token to become active

        response = requests.get(url, params=params)
        data = response.json()

        for result in data.get("results", []):
            all_place_ids.append(result["place_id"])
            all_names.append(result["name"])
            all_addresses.append(result.get("formatted_address", ""))

            if len(all_place_ids) >= max_results:
                break

        next_page_token = data.get("next_page_token")
        if not next_page_token:
            break

    return all_place_ids, all_names, all_addresses


def get_place_details(place_id):
    """Get detailed information about a place"""
    params = {
        "place_id": place_id,
        "fields": "name,formatted_address,website",
        "key": GOOGLE_API_KEY,
    }
    response = requests.get(PLACE_DETAILS_URL, params=params)
    if response.status_code != 200:
        return None
    return response.json().get("result", {})


# In[41]:


def categorize_sales_emails(email_list):
    best_keywords = ["sales@", "orders@"]

    better_keywords = [
        "info@",
        "contact@",
        "contactus@",
        "hello@",
        "admin@",
        "mail@",
        "support@",
        "store@",
        "clinic@",
        "hola@",  # generic or store-specific emails
    ]

    exclude_keywords = ["invoices@", "billing@", "guestservices@", "estimates@"]

    categorized = {"BEST": [], "BETTER": [], "EXCLUDE": []}

    for email in email_list:
        e = email.lower()

        if any(x in e for x in exclude_keywords):
            categorized["EXCLUDE"].append(e)
        elif any(x in e for x in best_keywords):
            categorized["BEST"].append(e)
        elif any(x in e for x in better_keywords):
            categorized["BETTER"].append(e)
        elif len(e.split("@")[0]) <= 4 or re.match(
            r".*@(gmail|yahoo|hotmail|outlook)\.", e
        ):
            categorized["EXCLUDE"].append(e)
        else:
            categorized["BETTER"].append(e)  # fallback if unknown

    return categorized


# In[42]:


def extract_emails(text, soup=None):
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    emails = set(re.findall(pattern, text))

    if soup:
        for a in soup.find_all("a", href=True):
            href = a["href"].lower()
            if href.startswith("mailto:"):
                email = href[7:].split("?")[0].strip()
                if re.match(pattern, email):
                    emails.add(email)
            visible = a.get_text(strip=True)
            if re.match(pattern, visible):
                emails.add(visible)

    return list(emails)


def categorize_emails(email_list):
    best_keywords = [
        "careers@",
        "joinus@",
        "hr@",
        "ta@",
        "talentaquisition@",
        "humanresources@",
        "apply@",
        "jobs@",
        "hiring@",
        "recruiting@",
        "recruitment@",
        "talent@",
        "talentteam@",
        "people@",
        "peopleops@",
        "applications@",
        "submit@",
        "cv@",
        "resume@",
        "workwithus@",
        "jobshr@",
        "hrteam@",
        "recruiters@",
        "talentmgmt@",
        "hiringteam@",
        "teamhr@",
        "opportunities@",
        "team@",
        "staffing@",
        "onboarding@",
    ]

    better_keywords = [
        "info@",
        "contact@",
        "contactus@",
        "hello@",
        "admin@",
        "mail@",
        "hola@",
        "store@",
        "clinic@",
        "office@",
    ]

    exclude_keywords = [
        "support@",
        "invoices@",
        "billing@",
        "guestservices@",
        "estimates@",
        "sales@",
        "orders@",
        "customerservice@",
    ]

    categorized = {"BEST": [], "BETTER": [], "EXCLUDE": []}

    for email in email_list:
        e = email.lower()

        if any(x in e for x in exclude_keywords):
            categorized["EXCLUDE"].append(e)
        elif any(x in e for x in best_keywords):
            categorized["BEST"].append(e)
        elif any(x in e for x in better_keywords):
            categorized["BETTER"].append(e)
        elif (
            len(e.split("@")[0]) <= 4
        ):  # <== only now apply short-length heuristic (very conservative)
            categorized["EXCLUDE"].append(e)
        else:
            categorized["BETTER"].append(e)  # fallback if unknown, but likely org email

    return categorized


# In[43]:


# ================= PAGE DETECTION FUNCTIONS =================


def detect_careers_page(url, text, soup):
    """Detect if this is a careers/jobs page and if it's internal"""
    if is_third_party_job_site(url):
        return False, ""

    careers_indicators = [
        "career",
        "careers",
        "job",
        "jobs",
        "employment",
        "hiring",
        "positions",
        "join our team",
        "work with us",
        "apply now",
        "open positions",
    ]

    url_lower = url.lower()
    text_lower = text.lower()

    # Check URL path
    url_has_careers = any(indicator in url_lower for indicator in careers_indicators)

    # Check page content
    content_has_careers = any(
        indicator in text_lower for indicator in careers_indicators
    )

    # Look for application forms
    has_application_form = False
    if soup:
        forms = soup.find_all("form")
        for form in forms:
            form_text = form.get_text().lower()
            if any(
                word in form_text
                for word in ["apply", "application", "resume", "cv", "position"]
            ):
                has_application_form = True
                break

    if url_has_careers or (content_has_careers and has_application_form):
        return True, url

    return False, ""


def detect_products_services_page(url, text, soup):
    """Detect pages showing products, services, or industries served"""
    if not text:
        return False, ""

    # Keywords that indicate products/services pages
    product_service_indicators = [
        "products",
        "services",
        "solutions",
        "offerings",
        "what we do",
        "industries",
        "sectors",
        "specialties",
        "capabilities",
        "expertise",
        "portfolio",
        "catalog",
        "menu",
        "pricing",
        "packages",
    ]

    url_lower = url.lower()
    text_lower = text.lower()

    # Check URL path
    url_indicates_products = any(
        indicator in url_lower for indicator in product_service_indicators
    )

    # Check if content has substantial product/service information
    content_score = sum(
        1 for indicator in product_service_indicators if indicator in text_lower
    )

    # Look for structured content (lists, grids, etc.)
    has_structured_content = False
    if soup:
        lists = soup.find_all(["ul", "ol", "div"])
        for element in lists:
            element_text = element.get_text().lower()
            if any(
                indicator in element_text for indicator in product_service_indicators
            ):
                has_structured_content = True
                break

    if url_indicates_products or (content_score >= 2 and has_structured_content):
        return True, url

    return False, ""


# In[44]:


# ================= EXTRACT SOCIAL MEDIA LINKS =================


def extract_socials(soup):
    """Extract Instagram, Facebook, and X (Twitter) links"""
    social_links = {"Instagram": "", "Facebook": "", "X": ""}

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "instagram.com" in href and not social_links["Instagram"]:
            social_links["Instagram"] = href
        elif "facebook.com" in href and not social_links["Facebook"]:
            social_links["Facebook"] = href
        elif "twitter.com" in href or "x.com" in href:
            social_links["X"] = href

    return social_links


# In[45]:


# ================= WEB CRAWLING =================


def crawl_site_comprehensive(base_url, max_pages=15):
    """Comprehensive website crawling with categorized emails, social media, and Selenium fallback"""
    visited = set()
    queue = deque([base_url])
    emails = set()
    social_links = {"Instagram": "", "Facebook": "", "X": ""}
    careers_pages = set()
    products_services_pages = set()
    debug_info = ""

    domain = urlparse(base_url).netloc.replace("www.", "")
    pages_crawled = 0

    try:
        while queue and pages_crawled < max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            headers = {"User-Agent": random.choice(USER_AGENTS)}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(" ", strip=True)

            emails.update(extract_emails(text, soup))

            if not all(social_links.values()):
                new_socials = extract_socials(soup)
                for key in social_links:
                    if not social_links[key] and new_socials[key]:
                        social_links[key] = new_socials[key]

            is_careers, careers_url = detect_careers_page(url, text, soup)
            if is_careers:
                careers_pages.add(careers_url)

            is_products, products_url = detect_products_services_page(url, text, soup)
            if is_products:
                products_services_pages.add(products_url)

            for a in soup.find_all("a", href=True):
                new_url = urljoin(url, a["href"])
                parsed_url = urlparse(new_url)
                if domain in parsed_url.netloc and new_url not in visited:
                    queue.append(new_url)

            pages_crawled += 1

    except Exception as e:
        debug_info += f"Requests failed: {type(e).__name__}: {str(e).split(':')[0]}. "

    # Retry with Selenium
    if not emails or not all(social_links.values()):
        try:
            driver.set_page_load_timeout(30)
            driver.get(base_url)
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(" ", strip=True)

            new_emails = extract_emails(text, soup)
            emails.update(new_emails)

            new_socials = extract_socials(soup)
            for key in social_links:
                if not social_links[key] and new_socials[key]:
                    social_links[key] = new_socials[key]

            if not new_emails and not debug_info:
                debug_info = "Selenium used but still no emails found."

        except Exception as e:
            error_type = type(e).__name__
            short_message = str(e).split(":")[0][:100]
            debug_info += f"Selenium failed ({error_type}): {short_message.strip()}."

    if emails:
        debug_info = ""
    elif not debug_info:
        debug_info = "No emails found from requests or Selenium."

    categorized = categorize_emails(list(emails))
    sales_categorized = categorize_sales_emails(list(emails))

    return {
        "best_email": categorized["BEST"],
        "better_email": categorized["BETTER"],
        "excluded_email": categorized["EXCLUDE"],
        "sales_best": sales_categorized["BEST"],
        "sales_better": sales_categorized["BETTER"],
        "sales_exclude": sales_categorized["EXCLUDE"],
        "careers_pages": list(careers_pages),
        "products_services_pages": list(products_services_pages),
        "social_links": social_links,
        "debug_info": debug_info,
        "pages_crawled": pages_crawled,
    }


# In[46]:


# ================= BUSINESS PROCESSING =================


def process_business_comprehensive(place_id, name, fallback_address):
    """Process a business with categorized emails and links"""
    print(f"Processing business: {name}")

    details = get_place_details(place_id)
    if not details:
        return None

    website = details.get("website", "")
    address = details.get("formatted_address", fallback_address)

    result = {
        "Company Name": name,
        "Company Address": address,
        "Company Website": website,
        "Best Email": "",
        "Better Email": "",
        "Excluded Email": "",
        "Sales BEST": "",
        "Sales BETTER": "",
        "Sales EXCLUDE": "",
        "Company Careers Page": "",
        "Company Products/Services Page": "",
        "Instagram": "",
        "Facebook": "",
        "X": "",
    }

    if website:
        try:
            crawl_results = crawl_site_comprehensive(website, MAX_PAGES_PER_SITE)

            if crawl_results["best_email"]:
                result["Best Email"] = "; ".join(crawl_results["best_email"])

            if crawl_results["better_email"]:
                result["Better Email"] = "; ".join(crawl_results["better_email"])

            if crawl_results["excluded_email"]:
                result["Excluded Email"] = "; ".join(crawl_results["excluded_email"])

            if crawl_results["sales_best"]:
                result["Sales BEST"] = "; ".join(crawl_results["sales_best"])

            if crawl_results["sales_better"]:
                result["Sales BETTER"] = "; ".join(crawl_results["sales_better"])

            if crawl_results["sales_exclude"]:
                result["Sales EXCLUDE"] = "; ".join(crawl_results["sales_exclude"])

            if crawl_results["careers_pages"]:
                result["Company Careers Page"] = "; ".join(
                    crawl_results["careers_pages"]
                )

            if crawl_results["products_services_pages"]:
                result["Company Products/Services Page"] = "; ".join(
                    crawl_results["products_services_pages"]
                )

            socials = crawl_results["social_links"]
            result["Instagram"] = socials.get("Instagram", "")
            result["Facebook"] = socials.get("Facebook", "")
            result["X"] = socials.get("X", "")

        except Exception as e:
            print(f"Failed to crawl {website}: {e}")

    return result


# In[47]:


# ================= MAIN FUNCTION =================


def main():
    """Main execution function"""
    print("🚀 Starting comprehensive business data scraper")

    lock = threading.Lock()
    checkpoint_file = "business_directory.csv"

    # Define CSV columns
    csv_columns = [
        "Company Name",
        "Company Address",
        "Company Website",
        "HR Email",
        "Sales BEST",
        "Sales BETTER",
        "Sales EXCLUDE",
        "Company Careers Page",
        "Company Products/Services Page",
        "Instagram",
        "Facebook",
        "X",
    ]

    # Load existing data
    if os.path.exists(checkpoint_file):
        existing_df = pd.read_csv(checkpoint_file)
        print(f"📄 Loaded {len(existing_df)} existing records")
    else:
        existing_df = pd.DataFrame(columns=csv_columns)

    existing_data = (
        {row["Company Name"]: row for _, row in existing_df.iterrows()}
        if not existing_df.empty
        else {}
    )

    # crawl_times = []
    metrics = {
        "total_businesses": 0,
        "emails_found": 0,
        "best_emails": 0,
        "better_emails": 0,
        "excluded_emails": 0,
        "priority_email_used": 0,
        "social_found": 0,
        "all_socials_found": 0,
        "nothing_scraped": 0,
    }

    for location in SEARCH_LOCATIONS:
        for tag in SEARCH_TAGS:
            print(f"\n🔍 Searching for {tag} in {location}")
            place_ids, names, addresses = search_places(tag, location)
            print(f"📍 Found {len(place_ids)} businesses")

            with ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS) as executor:
                futures = {
                    executor.submit(
                        process_business_comprehensive, pid, name, addr
                    ): name
                    for pid, name, addr in zip(place_ids, names, addresses)
                    if name not in existing_data
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        with lock:
                            # Apply email priority logic
                            priority_email = ""
                            if (
                                result["Best Email"]
                                and result["Best Email"] != "None found"
                            ):
                                priority_email = result["Best Email"].split(";")[0]
                            elif (
                                result["Better Email"]
                                and result["Better Email"] != "None found"
                            ):
                                priority_email = result["Better Email"].split(";")[0]
                            elif result["Excluded Email"] and any(
                                "@" in e
                                and any(char.isalpha() for char in e.split("@")[0])
                                for e in result["Excluded Email"].split(";")
                            ):
                                for e in result["Excluded Email"].split(";"):
                                    if "@" in e and any(
                                        char.isalpha() for char in e.split("@")[0]
                                    ):
                                        priority_email = e
                                        break

                            result["HR Email"] = priority_email

                            # Clean URLs
                            for field in [
                                "Company Careers Page",
                                "Company Products/Services Page",
                            ]:
                                if result[field]:
                                    result[field] = process_urls(result[field])
                                # else:
                                #     result[field] = "None found"

                            # Fill blanks
                            # for field in ["Best Email", "Better Email", "Excluded Email", "Instagram", "Facebook", "X", "Priority Email"]:
                            #     if not result[field]:
                            #         result[field] = "None found"

                            # Update metrics
                            metrics["total_businesses"] += 1
                            if any(
                                result[f] != "None found"
                                for f in [
                                    "Best Email",
                                    "Better Email",
                                    "Excluded Email",
                                ]
                            ):
                                metrics["emails_found"] += 1
                            if result["Best Email"] != "None found":
                                metrics["best_emails"] += 1
                            if result["Better Email"] != "None found":
                                metrics["better_emails"] += 1
                            if result["Excluded Email"] != "None found":
                                metrics["excluded_emails"] += 1
                            if result["HR Email"] != "None found":
                                metrics["priority_email_used"] += 1
                            if any(
                                result[f] != "None found"
                                for f in ["Instagram", "Facebook", "X"]
                            ):
                                metrics["social_found"] += 1
                            if all(
                                result[f] != "None found"
                                for f in ["Instagram", "Facebook", "X"]
                            ):
                                metrics["all_socials_found"] += 1
                            if all(
                                result[f] == "None found"
                                for f in [
                                    "Best Email",
                                    "Better Email",
                                    "Excluded Email",
                                    "Instagram",
                                    "Facebook",
                                    "X",
                                ]
                            ):
                                metrics["nothing_scraped"] += 1

                            for col in ["Best Email", "Better Email", "Excluded Email"]:
                                if col in result:
                                    del result[col]
                            # Save result
                            existing_df = pd.concat(
                                [existing_df, pd.DataFrame([result])], ignore_index=True
                            )
                            existing_df.to_csv(checkpoint_file, index=False)
                            print(f"✅ Saved: {result['Company Name']}")

    # Report
    print("\n" + "=" * 50)
    print("📊 FINAL METRICS REPORT")
    print("=" * 50)

    total = metrics["total_businesses"]

    def pct(v):
        return f"{(v/total*100):.1f}%" if total else "0.0%"

    print(f"📈 Total businesses processed: {total}")
    print(
        f"🌐 Businesses with any emails: {metrics['emails_found']} ({pct(metrics['emails_found'])})"
    )
    print(
        f"👔 Businesses with BEST emails: {metrics['best_emails']} ({pct(metrics['best_emails'])})"
    )
    print(
        f"📧 Businesses with BETTER emails: {metrics['better_emails']} ({pct(metrics['better_emails'])})"
    )
    print(
        f"🚫 Businesses with EXCLUDED emails: {metrics['excluded_emails']} ({pct(metrics['excluded_emails'])})"
    )
    print(
        f"🎯 Businesses where Priority Email used: {metrics['priority_email_used']} ({pct(metrics['priority_email_used'])})"
    )
    print(
        f"📱 Businesses with any social media: {metrics['social_found']} ({pct(metrics['social_found'])})"
    )
    print(
        f"✔️ Businesses with all 3 socials: {metrics['all_socials_found']} ({pct(metrics['all_socials_found'])})"
    )
    print(
        f"❌ Businesses with no contact info: {metrics['nothing_scraped']} ({pct(metrics['nothing_scraped'])})"
    )
    print("🎉 Scraping completed successfully!")
    print("=" * 50)


# In[48]:


# ================= CODE DRIVER METHOD =================

if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            driver.quit()
            print("🔧 Selenium driver closed successfully")
        except Exception as e:
            print(
                f"⚠️ Error while closing Selenium driver: {type(e).__name__} - {str(e)}"
            )
