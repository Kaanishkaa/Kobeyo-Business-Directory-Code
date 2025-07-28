"""
careersPage.py - Career Page Extraction Module
Extracts career information and job postings from business websites
Integrates with main.py workflow
"""

import os
import requests
import time
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse

# Try to import Selenium for dynamic content
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
    print("Selenium loaded for careers page scraping")
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Selenium not available. Will use requests only.")


class CareersPageExtractor:
    """Career page extraction and analysis system"""
    
    def __init__(self, max_workers: int = 5, timeout: int = 10):
        self.max_workers = max_workers
        self.timeout = timeout
        
        print("Initializing Careers Page Extractor...")
        
        # Common career page URL patterns
        self.career_url_patterns = [
            'careers', 'jobs', 'hiring', 'employment', 'work-with-us', 
            'join-us', 'opportunities', 'job-openings', 'work', 'team',
            'apply', 'positions', 'openings', 'join-our-team'
        ]
        
        # Job-related keywords to identify relevant content
        self.job_keywords = [
            'job', 'position', 'role', 'career', 'employment', 'hiring',
            'apply', 'candidate', 'opening', 'opportunity', 'vacancy',
            'full-time', 'part-time', 'remote', 'salary', 'benefits'
        ]
        
        print("Careers Page Extractor ready!")
    
    def find_career_pages(self, base_url: str) -> List[str]:
        """Find potential career page URLs from a website"""
        if not base_url:
            return []
        
        career_urls = []
        
        try:
            # Clean and validate URL
            if not base_url.startswith(('http://', 'https://')):
                base_url = 'https://' + base_url
            
            # Try direct career URL patterns first
            parsed_url = urlparse(base_url)
            base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            for pattern in self.career_url_patterns:
                potential_urls = [
                    f"{base_domain}/{pattern}",
                    f"{base_domain}/{pattern}/",
                    f"{base_domain}/pages/{pattern}",
                    f"{base_url.rstrip('/')}/{pattern}",
                ]
                career_urls.extend(potential_urls)
            
            # Try to scrape main page for career links
            try:
                response = requests.get(base_url, timeout=self.timeout, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for links containing career keywords
                for link in soup.find_all('a', href=True):
                    href = link['href'].lower()
                    link_text = link.get_text(strip=True).lower()
                    
                    # Check if link or text contains career keywords
                    if any(keyword in href or keyword in link_text for keyword in self.career_url_patterns):
                        full_url = urljoin(base_url, link['href'])
                        career_urls.append(full_url)
                        
            except Exception:
                pass
                
        except Exception:
            pass
        
        # Remove duplicates and limit results
        return list(set(career_urls))[:10]
    
    def scrape_career_page(self, url: str) -> Dict[str, any]:
        """Scrape a single career page for job information"""
        career_data = {
            'careers_page_url': url,
            'job_postings': [],
            'career_info': '',
            'hiring_status': 'unknown',
            'total_jobs_found': 0
        }
        
        try:
            # Try requests first
            try:
                response = requests.get(url, timeout=self.timeout, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                else:
                    return career_data
                    
            except Exception:
                # Fallback to Selenium if available
                if SELENIUM_AVAILABLE:
                    return self.scrape_with_selenium(url)
                else:
                    return career_data
            
            # Extract job postings
            job_postings = self.extract_job_postings(soup)
            career_data['job_postings'] = job_postings
            career_data['total_jobs_found'] = len(job_postings)
            
            # Extract general career information
            career_info = self.extract_career_info(soup)
            career_data['career_info'] = career_info
            
            # Determine hiring status
            career_data['hiring_status'] = self.determine_hiring_status(soup, job_postings)
            
        except Exception as e:
            print(f"  Error scraping career page {url}: {e}")
        
        return career_data
    
    def scrape_with_selenium(self, url: str) -> Dict[str, any]:
        """Scrape career page using Selenium for dynamic content"""
        career_data = {
            'careers_page_url': url,
            'job_postings': [],
            'career_info': '',
            'hiring_status': 'unknown',
            'total_jobs_found': 0
        }
        
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
            driver = webdriver.Chrome(options=options)
            driver.get(url)
            time.sleep(3)  # Wait for dynamic content
            
            # Scroll to load more content
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            driver.quit()
            
            # Extract data same as requests method
            job_postings = self.extract_job_postings(soup)
            career_data['job_postings'] = job_postings
            career_data['total_jobs_found'] = len(job_postings)
            career_data['career_info'] = self.extract_career_info(soup)
            career_data['hiring_status'] = self.determine_hiring_status(soup, job_postings)
            
        except Exception as e:
            print(f"  Selenium error for {url}: {e}")
            try:
                driver.quit()
            except:
                pass
        
        return career_data
    
    def extract_job_postings(self, soup: BeautifulSoup) -> List[str]:
        """Extract job postings from the page"""
        job_postings = []
        
        # Look for common job posting patterns
        job_selectors = [
            'div[class*="job"]', 'div[class*="position"]', 'div[class*="opening"]',
            'li[class*="job"]', 'li[class*="position"]', 'li[class*="opening"]',
            'h3', 'h4', 'h5'  # Job titles often in headings
        ]
        
        for selector in job_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                
                # Check if text contains job-related keywords
                if any(keyword in text.lower() for keyword in self.job_keywords):
                    # Clean and validate job posting
                    if len(text) > 10 and len(text) < 200:  # Reasonable length
                        job_postings.append(text)
        
        # Also look for text that might be job titles
        all_text = soup.get_text()
        lines = [line.strip() for line in all_text.split('\n') if line.strip()]
        
        for line in lines:
            if (any(keyword in line.lower() for keyword in ['manager', 'assistant', 'specialist', 'coordinator', 'director', 'analyst']) and
                len(line) > 10 and len(line) < 100):
                job_postings.append(line)
        
        # Remove duplicates and limit results
        unique_jobs = []
        seen = set()
        for job in job_postings:
            job_lower = job.lower().strip()
            if job_lower not in seen and len(job.split()) > 1:
                seen.add(job_lower)
                unique_jobs.append(job)
        
        return unique_jobs[:20]  # Limit to 20 job postings
    
    def extract_career_info(self, soup: BeautifulSoup) -> str:
        """Extract general career information from the page"""
        career_info_parts = []
        
        # Look for career-related paragraphs
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            text = p.get_text(strip=True)
            if (any(keyword in text.lower() for keyword in self.job_keywords + ['company', 'team', 'culture', 'benefits']) and
                len(text) > 50):
                career_info_parts.append(text)
        
        # Look for career sections
        career_sections = soup.find_all(['div', 'section'], class_=re.compile(r'career|job|hiring', re.I))
        for section in career_sections:
            text = section.get_text(strip=True)
            if len(text) > 50:
                career_info_parts.append(text)
        
        # Combine and clean career info
        full_info = ' '.join(career_info_parts)
        
        # Limit length and clean up
        if len(full_info) > 1000:
            full_info = full_info[:1000] + "..."
        
        return full_info.strip()
    
    def determine_hiring_status(self, soup: BeautifulSoup, job_postings: List[str]) -> str:
        """Determine if company is actively hiring"""
        page_text = soup.get_text().lower()
        
        # Positive hiring indicators
        hiring_indicators = [
            'now hiring', 'we are hiring', 'join our team', 'apply now',
            'open positions', 'current openings', 'available positions'
        ]
        
        # Negative hiring indicators
        no_hiring_indicators = [
            'no positions available', 'no current openings', 'not hiring',
            'positions filled', 'no vacancies'
        ]
        
        # Check for positive indicators
        if any(indicator in page_text for indicator in hiring_indicators):
            return 'active'
        
        # Check for negative indicators
        if any(indicator in page_text for indicator in no_hiring_indicators):
            return 'not_hiring'
        
        # Check if there are job postings
        if len(job_postings) > 0:
            return 'active'
        
        # Check if page exists and has career-related content
        if any(keyword in page_text for keyword in self.job_keywords):
            return 'possible'
        
        return 'unknown'
    
    def process_single_business(self, business: Dict) -> Dict:
        """Process a single business for career information"""
        try:
            business_name = business.get('name', business.get('business_name', 'Unknown'))
            website = business.get('website', '')
            
            print(f"  Processing careers for: {business_name}")
            
            if not website:
                print(f"    No website available")
                return self.add_empty_career_fields(business)
            
            # Find career pages
            career_urls = self.find_career_pages(website)
            
            if not career_urls:
                print(f"    No career pages found")
                return self.add_empty_career_fields(business)
            
            print(f"    Found {len(career_urls)} potential career pages")
            
            # Try to scrape the most promising career URLs
            best_career_data = None
            max_jobs = 0
            
            for url in career_urls[:3]:  # Try up to 3 URLs
                try:
                    # Quick check if URL is accessible
                    response = requests.head(url, timeout=5)
                    if response.status_code == 200:
                        career_data = self.scrape_career_page(url)
                        
                        # Keep the data with most job postings
                        if career_data['total_jobs_found'] > max_jobs:
                            max_jobs = career_data['total_jobs_found']
                            best_career_data = career_data
                            
                except Exception:
                    continue
            
            # Add career data to business
            enhanced_business = business.copy()
            
            if best_career_data:
                enhanced_business.update({
                    'careers_page_url': best_career_data['careers_page_url'],
                    'job_postings': ', '.join(best_career_data['job_postings']),
                    'career_info': best_career_data['career_info'],
                    'hiring_status': best_career_data['hiring_status'],
                    'total_jobs_found': best_career_data['total_jobs_found']
                })
                print(f"    Found {best_career_data['total_jobs_found']} job postings")
            else:
                enhanced_business = self.add_empty_career_fields(enhanced_business)
                print(f"    No career data extracted")
            
            return enhanced_business
            
        except Exception as e:
            print(f"  Error processing {business.get('name', 'Unknown')}: {e}")
            return self.add_empty_career_fields(business)
    
    def add_empty_career_fields(self, business: Dict) -> Dict:
        """Add empty career fields to business"""
        business.update({
            'careers_page_url': '',
            'job_postings': '',
            'career_info': '',
            'hiring_status': 'unknown',
            'total_jobs_found': 0
        })
        return business
    
    def add_careers_data_to_businesses(self, businesses_data: List[Dict]) -> List[Dict]:
        """
        Main function to add career data to business data
        """
        print(f"\nAdding career data to {len(businesses_data)} businesses...")
        
        enhanced_businesses = []
        successful_extractions = 0
        total_jobs_found = 0
        
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
                    jobs_count = enhanced_business.get('total_jobs_found', 0)
                    if jobs_count > 0:
                        successful_extractions += 1
                        total_jobs_found += jobs_count
                        
                except Exception as e:
                    original_business = future_to_business[future]
                    print(f"  Failed to process {original_business.get('name', 'Unknown')}: {e}")
                    # Add original business with empty fields
                    enhanced_businesses.append(self.add_empty_career_fields(original_business))
        
        print(f"Career extraction complete!")
        print(f"Successfully found career data for {successful_extractions}/{len(businesses_data)} businesses")
        print(f"Total job postings found: {total_jobs_found}")
        
        return enhanced_businesses


# ================================
# MAIN FUNCTION FOR main.py INTEGRATION
# ================================

def add_careers_data(businesses_data: List[Dict], 
                    max_workers: int = 5,
                    timeout: int = 10) -> List[Dict]:
    """
    Main function called by main.py workflow
    
    Args:
        businesses_data: List of business dictionaries from previous steps
        max_workers: Number of parallel threads for processing
        timeout: Request timeout in seconds
    
    Returns:
        List of enhanced business dictionaries with career data added
    """
    try:
        careers_extractor = CareersPageExtractor(
            max_workers=max_workers,
            timeout=timeout
        )
        
        return careers_extractor.add_careers_data_to_businesses(businesses_data)
        
    except Exception as e:
        print(f"Error in careers extractor: {e}")
        # Return original data with empty career fields to avoid breaking pipeline
        for business in businesses_data:
            business.update({
                'careers_page_url': '',
                'job_postings': '',
                'career_info': '',
                'hiring_status': 'unknown',
                'total_jobs_found': 0
            })
        return businesses_data


# ================================
# UTILITY FUNCTIONS
# ================================

def test_careers_extractor():
    """Test function for the careers extractor"""
    print("Testing Careers Extractor...")
    
    # Test with sample data
    sample_businesses = [
        {
            'name': 'Test Company',
            'website': 'https://example-company.com',
            'address': '123 Main St, Los Angeles, CA'
        }
    ]
    
    enhanced_businesses = add_careers_data(sample_businesses)
    
    print("\nTest Results:")
    for business in enhanced_businesses:
        print(f"Business: {business['name']}")
        print(f"  Jobs found: {business.get('total_jobs_found', 0)}")
        print(f"  Hiring status: {business.get('hiring_status', 'Unknown')}")
        print(f"  Career page: {business.get('careers_page_url', 'None')}")
        if business.get('job_postings'):
            print(f"  Sample jobs: {business.get('job_postings', 'None')[:100]}...")


if __name__ == "__main__":
    # Run test when file is executed directly
    test_careers_extractor()