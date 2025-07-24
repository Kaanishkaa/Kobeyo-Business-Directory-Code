import asyncio
import pandas as pd
import requests
import json
from typing import List, Dict
import time
import os

# Import the full crawler
from email_crawler import EmailCareerWebCrawler, CrawlResult

class CrawlAndClassifySystem:
    """
    Integrated system that:
    1. Crawls websites to extract emails
    2. Classifies emails as HR/Sales using the GraphRAG API
    3. Exports enriched results
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000", csv_file: str = None):
        self.api_base_url = api_base_url
        self.crawler = EmailCareerWebCrawler(csv_file)
        
    def classify_emails_batch(self, emails: List[str]) -> Dict:
        """Classify a batch of emails using the API"""
        if not emails:
            return {"results": [], "summary": {}}
            
        try:
            response = requests.post(
                f"{self.api_base_url}/classify/batch",
                json={"emails": list(emails)}
            )
            return response.json()
        except Exception as e:
            print(f"Error classifying emails: {e}")
            return {"results": [], "summary": {}}
    
    async def process_website_with_classification(self, website: str, index: int = 1, total: int = 1) -> Dict:
        """Crawl a website and classify its emails"""
        print(f"\n[{index}/{total}] Processing: {website}")
        
        # Step 1: Crawl website
        crawl_result = await self.crawler.process_single_website(website, index, total)
        
        # Step 2: Classify emails if any found
        classification_results = {"results": [], "summary": {}}
        if crawl_result.emails:
            print(f"  📧 Classifying {len(crawl_result.emails)} emails...")
            classification_results = self.classify_emails_batch(list(crawl_result.emails))
            
            # Print classification summary
            if classification_results.get("summary"):
                summary = classification_results["summary"]
                print(f"  📊 Classification: HR={summary.get('hr', 0)}, "
                      f"Sales={summary.get('sales', 0)}, "
                      f"Review needed={summary.get('requires_review', 0)}")
        
        # Combine results
        return {
            "website": website,
            "crawl_result": crawl_result,
            "classification": classification_results,
            "emails_by_category": self._organize_emails_by_category(classification_results)
        }
    
    def _organize_emails_by_category(self, classification_results: Dict) -> Dict:
        """Organize emails by their classification"""
        organized = {
            "hr_emails": [],
            "sales_emails": [],
            "review_needed": []
        }
        
        for result in classification_results.get("results", []):
            email_info = {
                "email": result["email"],
                "confidence": result["confidence"]
            }
            
            if result.get("requires_review", False):
                organized["review_needed"].append(email_info)
            elif result["category"] == "HR":
                organized["hr_emails"].append(email_info)
            else:  # Sales
                organized["sales_emails"].append(email_info)
        
        return organized
    
    async def process_all_websites(self, limit: int = None):
        """Process all websites from the CSV"""
        websites = self.crawler.load_websites_from_csv(limit=limit or 100)
        if not websites:
            print("No valid websites found")
            return []
        
        print(f"\n🚀 Processing {len(websites)} websites")
        print("=" * 80)
        
        all_results = []
        
        for i, website in enumerate(websites, 1):
            result = await self.process_website_with_classification(website, i, len(websites))
            all_results.append(result)
            
            # Rate limiting
            await asyncio.sleep(1)
        
        print("\n✅ All websites processed and classified!")
        return all_results
    
    def export_enriched_results(self, results: List[Dict], filename: str = None) -> str:
        """Export results with email classifications to CSV"""
        if not filename:
            filename = f"crawled_classified_emails_{int(time.time())}.csv"
        
        export_data = []
        
        for result in results:
            website = result["website"]
            crawl_result = result["crawl_result"]
            emails_by_cat = result["emails_by_category"]
            
            # Create row with all information
            row = {
                "website": website,
                "total_emails_found": len(crawl_result.emails),
                "hr_email_count": len(emails_by_cat["hr_emails"]),
                "sales_email_count": len(emails_by_cat["sales_emails"]),
                "review_needed_count": len(emails_by_cat["review_needed"]),
                "hr_emails": ", ".join([e["email"] for e in emails_by_cat["hr_emails"]]),
                "sales_emails": ", ".join([e["email"] for e in emails_by_cat["sales_emails"]]),
                "review_needed_emails": ", ".join([e["email"] for e in emails_by_cat["review_needed"]]),
                "career_links": ", ".join(crawl_result.career_links) if crawl_result.career_links else "",
                "pages_crawled": crawl_result.pages_crawled,
                "crawl_status": crawl_result.status,
                "crawl_time": f"{crawl_result.time_taken:.1f}s"
            }
            export_data.append(row)
        
        # Export to CSV
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        print(f"\n📁 Results exported to: {filename}")
        
        return filename

# Main execution
async def main():
    """Main function to run the integrated crawler and classifier"""
    
    # Configuration
    CSV_FILE = "final_classified_businesses.csv"  # Your CSV file
    API_URL = "http://localhost:8000"
    LIMIT_WEBSITES = 5  # Start with just 5 websites for testing
    
    print(f"🚀 Email Crawler & Classifier System")
    print(f"📋 Using CSV: {CSV_FILE}")
    print(f"🔗 API URL: {API_URL}")
    print(f"🎯 Processing first {LIMIT_WEBSITES} websites")
    print("=" * 80)
    
    # Create the integrated system
    system = CrawlAndClassifySystem(API_URL, CSV_FILE)
    
    # Process websites and classify emails
    results = await system.process_all_websites(limit=LIMIT_WEBSITES)
    
    # Export results
    if results:
        export_file = system.export_enriched_results(results)
        print(f"\n✅ Complete! Results saved to: {export_file}")
        
        # Print summary
        total_emails = sum(len(r["crawl_result"].emails) for r in results)
        total_hr = sum(len(r["emails_by_category"]["hr_emails"]) for r in results)
        total_sales = sum(len(r["emails_by_category"]["sales_emails"]) for r in results)
        
        print(f"\n📊 Summary:")
        print(f"   Total emails found: {total_emails}")
        print(f"   HR emails: {total_hr}")
        print(f"   Sales emails: {total_sales}")
    else:
        print("\n❌ No results to export")

if __name__ == "__main__":
    # First check if the email_crawler module exists
    try:
        from email_crawler import EmailCareerWebCrawler
        print("✅ Email crawler module loaded successfully")
    except ImportError:
        print("❌ Email crawler module not found!")
        print("Please save your crawler code to scripts/email_crawler.py first")
        exit(1)
    
    # Run the main function
    asyncio.run(main())