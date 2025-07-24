import asyncio
import pandas as pd
import requests
import json
from typing import List, Dict, Set
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from email_crawler import EmailCareerWebCrawler, CrawlResult
from career_page_graphrag import CareerPageClassifier

class CompleteBusinessAnalyzer:
    def __init__(self, api_base_url: str = "http://localhost:8000", csv_file: str = None):
        self.api_base_url = api_base_url
        self.crawler = EmailCareerWebCrawler(csv_file)
        self.career_classifier = CareerPageClassifier()
        
    def classify_emails_batch(self, emails: List[str]) -> Dict:
        """Classify emails as HR/Sales using the API"""
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
    
    def find_best_career_page(self, career_links: Set[str]) -> Dict:
        """Find the most accurate career page from a list of URLs"""
        if not career_links:
            return {
                'best_career_page': None,
                'confidence': 0,
                'all_career_pages': []
            }
        
        print(f"     Analyzing {len(career_links)} career link candidates:")
        
        # Convert set to list and clean duplicates (http vs https)
        unique_links = []
        seen_normalized = set()
        
        for link in career_links:
            # Normalize URL (remove protocol differences)
            normalized = link.replace('https://', '').replace('http://', '').rstrip('/')
            if normalized not in seen_normalized:
                seen_normalized.add(normalized)
                unique_links.append(link)
        
        # Filter out obviously bad URLs (tracking pixels, etc)
        filtered_links = []
        for link in unique_links:
            # Skip tracking/analytics URLs
            if any(bad in link for bad in ['@', 'pixel', 'analytics', 'tracking', 'wpm@']):
                continue
            filtered_links.append(link)
        
        if not filtered_links:
            return {
                'best_career_page': None,
                'confidence': 0,
                'all_career_pages': []
            }
        
        # Classify all URLs
        urls_to_rank = [{'url': url} for url in filtered_links]
        ranked_pages = self.career_classifier.rank_career_pages(urls_to_rank)
        
        # Debug: Show all classifications
        for page in ranked_pages[:5]:
            print(f"       - {page['url']}")
            print(f"         Is Career: {page['is_career_page']}, Score: {page['final_score']:.2%}, "
                  f"Patterns: {', '.join(page['found_patterns']) if page['found_patterns'] else 'none'}")
        
        # Get all pages with any career indication
        career_pages = []
        for page in ranked_pages:
            # Include if marked as career page OR has good URL patterns OR decent score
            if (page['is_career_page'] or 
                page['url_pattern_score'] > 0.5 or 
                page['final_score'] > 0.3):
                career_pages.append(page)
        
        if career_pages:
            # Sort by final score
            career_pages.sort(key=lambda x: x['final_score'], reverse=True)
            best_page = career_pages[0]
            
            return {
                'best_career_page': best_page['url'],
                'confidence': best_page['confidence'],
                'pattern_score': best_page['url_pattern_score'],
                'patterns_found': best_page['found_patterns'],
                'all_career_pages': [
                    {
                        'url': p['url'],
                        'confidence': p['confidence'],
                        'score': p['final_score'],
                        'patterns': p['found_patterns']
                    } for p in career_pages[:5]  # Top 5
                ]
            }
        else:
            # Even if no high-confidence pages, return the best candidate
            if ranked_pages:
                best = ranked_pages[0]
                return {
                    'best_career_page': best['url'],
                    'confidence': best['confidence'],
                    'pattern_score': best['url_pattern_score'],
                    'patterns_found': best['found_patterns'],
                    'all_career_pages': [{
                        'url': best['url'],
                        'confidence': best['confidence'],
                        'score': best['final_score'],
                        'patterns': best['found_patterns']
                    }]
                }
            
            return {
                'best_career_page': None,
                'confidence': 0,
                'all_career_pages': []
            }
    
    async def analyze_website(self, website: str, index: int = 1, total: int = 1) -> Dict:
        """Complete analysis of a website"""
        print(f"\n[{index}/{total}] Analyzing: {website}")
        print("-" * 60)
        
        # Step 1: Crawl website
        crawl_result = await self.crawler.process_single_website(website, index, total)
        
        # Step 2: Classify emails if found
        email_classification = {"results": [], "summary": {}}
        if crawl_result.emails:
            print(f"  📧 Classifying {len(crawl_result.emails)} emails...")
            email_classification = self.classify_emails_batch(list(crawl_result.emails))
            
            if email_classification.get("summary"):
                summary = email_classification["summary"]
                print(f"     HR: {summary.get('hr', 0)}, Sales: {summary.get('sales', 0)}")
        
        # Step 3: Find best career page
        career_analysis = {'best_career_page': None, 'confidence': 0, 'all_career_pages': []}
        if crawl_result.career_links:
            print(f"  💼 Analyzing potential career pages...")
            career_analysis = self.find_best_career_page(crawl_result.career_links)
            
            if career_analysis['best_career_page']:
                print(f"     ✅ Best career page: {career_analysis['best_career_page']}")
                print(f"        Confidence: {career_analysis['confidence']:.1%}")
                if len(career_analysis['all_career_pages']) > 1:
                    print(f"        Other candidates: {len(career_analysis['all_career_pages']) - 1}")
            else:
                print(f"     ❌ No career pages identified from {len(crawl_result.career_links)} links")
        
        # Organize email results
        email_categories = self._organize_emails_by_category(email_classification)
        
        # Combine all results
        return {
            "website": website,
            "crawl_status": crawl_result.status,
            "pages_crawled": crawl_result.pages_crawled,
            "emails": {
                "total_found": len(crawl_result.emails),
                "hr_emails": email_categories["hr_emails"],
                "sales_emails": email_categories["sales_emails"],
                "review_needed": email_categories["review_needed"]
            },
            "career_page": {
                "best_url": career_analysis.get('best_career_page'),
                "confidence": career_analysis.get('confidence', 0),
                "all_found": len(crawl_result.career_links),
                "top_candidates": career_analysis.get('all_career_pages', [])
            },
            "raw_data": {
                "all_emails": list(crawl_result.emails),
                "all_career_links": list(crawl_result.career_links)
            }
        }
    
    def _organize_emails_by_category(self, classification_results: Dict) -> Dict:
        """Organize classified emails by category"""
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
    
    async def analyze_all_websites(self, limit: int = None):
        """Analyze all websites from the CSV"""
        websites = self.crawler.load_websites_from_csv(limit=limit or 100)
        if not websites:
            print("No valid websites found")
            return []
        
        print(f"\n🚀 Analyzing {len(websites)} websites")
        print("=" * 80)
        
        all_results = []
        start_time = time.time()
        
        for i, website in enumerate(websites, 1):
            result = await self.analyze_website(website, i, len(websites))
            all_results.append(result)
            
            # Rate limiting
            await asyncio.sleep(1)
        
        total_time = time.time() - start_time
        print(f"\n✅ Analysis complete! Total time: {total_time:.1f}s")
        
        return all_results
    
    def export_results(self, results: List[Dict], filename: str = None) -> str:
        """Export comprehensive results to CSV"""
        if not filename:
            filename = f"business_analysis_complete_{int(time.time())}.csv"
        
        export_data = []
        
        for result in results:
            # Prepare career page info
            career_page_list = []
            if result["career_page"]["top_candidates"]:
                for i, cp in enumerate(result["career_page"]["top_candidates"][:3]):
                    career_page_list.append(f"{cp['url']} ({cp['confidence']:.0%})")
            
            row = {
                # Basic info
                "website": result["website"],
                "crawl_status": result["crawl_status"],
                "pages_crawled": result["pages_crawled"],
                
                # Email summary
                "total_emails": result["emails"]["total_found"],
                "hr_email_count": len(result["emails"]["hr_emails"]),
                "sales_email_count": len(result["emails"]["sales_emails"]),
                "review_email_count": len(result["emails"]["review_needed"]),
                
                # Email lists
                "hr_emails": ", ".join([e["email"] for e in result["emails"]["hr_emails"]]),
                "sales_emails": ", ".join([e["email"] for e in result["emails"]["sales_emails"]]),
                "review_emails": ", ".join([e["email"] for e in result["emails"]["review_needed"]]),
                
                # Career page info
                "best_career_page": result["career_page"]["best_url"] or "",
                "career_page_confidence": result["career_page"]["confidence"],
                "total_career_links_found": result["career_page"]["all_found"],
                
                # All career pages found
                "all_career_pages": " | ".join(career_page_list) if career_page_list else "",
                
                # Raw career links for debugging
                "raw_career_links": ", ".join(list(result["raw_data"]["all_career_links"])[:5])
            }
            export_data.append(row)
        
        # Export to CSV
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        print(f"\n📁 Results exported to: {filename}")
        
        # Also save detailed JSON
        json_filename = filename.replace('.csv', '_detailed.json')
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📁 Detailed JSON saved to: {json_filename}")
        
        return filename
    
    def print_summary(self, results: List[Dict]):
        """Print analysis summary"""
        print("\n" + "="*80)
        print("📊 COMPLETE BUSINESS ANALYSIS SUMMARY")
        print("="*80)
        
        # Calculate totals
        total_websites = len(results)
        total_emails = sum(r["emails"]["total_found"] for r in results)
        total_hr = sum(len(r["emails"]["hr_emails"]) for r in results)
        total_sales = sum(len(r["emails"]["sales_emails"]) for r in results)
        websites_with_career = sum(1 for r in results if r["career_page"]["best_url"])
        total_career_links = sum(r["career_page"]["all_found"] for r in results)
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Websites analyzed: {total_websites}")
        print(f"   Total emails found: {total_emails}")
        if total_emails > 0:
            print(f"   - HR emails: {total_hr} ({total_hr/total_emails*100:.1f}%)")
            print(f"   - Sales emails: {total_sales} ({total_sales/total_emails*100:.1f}%)")
        print(f"   Career pages identified: {websites_with_career}/{total_websites} ({websites_with_career/total_websites*100:.1f}%)")
        print(f"   Total career links analyzed: {total_career_links}")
        
        # Show all career pages found
        print(f"\n💼 All career pages found:")
        for result in results:
            if result["career_page"]["best_url"]:
                print(f"\n   📍 {result['website']}")
                for i, cp in enumerate(result["career_page"]["top_candidates"][:3], 1):
                    emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                    print(f"      {emoji} {cp['url']}")
                    print(f"         Confidence: {cp['confidence']:.0%}, Patterns: {', '.join(cp['patterns']) if cp['patterns'] else 'none'}")

# Main execution
async def main():
    """Run the complete business analyzer"""
    
    # Configuration
    CSV_FILE = "final_classified_businesses.csv"
    API_URL = "http://localhost:8000"
    LIMIT = 5  # Process first 5 websites
    
    print("🚀 Complete Business Analyzer (Fixed)")
    print(f"📋 CSV: {CSV_FILE}")
    print(f"🔗 API: {API_URL}")
    print(f"🎯 Limit: {LIMIT} websites")
    
    # Check API
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code != 200:
            raise Exception("API not healthy")
        print("✅ Email classifier API is running")
    except:
        print("❌ Email classifier API is not running!")
        print("Please start it with: python scripts/api_server.py")
        return
    
    # Create analyzer
    analyzer = CompleteBusinessAnalyzer(API_URL, CSV_FILE)
    
    # Analyze websites
    results = await analyzer.analyze_all_websites(limit=LIMIT)
    
    if results:
        # Print summary
        analyzer.print_summary(results)
        
        # Export results
        export_file = analyzer.export_results(results)
        print(f"\n✅ Analysis complete! Check {export_file}")
    else:
        print("\n❌ No results to export")

if __name__ == "__main__":
    asyncio.run(main())