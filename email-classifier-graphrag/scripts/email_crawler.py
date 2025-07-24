import asyncio
import re
import time
import json
import pandas as pd
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from crawl4ai import AsyncWebCrawler
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

@dataclass
class CrawlResult:
    """Data structure to store crawling results for a single website"""
    website: str
    emails: Set[str]
    career_links: Set[str]
    pages_crawled: int
    status: str
    time_taken: float
    email_sources: Dict[str, List[str]]
    career_sources: Dict[str, str]

class EmailCareerWebCrawler:
    """
    Unified web crawler for extracting emails and career page links from websites.
    
    This class handles the complete workflow of:
    1. Loading websites from CSV files
    2. Crawling websites with depth-first search
    3. Extracting emails using multiple regex patterns
    4. Finding career-related pages and links
    5. Exporting results to CSV format
    """
    
    def __init__(self, csv_file_path: str = None):
        """
        Initialize the crawler with configuration and data structures.
        
        Args:
            csv_file_path (str): Path to CSV file containing websites to crawl
        """
        # Configuration
        self.RESPECT_ROBOTS_TXT = False
        self.MAX_PAGES_TO_CRAWL = 20
        self.MAX_CRAWL_DEPTH = 2
        self.CONCURRENT_REQUESTS = 10
        self.USER_AGENT = "Email-Career-Extractor-Bot/1.0"
        self.TIMEOUT_PER_WEBSITE = 120
        
        # Email extraction patterns
        self.EMAIL_PATTERNS = [
            r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            r'(?i)\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            r'[a-zA-Z0-9._%+-]+\s*@\s*[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            r'[a-zA-Z0-9._%+-]+\s*\[\s*at\s*\]\s*[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            r'[a-zA-Z0-9._%+-]+\s*\(\s*at\s*\)\s*[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            r'[a-zA-Z0-9._%+-]+\s*\{\s*at\s*\}\s*[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            r'[a-zA-Z0-9._%+-]+\s*AT\s*[a-zA-Z0-9.-]+\s*DOT\s*[a-zA-Z]{2,}',
        ]
        
        # Keywords for career page detection
        self.CAREER_KEYWORDS = [
            "career", "careers", "jobs", "job", "join-us", "join-our-team", "work-with-us",
            "opportunities", "hiring", "employment", "apply", "recruit", "team", "recruitment",
            "work-here", "join-team", "open-positions", "job-opportunities", "hiring-now",
            "employment-opportunities", "work-at", "join-our-company", "be-part-of",
            "work-opportunities", "job-openings", "join-the-team", "career-opportunities"
        ]
        
        # Keywords for about/company pages
        self.ABOUT_KEYWORDS = [
            "about", "company", "team", "staff", "people", "culture", "mission", "vision",
            "values", "leadership", "management", "organization", "corporate", "info",
            "information", "who-we-are", "our-story", "our-company", "our-team"
        ]
        
        # File extensions to ignore
        self.IGNORED_EXTENSIONS = {
            '.pdf', '.zip', '.rar', '.doc', '.docx', '.xls', '.xlsx',
            '.ppt', '.pptx', '.mp3', '.mp4', '.avi', '.mov', '.jpg',
            '.jpeg', '.png', '.gif', '.svg', '.webp', '.bmp', '.css', '.js'
        }
        
        # Data structures
        self.csv_file_path = csv_file_path
        self.results = []

    def load_websites_from_csv(self, limit: int = 100) -> List[str]:
        """
        Load and validate website URLs from a CSV file.
        
        Args:
            limit (int): Maximum number of websites to load
            
        Returns:
            List[str]: List of validated website URLs
            
        Raises:
            ValueError: If CSV doesn't contain 'Website' column
        """
        if not self.csv_file_path:
            return []

        try:
            df = pd.read_csv(self.csv_file_path)
            if 'Website' not in df.columns:
                raise ValueError("CSV must contain 'Website' column")

            websites = df['Website'].dropna().head(limit).tolist()
            cleaned_websites = []
            
            for website in websites:
                website = str(website).strip()
                if website and not website.startswith(('http://', 'https://')):
                    website = 'https://' + website
                if website and website.startswith(('http://', 'https://')):
                    cleaned_websites.append(website)
                    
            return cleaned_websites
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return []

    def normalize_email_format(self, email: str) -> str:
        """
        Normalize obfuscated email addresses to standard format.
        
        Handles various obfuscation patterns like:
        - user [at] domain.com -> user@domain.com
        - user AT domain DOT com -> user@domain.com
        
        Args:
            email (str): Raw email string that may be obfuscated
            
        Returns:
            str: Normalized email address in lowercase
        """
        email = email.strip()
        replacements = [
            (r'\s*\[\s*at\s*\]\s*', '@'),
            (r'\s*\(\s*at\s*\)\s*', '@'),
            (r'\s*\{\s*at\s*\}\s*', '@'),
            (r'\s*AT\s*', '@'),
            (r'\s*DOT\s*', '.'),
            (r'\s+', '')
        ]
        
        for pattern, replacement in replacements:
            email = re.sub(pattern, replacement, email, flags=re.IGNORECASE)
        return email.lower()

    def validate_email_format(self, email: str) -> bool:
        """
        Validate if an email address has a proper format.
        
        Checks for:
        - Basic format (local@domain)
        - Valid domain structure
        - Filters out common false positives
        - Excludes file extensions
        
        Args:
            email (str): Email address to validate
            
        Returns:
            bool: True if email format is valid, False otherwise
        """
        if not email or '@' not in email:
            return False
            
        try:
            local, domain = email.split('@', 1)
        except ValueError:
            return False

        if not local or not domain or '.' not in domain:
            return False
            
        if any(email.endswith(ext) for ext in self.IGNORED_EXTENSIONS):
            return False

        # Filter out common false positives
        false_positives = [
            'example@example.com', 'test@test.com', 'admin@admin.com',
            'user@domain.com', 'email@email.com', 'name@domain.com'
        ]
        if email in false_positives:
            return False

        domain_parts = domain.split('.')
        return len(domain_parts) >= 2 and len(domain_parts[-1]) >= 2

    def extract_emails_from_content(self, content: str, source_url: str, email_sources: Dict) -> Set[str]:
        """
        Extract all valid email addresses from web page content.
        
        Uses multiple regex patterns to find emails, including obfuscated ones.
        Tracks the source URL where each email was found.
        
        Args:
            content (str): Web page content (HTML, text, etc.)
            source_url (str): URL where the content was found
            email_sources (Dict): Dictionary to track email sources
            
        Returns:
            Set[str]: Set of valid email addresses found
        """
        emails = set()
        
        for pattern in self.EMAIL_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                normalized = self.normalize_email_format(match)
                if self.validate_email_format(normalized):
                    emails.add(normalized)
                    if normalized not in email_sources:
                        email_sources[normalized] = []
                    email_sources[normalized].append(source_url)
                    
        return emails

    def is_career_related_link(self, href: str, text: str) -> bool:
        """
        Determine if a link is related to careers or job opportunities.
        
        Checks both URL path and link text for career-related keywords.
        
        Args:
            href (str): URL/href attribute of the link
            text (str): Visible text of the link
            
        Returns:
            bool: True if link appears to be career-related
        """
        return any(keyword in href.lower() or keyword in text.lower()
                  for keyword in self.CAREER_KEYWORDS)

    def is_about_company_link(self, href: str, text: str) -> bool:
        """
        Determine if a link is related to company information or about pages.
        
        Args:
            href (str): URL/href attribute of the link
            text (str): Visible text of the link
            
        Returns:
            bool: True if link appears to be about company/team
        """
        return any(keyword in href.lower() or keyword in text.lower()
                  for keyword in self.ABOUT_KEYWORDS)

    def calculate_url_priority(self, href: str, text: str) -> int:
        """
        Calculate priority score for a URL to determine crawling order.
        
        Higher scores indicate more relevant pages for email/career extraction.
        Career pages get highest priority, followed by about pages.
        
        Args:
            href (str): URL/href attribute
            text (str): Link text
            
        Returns:
            int: Priority score (higher = more important)
        """
        priority = 0
        
        if self.is_career_related_link(href, text):
            priority += 100
        if self.is_about_company_link(href, text):
            priority += 50

        # Team-related pages get medium priority
        team_keywords = ["team", "people", "staff", "employees"]
        if any(keyword in href.lower() or keyword in text.lower() for keyword in team_keywords):
            priority += 25

        # Lower priority for non-essential pages
        low_priority = ["blog", "news", "press", "media", "contact", "faq", "help"]
        if any(keyword in href.lower() or keyword in text.lower() for keyword in low_priority):
            priority -= 10
            
        return priority

    def should_ignore_url(self, parsed_url) -> bool:
        """
        Check if a URL should be ignored based on file extension.
        
        Args:
            parsed_url: Parsed URL object
            
        Returns:
            bool: True if URL should be ignored
        """
        return any(parsed_url.path.lower().endswith(ext) for ext in self.IGNORED_EXTENSIONS)

    def convert_to_absolute_url(self, link: str, source_url: str) -> str:
        """
        Convert relative URLs to absolute URLs.
        
        Args:
            link (str): Potentially relative URL
            source_url (str): Base URL for relative resolution
            
        Returns:
            str: Absolute URL
        """
        if link.startswith(('http://', 'https://')):
            return link
        return urljoin(source_url, link)

    def is_valid_internal_url(self, url: str, base_netloc: str) -> bool:
        """
        Check if URL is valid and belongs to the same domain.
        
        Args:
            url (str): URL to validate
            base_netloc (str): Base domain to compare against
            
        Returns:
            bool: True if URL is valid and internal
        """
        parsed_url = urlparse(url)
        return (parsed_url.netloc == base_netloc and
                parsed_url.scheme in ['http', 'https'] and
                not self.should_ignore_url(parsed_url))

    def clean_url_parameters(self, url: str) -> str:
        """
        Remove query parameters and fragments from URL.
        
        Args:
            url (str): URL with potential parameters
            
        Returns:
            str: Clean URL without query/fragment
        """
        return urlparse(url)._replace(query="", fragment="").geturl()

    def extract_career_links_from_content(self, content: str, source_url: str, base_netloc: str, career_sources: Dict) -> Set[str]:
        """
        Extract career-related links from web page content.
        
        Finds links that appear to lead to career/job pages based on
        URL patterns and link text analysis.
        
        Args:
            content (str): Web page content
            source_url (str): Source URL of the content
            base_netloc (str): Base domain for internal link validation
            career_sources (Dict): Dictionary to track career link sources
            
        Returns:
            Set[str]: Set of career-related URLs
        """
        career_links = set()
        link_patterns = [r'href=[\'"]([^\'"]*)[\'"]', r'src=[\'"]([^\'"]*)[\'"]']

        for pattern in link_patterns:
            found_links = re.findall(pattern, content, re.IGNORECASE)
            for link in found_links:
                absolute_url = self.convert_to_absolute_url(link, source_url)
                if self.is_valid_internal_url(absolute_url, base_netloc):
                    # Extract link text for analysis
                    text_pattern = rf'<a[^>]*href=[\'"]?{re.escape(link)}[\'"]?[^>]*>([^<]*)</a>'
                    text_match = re.search(text_pattern, content, re.IGNORECASE)
                    link_text = text_match.group(1).strip() if text_match else ""

                    if self.is_career_related_link(link, link_text):
                        clean_url = self.clean_url_parameters(absolute_url)
                        career_links.add(clean_url)
                        career_sources[clean_url] = source_url
                        
        return career_links

    def extract_all_page_content(self, crawl_result) -> str:
        """
        Extract all available content from a crawl result object.
        
        Combines HTML, cleaned HTML, markdown, and structured data
        into a single text block for analysis.
        
        Args:
            crawl_result: Result object from web crawler
            
        Returns:
            str: Combined content from all sources
        """
        content_parts = []
        if crawl_result.cleaned_html:
            content_parts.append(crawl_result.cleaned_html)
        if crawl_result.html:
            content_parts.append(crawl_result.html)
        if crawl_result.markdown:
            content_parts.append(str(crawl_result.markdown))
        if hasattr(crawl_result, 'structured_data') and crawl_result.structured_data:
            content_parts.append(str(crawl_result.structured_data))
        return '\n'.join(content_parts)

    def extract_links_with_priority_scoring(self, crawl_result, source_url: str, base_netloc: str) -> List[Tuple[str, int]]:
        """
        Extract all internal links from page content with priority scores.
        
        Attempts to use structured link data first, falls back to regex extraction.
        Each link gets a priority score based on relevance to email/career extraction.
        
        Args:
            crawl_result: Crawl result object containing page data
            source_url (str): URL of the source page
            base_netloc (str): Base domain for validation
            
        Returns:
            List[Tuple[str, int]]: List of (URL, priority_score) tuples
        """
        links_with_priority = []

        # Try structured links first (if available from crawler)
        if hasattr(crawl_result, 'links') and crawl_result.links:
            for link_data in crawl_result.links.get("internal", []):
                href = link_data.get("href", "")
                text = link_data.get("text", "")
                if href and self.is_valid_internal_url(href, base_netloc):
                    clean_url = self.clean_url_parameters(href)
                    priority = self.calculate_url_priority(href, text)
                    links_with_priority.append((clean_url, priority))

        # Fallback to regex extraction from content
        if not links_with_priority:
            content = self.extract_all_page_content(crawl_result)
            found_links = re.findall(r'href=[\'"]([^\'"]*)[\'"]', content, re.IGNORECASE)
            for link in found_links:
                absolute_url = self.convert_to_absolute_url(link, source_url)
                if self.is_valid_internal_url(absolute_url, base_netloc):
                    clean_url = self.clean_url_parameters(absolute_url)
                    text_pattern = rf'<a[^>]*href=[\'"]?{re.escape(link)}[\'"]?[^>]*>([^<]*)</a>'
                    text_match = re.search(text_pattern, content, re.IGNORECASE)
                    link_text = text_match.group(1).strip() if text_match else ""
                    priority = self.calculate_url_priority(link, link_text)
                    links_with_priority.append((clean_url, priority))

        return links_with_priority

    def check_robots_txt_permission(self, robot_parser, url: str) -> bool:
        """
        Check if crawling is allowed by robots.txt.
        
        Args:
            robot_parser: RobotFileParser object
            url (str): URL to check
            
        Returns:
            bool: True if crawling is allowed
        """
        return not robot_parser or robot_parser.can_fetch(self.USER_AGENT, url)

    async def crawl_single_page(self, crawler, url: str, depth: int, crawl_state: Dict):
        """
        Crawl a single web page and extract emails and career links.
        
        This method handles:
        - Page content retrieval using the web crawler
        - Email extraction from page content
        - Career link discovery and validation
        - Link queue management for further crawling
        
        Args:
            crawler: AsyncWebCrawler instance
            url (str): URL to crawl
            depth (int): Current crawling depth
            crawl_state (Dict): Shared state containing visited URLs, found data, etc.
        """
        if (crawl_state['is_cancelled'] or 
            url in crawl_state['visited_urls'] or
            len(crawl_state['visited_urls']) >= self.MAX_PAGES_TO_CRAWL or
            not self.check_robots_txt_permission(crawl_state['robot_parser'], url)):
            return

        try:
            result = await crawler.arun(
                url=url, bypass_cache=False, user_agent=self.USER_AGENT,
                word_count_threshold=1, timeout=30000, delay_before_return_html=2.0,
                remove_overlay_elements=True, simulate_user=True, override_navigator=True,
                extract_links=True, only_text=False, extract_structured_data=True
            )

            if result.success:
                crawl_state['visited_urls'].add(url)
                all_content = self.extract_all_page_content(result)

                # Extract emails
                new_emails = self.extract_emails_from_content(
                    all_content, url, crawl_state['email_sources']
                )
                fresh_emails = new_emails - crawl_state['found_emails']
                if fresh_emails:
                    crawl_state['found_emails'].update(fresh_emails)
                    print(f"  Found {len(fresh_emails)} new email(s) on {url}")

                # Extract career links
                new_career_links = self.extract_career_links_from_content(
                    all_content, url, crawl_state['base_netloc'], crawl_state['career_sources']
                )
                fresh_career_links = new_career_links - crawl_state['found_career_links']
                if fresh_career_links:
                    crawl_state['found_career_links'].update(fresh_career_links)
                    print(f"  Found {len(fresh_career_links)} new career link(s) on {url}")
                    for link in fresh_career_links:
                        print(f"    Career Link: {link}")

                # Add new URLs to crawling queue
                if depth < self.MAX_CRAWL_DEPTH and not crawl_state['is_cancelled']:
                    new_links = self.extract_links_with_priority_scoring(
                        result, url, crawl_state['base_netloc']
                    )
                    for link_url, priority in new_links:
                        if link_url not in crawl_state['visited_urls']:
                            existing = [p for p, u in crawl_state['urls_to_crawl'] if u == link_url]
                            if not existing:
                                crawl_state['urls_to_crawl'].append((priority, link_url))
                            elif priority > max(existing):
                                crawl_state['urls_to_crawl'] = [
                                    (p, u) for p, u in crawl_state['urls_to_crawl'] if u != link_url
                                ]
                                crawl_state['urls_to_crawl'].append((priority, link_url))

                    # Limit queue size to prevent memory issues
                    if len(crawl_state['urls_to_crawl']) > 50:
                        crawl_state['urls_to_crawl'].sort(key=lambda x: -x[0])
                        crawl_state['urls_to_crawl'] = crawl_state['urls_to_crawl'][:50]

        except Exception as e:
            print(f"  Error crawling {url}: {str(e)}")

    async def run_website_crawl(self, start_url: str) -> Tuple[List[str], List[str], Dict, Dict, Set[str]]:
        """
        Execute complete crawling process for a single website.
        
        Manages the crawling workflow including:
        - Robots.txt checking (if enabled)
        - Priority-based URL queue management
        - Concurrent page crawling with semaphore
        - Depth-limited traversal
        
        Args:
            start_url (str): Starting URL for the crawl
            
        Returns:
            Tuple containing:
            - List of found emails
            - List of career links
            - Email sources mapping
            - Career sources mapping
            - Set of visited URLs
        """
        base_netloc = urlparse(start_url).netloc
        
        # Initialize crawl state
        crawl_state = {
            'base_netloc': base_netloc,
            'visited_urls': set(),
            'found_emails': set(),
            'found_career_links': set(),
            'urls_to_crawl': [(0, start_url)],
            'is_cancelled': False,
            'email_sources': {},
            'career_sources': {}
        }

        # Setup robots.txt parser if needed
        crawl_state['robot_parser'] = None
        if self.RESPECT_ROBOTS_TXT:
            crawl_state['robot_parser'] = RobotFileParser()
            try:
                robots_url = urljoin(start_url, '/robots.txt')
                crawl_state['robot_parser'].set_url(robots_url)
                crawl_state['robot_parser'].read()
            except Exception:
                pass

        async with AsyncWebCrawler(
            headless=True, browser_type="chromium", user_agent=self.USER_AGENT,
            verbose=False, max_concurrent_sessions=self.CONCURRENT_REQUESTS
        ) as crawler:

            semaphore = asyncio.Semaphore(self.CONCURRENT_REQUESTS)

            async def crawl_with_semaphore(url, depth):
                async with semaphore:
                    await self.crawl_single_page(crawler, url, depth, crawl_state)

            depth = 0
            while (crawl_state['urls_to_crawl'] and 
                   len(crawl_state['visited_urls']) < self.MAX_PAGES_TO_CRAWL and 
                   not crawl_state['is_cancelled']):
                
                # Sort URLs by priority (highest first)
                crawl_state['urls_to_crawl'].sort(key=lambda x: -x[0])

                # Prepare batch for concurrent processing
                current_batch = []
                batch_size = min(self.CONCURRENT_REQUESTS, len(crawl_state['urls_to_crawl']))
                for _ in range(batch_size):
                    if crawl_state['urls_to_crawl']:
                        priority, url = crawl_state['urls_to_crawl'].pop(0)
                        if url not in crawl_state['visited_urls']:
                            current_batch.append((url, depth))

                if not current_batch:
                    break

                # Execute batch crawling
                tasks = [crawl_with_semaphore(url, d) for url, d in current_batch]
                await asyncio.gather(*tasks, return_exceptions=True)

                depth += 1
                if depth >= self.MAX_CRAWL_DEPTH:
                    break
                await asyncio.sleep(0.5)  # Rate limiting

        return (
            list(crawl_state['found_emails']),
            list(crawl_state['found_career_links']),
            crawl_state['email_sources'],
            crawl_state['career_sources'],
            crawl_state['visited_urls']
        )

    async def process_single_website(self, website: str, index: int = 1, total: int = 1) -> CrawlResult:
        """
        Process a single website and return structured results.
        
        Handles the complete workflow for one website:
        - Initiating the crawl process
        - Managing timeouts and cancellation
        - Collecting and organizing results
        - Error handling and status reporting
        
        Args:
            website (str): Website URL to process
            index (int): Current website number (for progress display)
            total (int): Total number of websites being processed
            
        Returns:
            CrawlResult: Structured results containing all extracted data
        """
        print(f"[{index}/{total}] Processing: {website}")

        start_time = time.time()
        status = 'completed'

        try:
            emails, career_links, email_sources, career_sources, visited_urls = await asyncio.wait_for(
                self.run_website_crawl(website), 
                timeout=self.TIMEOUT_PER_WEBSITE
            )
        except asyncio.TimeoutError:
            emails, career_links, email_sources, career_sources = [], [], {}, {}
            visited_urls = set()
            status = 'timeout'
        except Exception as e:
            emails, career_links, email_sources, career_sources = [], [], {}, {}
            visited_urls = set()
            status = f'error: {str(e)}'

        elapsed_time = time.time() - start_time

        result = CrawlResult(
            website=website,
            emails=set(emails),
            career_links=set(career_links),
            pages_crawled=len(visited_urls),
            status=status,
            time_taken=elapsed_time,
            email_sources=email_sources,
            career_sources=career_sources
        )

        # Print result summary
        status_symbol = {
            'completed': '✅ SUCCESS',
            'timeout': '⏰ TIMEOUT'
        }.get(status.split(':')[0], '❌ ERROR')
        
        print(f"  {status_symbol} {status.title()} in {elapsed_time:.1f}s - "
              f"Found {len(emails)} emails, {len(career_links)} career pages")

        return result

    async def process_all_websites_from_csv(self):
        """
        Process all websites loaded from the CSV file.
        
        Main orchestration method that:
        - Loads websites from CSV
        - Processes each website sequentially
        - Handles rate limiting between requests
        - Provides progress updates and summary
        """
        websites = self.load_websites_from_csv()
        if not websites:
            print("No valid websites found in CSV")
            return

        print(f"Processing {len(websites)} websites from CSV")
        print(f"Timeout per website: {self.TIMEOUT_PER_WEBSITE} seconds")
        print("=" * 60)

        for i, website in enumerate(websites, 1):
            result = await self.process_single_website(website, i, len(websites))
            self.results.append(result)
            await asyncio.sleep(1)  # Rate limiting between websites

        print("\n🎉 All websites processed!")
        self.print_processing_summary()

    def print_processing_summary(self):
        """
        Print a comprehensive summary of all crawling results.
        
        Displays:
        - Overall statistics (total emails, career links, success rates)
        - Per-website breakdown with key metrics
        - Sample emails and career links found
        """
        print("\n" + "="*80)
        print("PROCESSING SUMMARY")
        print("="*80)

        total_emails = sum(len(r.emails) for r in self.results)
        total_career_links = sum(len(r.career_links) for r in self.results)
        completed = sum(1 for r in self.results if r.status == 'completed')
        timeouts = sum(1 for r in self.results if r.status == 'timeout')
        errors = sum(1 for r in self.results if r.status.startswith('error'))

        print(f"📊 Total websites: {len(self.results)} | ✅ Completed: {completed} | "
              f"⏰ Timeouts: {timeouts} | ❌ Errors: {errors}")
        print(f"📧 Total emails: {total_emails} | 💼 Total career pages: {total_career_links}")

        print(f"\n📋 RESULTS BY WEBSITE:")
        for i, result in enumerate(self.results, 1):
            status_symbol = {
                'completed': '✅',
                'timeout': '⏰'
            }.get(result.status.split(':')[0], '❌')
            
            print(f"{i:2d}. {status_symbol} {result.website}")
            print(f"     📧 Emails: {len(result.emails)} | 💼 Career Pages: {len(result.career_links)} | "
                  f"📄 Pages: {result.pages_crawled} | ⏱️ Time: {result.time_taken:.1f}s")

            if result.emails:
                sample = list(result.emails)[:3]
                more = '...' if len(result.emails) > 3 else ''
                print(f"     📧 Emails: {', '.join(sample)}{more}")

            for career_link in result.career_links:
                print(f"     💼 Career: {career_link}")
        print("="*80)

    def export_results_to_csv(self, filename: str = None) -> str:
        """
        Export all crawling results to a CSV file.
        
        Creates a structured CSV with columns for:
        - Website URL
        - Email count and list
        - Career link count and list
        - Pages crawled, status, and timing
        
        Args:
            filename (str): Output filename (auto-generated if None)
            
        Returns:
            str: Path to the exported CSV file
        """
        if not filename:
            filename = f"email_career_results_{int(time.time())}.csv"

        export_data = []
        for result in self.results:
            export_data.append({
                'website': result.website,
                'email_count': len(result.emails),
                'emails': ', '.join(result.emails),
                'career_count': len(result.career_links),
                'career_links': ', '.join(result.career_links),
                'pages_crawled': result.pages_crawled,
                'status': result.status,
                'time_taken': result.time_taken
            })

        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        print(f"📁 Results exported to: {filename}")
        return filename

# Example usage and main execution functions
async def process_csv_file(csv_file_path: str) -> Tuple[List[CrawlResult], str]:
    """
    Convenience function to process a CSV file and export results.
    
    Args:
        csv_file_path (str): Path to CSV file containing websites
        
    Returns:
        Tuple of (results_list, export_filename)
    """
    crawler = EmailCareerWebCrawler(csv_file_path)
    await crawler.process_all_websites_from_csv()
    export_file = crawler.export_results_to_csv()
    return crawler.results, export_file

async def process_single_website(url: str) -> CrawlResult:
    """
    Convenience function to process a single website.
    
    Args:
        url (str): Website URL to crawl
        
    Returns:
        CrawlResult: Results for the single website
    """
    crawler = EmailCareerWebCrawler()
    return await crawler.process_single_website(url)

# Main execution example
async def main():
    """Example usage of the EmailCareerWebCrawler"""
    csv_file_path = "30-businesses.csv"

    try:
        # Process all websites from CSV
        results, export_file = await process_csv_file(csv_file_path)
        print(f"\n🎉 Processing complete! Results saved to: {export_file}")

        # Example: Process single website
        # single_result = await process_single_website("https://example.com")

        # Show detailed results for first few websites
        for result in results[:2]:
            print(f"\n🌐 {result.website} | Status: {result.status}")
            print(f"📧 Emails: {list(result.emails)}")
            print(f"💼 Career links: {list(result.career_links)}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())