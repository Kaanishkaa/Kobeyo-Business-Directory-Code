import json
import random
from datetime import datetime
from typing import List, Dict

class CareerPageDatasetCreator:
    def __init__(self):
        # Clear career page patterns
        self.CAREER_PATTERNS = {
            # High confidence patterns
            'careers': 0.95,
            'career': 0.95,
            'jobs': 0.90,
            'job-openings': 0.90,
            'join-us': 0.85,
            'join-our-team': 0.90,
            'work-with-us': 0.85,
            'work-at': 0.85,
            'employment': 0.85,
            'opportunities': 0.80,
            'hiring': 0.85,
            'recruitment': 0.85,
            'open-positions': 0.90,
            'current-openings': 0.90,
            'vacancies': 0.85,
            'job-opportunities': 0.90,
            'career-opportunities': 0.95,
            'were-hiring': 0.90,
            'join-team': 0.85,
            'work-here': 0.85,
            'apply-now': 0.80,
            'available-positions': 0.90,
        }
        
        # Non-career patterns (often confused)
        self.NON_CAREER_PATTERNS = {
            # About/Company pages
            'about': 0.95,
            'about-us': 0.95,
            'our-story': 0.90,
            'company': 0.85,
            'who-we-are': 0.90,
            'mission': 0.85,
            'values': 0.85,
            'team': 0.80,  # Could be either
            'our-team': 0.85,
            'leadership': 0.90,
            'management': 0.85,
            'board': 0.90,
            'investors': 0.90,
            
            # Contact/Support
            'contact': 0.95,
            'contact-us': 0.95,
            'support': 0.90,
            'help': 0.90,
            'faq': 0.95,
            'customer-service': 0.95,
            
            # Products/Services
            'products': 0.95,
            'services': 0.95,
            'solutions': 0.90,
            'pricing': 0.95,
            'features': 0.90,
            'portfolio': 0.85,
            'case-studies': 0.90,
            'testimonials': 0.90,
            
            # Legal/Policy
            'privacy': 0.95,
            'privacy-policy': 0.95,
            'terms': 0.95,
            'legal': 0.95,
            'disclaimer': 0.95,
            
            # News/Blog
            'blog': 0.95,
            'news': 0.90,
            'press': 0.90,
            'media': 0.85,
            'events': 0.85,
            'resources': 0.85,
        }
        
        # Common domains
        self.DOMAINS = [
            'techcorp.com', 'globaltech.com', 'innovate.io', 'startup.com',
            'enterprise.com', 'solutions.net', 'digital.io', 'software.dev',
            'consulting.com', 'agency.io', 'studio.com', 'group.com',
            'corporation.net', 'company.com', 'business.io', 'firm.com'
        ]
        
    def generate_url(self, pattern: str, domain: str, variation: int = 0) -> str:
        """Generate URL variations"""
        variations = [
            f"https://www.{domain}/{pattern}",
            f"https://www.{domain}/{pattern}/",
            f"https://www.{domain}/en/{pattern}",
            f"https://www.{domain}/us/{pattern}",
            f"https://www.{domain}/pages/{pattern}",
            f"https://{pattern}.{domain}",
            f"https://www.{domain}/{pattern}.html",
            f"https://www.{domain}/{pattern}/index",
            f"https://www.{domain}/company/{pattern}",
            f"https://www.{domain}/about/{pattern}",
        ]
        
        if variation < len(variations):
            return variations[variation]
        return variations[0]
    
    def create_dataset(self, size_per_category: int = 500) -> List[Dict]:
        """Create balanced dataset of career and non-career pages"""
        dataset = []
        
        # Generate career pages
        print("🎯 Generating career page URLs...")
        for pattern, confidence in self.CAREER_PATTERNS.items():
            for _ in range(min(25, size_per_category // len(self.CAREER_PATTERNS))):
                domain = random.choice(self.DOMAINS)
                variation = random.randint(0, 9)
                url = self.generate_url(pattern, domain, variation)
                
                dataset.append({
                    'url': url,
                    'is_career_page': True,
                    'confidence': confidence,
                    'pattern': pattern,
                    'domain': domain,
                    'page_title': self.generate_page_title(pattern, True),
                    'meta_description': self.generate_meta_description(pattern, True)
                })
        
        # Generate non-career pages
        print("📄 Generating non-career page URLs...")
        for pattern, confidence in self.NON_CAREER_PATTERNS.items():
            for _ in range(min(25, size_per_category // len(self.NON_CAREER_PATTERNS))):
                domain = random.choice(self.DOMAINS)
                variation = random.randint(0, 9)
                url = self.generate_url(pattern, domain, variation)
                
                dataset.append({
                    'url': url,
                    'is_career_page': False,
                    'confidence': confidence,
                    'pattern': pattern,
                    'domain': domain,
                    'page_title': self.generate_page_title(pattern, False),
                    'meta_description': self.generate_meta_description(pattern, False)
                })
        
        # Add some edge cases
        print("🔀 Adding edge cases...")
        edge_cases = [
            {
                'url': 'https://www.company.com/team#careers',
                'is_career_page': True,
                'confidence': 0.7,
                'pattern': 'team-careers',
                'domain': 'company.com',
                'page_title': 'Our Team - Join Us',
                'meta_description': 'Meet our team and explore career opportunities'
            },
            {
                'url': 'https://www.company.com/about/culture-and-careers',
                'is_career_page': True,
                'confidence': 0.8,
                'pattern': 'culture-careers',
                'domain': 'company.com',
                'page_title': 'Culture & Careers',
                'meta_description': 'Learn about our culture and open positions'
            },
            {
                'url': 'https://www.company.com/news/we-are-hiring',
                'is_career_page': False,  # News article, not careers page
                'confidence': 0.6,
                'pattern': 'news-hiring',
                'domain': 'company.com',
                'page_title': 'News: We Are Hiring',
                'meta_description': 'Latest news about our expansion'
            }
        ]
        dataset.extend(edge_cases)
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        return dataset
    
    def generate_page_title(self, pattern: str, is_career: bool) -> str:
        """Generate realistic page titles"""
        if is_career:
            titles = [
                f"Careers - Join Our Team",
                f"Jobs & Career Opportunities",
                f"Work With Us - Open Positions",
                f"Current Job Openings",
                f"Join Us - Career Opportunities",
                f"Employment Opportunities",
                f"We're Hiring - Open Roles",
                f"Career Portal",
            ]
        else:
            titles = [
                f"About Us - Our Story",
                f"Contact Us",
                f"Our Products and Services",
                f"Company Blog",
                f"Privacy Policy",
                f"Customer Support",
                f"Our Team",
                f"News & Updates",
            ]
        return random.choice(titles)
    
    def generate_meta_description(self, pattern: str, is_career: bool) -> str:
        """Generate meta descriptions"""
        if is_career:
            descriptions = [
                "Explore career opportunities and join our growing team",
                "View current job openings and apply online",
                "Build your career with us - see available positions",
                "Join our team - we're hiring talented professionals",
                "Discover exciting career opportunities",
            ]
        else:
            descriptions = [
                "Learn more about our company and mission",
                "Get in touch with our team",
                "Explore our products and services",
                "Read our latest news and updates",
                "Our commitment to your privacy",
            ]
        return random.choice(descriptions)
    
    def save_dataset(self, dataset: List[Dict], output_dir: str = "./data"):
        """Save dataset to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Split into train/test
        split_idx = int(len(dataset) * 0.8)
        train_data = dataset[:split_idx]
        test_data = dataset[split_idx:]
        
        # Save files
        with open(f"{output_dir}/career_pages_train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
            
        with open(f"{output_dir}/career_pages_test.json", 'w') as f:
            json.dump(test_data, f, indent=2)
            
        with open(f"{output_dir}/career_pages_full.json", 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Create metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'total_urls': len(dataset),
            'career_pages': len([d for d in dataset if d['is_career_page']]),
            'non_career_pages': len([d for d in dataset if not d['is_career_page']]),
            'train_size': len(train_data),
            'test_size': len(test_data),
            'patterns': {
                'career': list(self.CAREER_PATTERNS.keys()),
                'non_career': list(self.NON_CAREER_PATTERNS.keys())
            }
        }
        
        with open(f"{output_dir}/career_dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Dataset saved successfully!")
        print(f"   Total URLs: {len(dataset)}")
        print(f"   Career pages: {metadata['career_pages']}")
        print(f"   Non-career pages: {metadata['non_career_pages']}")
        print(f"   Train set: {len(train_data)}")
        print(f"   Test set: {len(test_data)}")

# Main execution
if __name__ == "__main__":
    creator = CareerPageDatasetCreator()
    
    # Create dataset
    dataset = creator.create_dataset(size_per_category=500)
    
    # Save to files
    creator.save_dataset(dataset)
    
    # Show samples
    print("\n📋 Sample career pages:")
    career_samples = [d for d in dataset if d['is_career_page']][:5]
    for sample in career_samples:
        print(f"  ✓ {sample['url']}")
        print(f"    Pattern: {sample['pattern']} (confidence: {sample['confidence']})")
    
    print("\n📋 Sample non-career pages:")
    non_career_samples = [d for d in dataset if not d['is_career_page']][:5]
    for sample in non_career_samples:
        print(f"  ✗ {sample['url']}")
        print(f"    Pattern: {sample['pattern']}")
