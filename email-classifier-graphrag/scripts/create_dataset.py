#!/usr/bin/env python3
"""
Email Dataset Creator and Cleaner
Creates a high-quality dataset for HR vs Sales email classification
Designed for GraphRAG implementation with Neo4j
"""

import json
import random
import re
from datetime import datetime
from typing import List, Dict, Tuple
import os

# Define clear patterns for HR and Sales
HR_PATTERNS = {
    # Core HR patterns
    'careers': 0.95,
    'hiring': 0.95,
    'recruitment': 0.95,
    'hr': 0.95,
    'humanresources': 0.95,
    'talent': 0.90,
    'jobs': 0.90,
    'resume': 0.95,
    'cv': 0.95,
    'application': 0.90,
    'apply': 0.85,
    'peopleops': 0.95,
    'workwithus': 0.90,
    'joinus': 0.90,
    'hiringteam': 0.95,
    'recruiting': 0.95,
    'talentacquisition': 0.95,
    'staffing': 0.90,
    'onboarding': 0.85,
    'interview': 0.85,
    'candidate': 0.85,
    'position': 0.80,
    'vacancy': 0.85,
    'opportunity': 0.75,
    'team': 0.70,  # Could be either
    'culture': 0.80,
    'benefits': 0.80,
    'compensation': 0.85,
    'payroll': 0.85,
    'employee': 0.85,
    'workforce': 0.85
}

SALES_PATTERNS = {
    # Core Sales patterns
    'sales': 0.95,
    'orders': 0.95,
    'purchase': 0.95,
    'deals': 0.95,
    'quotes': 0.95,
    'pricing': 0.90,
    'customer': 0.85,
    'client': 0.85,
    'account': 0.85,
    'business': 0.80,
    'revenue': 0.90,
    'leads': 0.90,
    'prospects': 0.90,
    'opportunities': 0.85,
    'pipeline': 0.90,
    'contract': 0.85,
    'proposal': 0.85,
    'demo': 0.85,
    'trial': 0.80,
    'subscription': 0.85,
    'renewal': 0.85,
    'upsell': 0.90,
    'crosssell': 0.90,
    'commission': 0.85,
    'quota': 0.85,
    'target': 0.80,
    'forecast': 0.80,
    'crm': 0.85,
    'deal': 0.90,
    'negotiate': 0.85,
    'closing': 0.85,
    'prospect': 0.85,
    'lead': 0.85,
    'convert': 0.80,
    'pitch': 0.85,
    'solution': 0.75,
    'partnership': 0.80,
    'vendor': 0.80,
    'supplier': 0.80,
    'procurement': 0.85,
    'rfp': 0.85,
    'bid': 0.85
}

# Ambiguous patterns that need context
AMBIGUOUS_PATTERNS = {
    'info': 0.5,
    'contact': 0.5,
    'admin': 0.5,
    'office': 0.5,
    'support': 0.3,  # Usually not HR/Sales
    'help': 0.3,
    'team': 0.5,
    'manager': 0.6,
    'director': 0.6,
    'coordinator': 0.6,
    'assistant': 0.6,
    'department': 0.5,
    'general': 0.4,
    'main': 0.4,
    'central': 0.4,
    'corporate': 0.5,
    'company': 0.5,
    'business': 0.6,
    'operations': 0.5,
    'strategy': 0.5,
    'planning': 0.5,
    'development': 0.6,
    'growth': 0.6,
    'success': 0.6,
    'relations': 0.6,
    'management': 0.6,
    'services': 0.5,
    'solutions': 0.6,
    'consulting': 0.6,
    'advisory': 0.6
}

# Patterns to exclude (not HR or Sales)
EXCLUDE_PATTERNS = {
    'noreply': 0.95,
    'no-reply': 0.95,
    'donotreply': 0.95,
    'mailer': 0.90,
    'notification': 0.90,
    'alert': 0.90,
    'automated': 0.90,
    'system': 0.85,
    'bounce': 0.95,
    'unsubscribe': 0.90,
    'billing': 0.85,
    'invoice': 0.85,
    'payment': 0.80,
    'accounting': 0.85,
    'finance': 0.75,
    'legal': 0.85,
    'compliance': 0.85,
    'security': 0.85,
    'it': 0.85,
    'tech': 0.70,
    'engineering': 0.75,
    'product': 0.70,
    'design': 0.70,
    'marketing': 0.75,
    'press': 0.85,
    'media': 0.80,
    'news': 0.85,
    'blog': 0.85,
    'newsletter': 0.90,
    'updates': 0.85,
    'announcements': 0.85,
    'events': 0.75,
    'webinar': 0.80,
    'training': 0.70,
    'support': 0.85,
    'help': 0.85,
    'service': 0.80,
    'desk': 0.80,
    'ticket': 0.85,
    'issue': 0.85,
    'bug': 0.90,
    'feedback': 0.80,
    'survey': 0.85,
    'research': 0.75,
    'data': 0.75,
    'analytics': 0.80,
    'report': 0.75,
    'admin': 0.70,
    'general': 0.70,
    'hello': 0.65,
    'hi': 0.65,
    'welcome': 0.70,
    'greetings': 0.70
}

# Common company domains for realistic data
DOMAINS = [
    'techcorp.com', 'innovate.io', 'globaltech.com', 'solutions.net',
    'enterprise.com', 'startup.io', 'ventures.co', 'digital.ai',
    'cloudware.com', 'systems.tech', 'software.dev', 'platform.io',
    'services.com', 'consulting.net', 'partners.co', 'group.com',
    'industries.com', 'corporation.net', 'company.com', 'business.io',
    'professional.com', 'global.net', 'international.com', 'worldwide.io',
    'nexus.tech', 'synergy.co', 'dynamics.com', 'solutions.ai',
    'innovations.io', 'technologies.com', 'digital.co', 'future.tech',
    'smart.io', 'data.ai', 'analytics.com', 'insights.net',
    'creative.co', 'design.io', 'studio.com', 'agency.net',
    'retail.com', 'shop.io', 'store.net', 'market.com',
    'finance.com', 'capital.io', 'invest.net', 'banking.com',
    'health.io', 'medical.com', 'care.net', 'wellness.co',
    'education.com', 'academy.io', 'institute.net', 'university.edu',
    'media.com', 'entertainment.io', 'broadcast.net', 'publishing.com',
    'realestate.com', 'properties.io', 'development.net', 'construction.com',
    'hospitality.com', 'hotels.io', 'restaurants.net', 'tourism.com',
    'transportation.com', 'logistics.io', 'shipping.net', 'delivery.com',
    'energy.com', 'power.io', 'renewable.net', 'utilities.com',
    'manufacturing.com', 'production.io', 'industrial.net', 'factory.com'
]

class EmailDatasetCreator:
    def __init__(self):
        self.hr_emails = []
        self.sales_emails = []
        self.excluded_emails = []
        self.all_emails = []
        
    def generate_email_variations(self, base_pattern: str, category: str, count: int = 5) -> List[Dict]:
        """Generate variations of an email pattern"""
        emails = []
        
        for i in range(count):
            # Add variations
            variations = [
                base_pattern,
                f"{base_pattern}{random.randint(1, 999)}",
                f"{base_pattern}.{random.choice(['dept', 'team', 'group', 'division', 'unit'])}",
                f"{base_pattern}_{random.choice(['us', 'global', 'main', 'central', 'primary'])}",
                f"{random.choice(['contact', 'reach', 'hello', 'info'])}.{base_pattern}",
            ]
            
            # Add compound variations for some patterns
            if category == "HR" and base_pattern in ['talent', 'hiring', 'recruit']:
                variations.extend([
                    f"{base_pattern}.acquisition",
                    f"{base_pattern}.management",
                    f"{base_pattern}.operations",
                    f"{base_pattern}.team",
                    f"{base_pattern}.dept"
                ])
            elif category == "Sales" and base_pattern in ['sales', 'business', 'account']:
                variations.extend([
                    f"{base_pattern}.development",
                    f"{base_pattern}.operations",
                    f"{base_pattern}.management",
                    f"{base_pattern}.team",
                    f"{base_pattern}.inquiries"
                ])
            
            email_local = random.choice(variations)
            domain = random.choice(DOMAINS)
            
            email_data = {
                "email": f"{email_local}@{domain}",
                "category": category,
                "confidence": self._calculate_confidence(email_local, category),
                "prefix_pattern": base_pattern,
                "domain": domain,
                "is_compound": '.' in email_local or '_' in email_local or '-' in email_local,
                "has_numbers": any(c.isdigit() for c in email_local),
                "created_at": datetime.now().isoformat()
            }
            
            emails.append(email_data)
        
        return emails
    
    def _calculate_confidence(self, email_local: str, category: str) -> float:
        """Calculate confidence score for classification"""
        email_clean = re.sub(r'[0-9_.-]', '', email_local.lower())
        
        if category == "HR":
            for pattern, conf in HR_PATTERNS.items():
                if pattern in email_clean:
                    return conf
        elif category == "Sales":
            for pattern, conf in SALES_PATTERNS.items():
                if pattern in email_clean:
                    return conf
        
        # Check for ambiguous patterns
        for pattern, conf in AMBIGUOUS_PATTERNS.items():
            if pattern in email_clean:
                return conf
        
        return 0.5  # Default uncertainty
    
    def create_balanced_dataset(self, size_per_category: int = 500) -> None:
        """Create a balanced dataset with high-quality examples"""
        
        # Generate HR emails
        print("🏢 Generating HR emails...")
        hr_count = 0
        for pattern in HR_PATTERNS.keys():
            if hr_count >= size_per_category:
                break
            variations_needed = min(10, size_per_category - hr_count)
            emails = self.generate_email_variations(pattern, "HR", variations_needed)
            self.hr_emails.extend(emails)
            hr_count += len(emails)
        
        # Generate Sales emails
        print("💼 Generating Sales emails...")
        sales_count = 0
        for pattern in SALES_PATTERNS.keys():
            if sales_count >= size_per_category:
                break
            variations_needed = min(10, size_per_category - sales_count)
            emails = self.generate_email_variations(pattern, "Sales", variations_needed)
            self.sales_emails.extend(emails)
            sales_count += len(emails)
        
        # Add some ambiguous cases for realism
        print("❓ Adding ambiguous cases...")
        ambiguous_count = int(size_per_category * 0.1)  # 10% ambiguous
        for pattern in list(AMBIGUOUS_PATTERNS.keys())[:ambiguous_count//2]:
            # Randomly assign to HR or Sales
            category = random.choice(["HR", "Sales"])
            emails = self.generate_email_variations(pattern, category, 2)
            if category == "HR":
                self.hr_emails.extend(emails)
            else:
                self.sales_emails.extend(emails)
        
        # Combine all emails
        self.all_emails = self.hr_emails + self.sales_emails
        random.shuffle(self.all_emails)
        
        print(f"\n✅ Dataset created successfully!")
        print(f"   - HR emails: {len(self.hr_emails)}")
        print(f"   - Sales emails: {len(self.sales_emails)}")
        print(f"   - Total emails: {len(self.all_emails)}")
    
    def add_metadata(self) -> None:
        """Add additional metadata for GraphRAG"""
        for email in self.all_emails:
            # Add relationship hints for graph building
            email['related_patterns'] = self._find_related_patterns(email['prefix_pattern'])
            email['industry_hint'] = self._guess_industry(email['domain'])
            email['priority_score'] = email['confidence'] * (0.9 if email['category'] in ['HR', 'Sales'] else 0.1)
    
    def _find_related_patterns(self, pattern: str) -> List[str]:
        """Find patterns that are related (for graph edges)"""
        related = []
        
        if pattern in HR_PATTERNS:
            # Find semantically related HR patterns
            if 'hire' in pattern or 'hiring' in pattern:
                related.extend(['recruitment', 'talent', 'careers', 'jobs'])
            elif 'talent' in pattern:
                related.extend(['hiring', 'recruitment', 'hr'])
            elif 'career' in pattern:
                related.extend(['jobs', 'opportunities', 'positions'])
        
        elif pattern in SALES_PATTERNS:
            # Find semantically related Sales patterns
            if 'sales' in pattern:
                related.extend(['deals', 'revenue', 'account', 'business'])
            elif 'customer' in pattern or 'client' in pattern:
                related.extend(['account', 'success', 'support'])
            elif 'order' in pattern:
                related.extend(['purchase', 'procurement', 'quotes'])
        
        return list(set(related))  # Remove duplicates
    
    def _guess_industry(self, domain: str) -> str:
        """Guess industry from domain name"""
        domain_lower = domain.lower()
        
        if any(word in domain_lower for word in ['tech', 'software', 'digital', 'ai', 'cloud', 'data']):
            return 'Technology'
        elif any(word in domain_lower for word in ['finance', 'capital', 'invest', 'bank']):
            return 'Finance'
        elif any(word in domain_lower for word in ['health', 'medical', 'care', 'wellness']):
            return 'Healthcare'
        elif any(word in domain_lower for word in ['retail', 'shop', 'store', 'market']):
            return 'Retail'
        elif any(word in domain_lower for word in ['consult', 'advisory', 'partners']):
            return 'Consulting'
        elif any(word in domain_lower for word in ['education', 'academy', 'university']):
            return 'Education'
        else:
            return 'General'
    
    def save_dataset(self, output_dir: str = "./data") -> None:
        """Save dataset in multiple formats"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full dataset
        with open(f"{output_dir}/email_dataset_full.json", 'w') as f:
            json.dump(self.all_emails, f, indent=2)
        
        # Save train/test split (80/20)
        split_idx = int(len(self.all_emails) * 0.8)
        train_data = self.all_emails[:split_idx]
        test_data = self.all_emails[split_idx:]
        
        with open(f"{output_dir}/email_dataset_train.json", 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(f"{output_dir}/email_dataset_test.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Save metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "total_emails": len(self.all_emails),
            "hr_emails": len(self.hr_emails),
            "sales_emails": len(self.sales_emails),
            "unique_domains": len(set(e['domain'] for e in self.all_emails)),
            "unique_patterns": len(set(e['prefix_pattern'] for e in self.all_emails)),
            "high_confidence_emails": len([e for e in self.all_emails if e['confidence'] > 0.8]),
            "low_confidence_emails": len([e for e in self.all_emails if e['confidence'] < 0.6]),
            "pattern_distribution": {
                "hr_patterns": list(HR_PATTERNS.keys()),
                "sales_patterns": list(SALES_PATTERNS.keys()),
                "ambiguous_patterns": list(AMBIGUOUS_PATTERNS.keys())
            }
        }
        
        with open(f"{output_dir}/dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save a sample for quick inspection
        sample_size = min(20, len(self.all_emails))
        sample_data = random.sample(self.all_emails, sample_size)
        
        with open(f"{output_dir}/email_dataset_sample.json", 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"\n💾 Dataset saved to {output_dir}/")
        print(f"   - email_dataset_full.json ({len(self.all_emails)} emails)")
        print(f"   - email_dataset_train.json ({len(train_data)} emails)")
        print(f"   - email_dataset_test.json ({len(test_data)} emails)")
        print(f"   - email_dataset_sample.json ({sample_size} emails)")
        print(f"   - dataset_metadata.json")

def clean_existing_dataset(file_path: str) -> List[Dict]:
    """Clean and fix an existing dataset"""
    print(f"\n🧹 Cleaning existing dataset from {file_path}...")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    cleaned_data = []
    fixes_made = 0
    
    for item in data:
        email = item['email']
        prefix = email.split('@')[0].lower()
        original_category = item.get('category', '')
        
        # Remove numbers and special chars for pattern matching
        prefix_clean = re.sub(r'[0-9_.-]', '', prefix)
        
        # Determine correct category
        correct_category = None
        confidence = 0.5
        
        # Check HR patterns
        for pattern, conf in HR_PATTERNS.items():
            if pattern in prefix_clean:
                correct_category = "HR"
                confidence = conf
                break
        
        # Check Sales patterns (higher priority for clear sales patterns)
        for pattern, conf in SALES_PATTERNS.items():
            if pattern in prefix_clean:
                if pattern in ['sales', 'orders', 'purchase', 'deals']:  # Strong sales indicators
                    correct_category = "Sales"
                    confidence = conf
                    break
                elif correct_category != "HR":  # Don't override HR if already set
                    correct_category = "Sales"
                    confidence = conf
        
        # Skip if it matches exclude patterns
        skip = False
        for pattern, conf in EXCLUDE_PATTERNS.items():
            if pattern in prefix_clean and conf > 0.8:
                skip = True
                break
        
        if skip:
            continue
        
        # If no clear pattern, use original category but mark as low confidence
        if correct_category is None:
            correct_category = original_category if original_category in ["HR", "Sales"] else "Unknown"
            confidence = 0.3
        
        # Check if we fixed an error
        if correct_category != original_category:
            fixes_made += 1
            print(f"   Fixed: {email} from {original_category} to {correct_category}")
        
        # Create cleaned entry
        cleaned_entry = {
            "email": email,
            "category": correct_category,
            "confidence": confidence,
            "prefix_pattern": prefix_clean,
            "domain": email.split('@')[1] if '@' in email else 'unknown',
            "original_category": original_category,
            "was_fixed": correct_category != original_category,
            "created_at": item.get('created_at', datetime.now().isoformat())
        }
        
        if correct_category in ["HR", "Sales"]:
            cleaned_data.append(cleaned_entry)
    
    print(f"\n✅ Cleaning complete!")
    print(f"   - Total emails processed: {len(data)}")
    print(f"   - Emails kept: {len(cleaned_data)}")
    print(f"   - Fixes made: {fixes_made}")
    print(f"   - Emails removed: {len(data) - len(cleaned_data)}")
    
    return cleaned_data

def main():
    """Main function to create or clean dataset"""
    print("🚀 Email Dataset Creator for HR/Sales Classification")
    print("=" * 50)
    
    # Check if we should clean existing dataset or create new one
    existing_dataset_path = "./email_training_dataset_for_pipeline.json"
    
    if os.path.exists(existing_dataset_path):
        print(f"\n📁 Found existing dataset at {existing_dataset_path}")
        choice = input("\nDo you want to:\n1. Clean existing dataset\n2. Create new dataset\nChoice (1/2): ")
        
        if choice == "1":
            cleaned_data = clean_existing_dataset(existing_dataset_path)
            
            # Save cleaned dataset
            output_dir = "./data"
            os.makedirs(output_dir, exist_ok=True)
            
            with open(f"{output_dir}/email_dataset_cleaned.json", 'w') as f:
                json.dump(cleaned_data, f, indent=2)
            
            print(f"\n💾 Cleaned dataset saved to {output_dir}/email_dataset_cleaned.json")
            return
    
    # Create new dataset
    creator = EmailDatasetCreator()
    
    # Get dataset size
    size = input("\nHow many emails per category (HR/Sales)? [default: 500]: ")
    size_per_category = int(size) if size.isdigit() else 500
    
    # Create dataset
    creator.create_balanced_dataset(size_per_category)
    creator.add_metadata()
    
    # Save dataset
    creator.save_dataset()
    
    # Display sample
    print("\n📋 Sample emails from dataset:")
    print("-" * 50)
    for i, email in enumerate(random.sample(creator.all_emails, min(10, len(creator.all_emails)))):
        print(f"{i+1}. {email['email']} -> {email['category']} (confidence: {email['confidence']:.2f})")

if __name__ == "__main__":
    main()