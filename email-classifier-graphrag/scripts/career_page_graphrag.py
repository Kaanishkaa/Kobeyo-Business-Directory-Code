import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from typing import List, Dict, Tuple
from urllib.parse import urlparse
import re

class CareerPageClassifier:
    def __init__(self, model_path='./models/career_classifier'):
        print("🤖 Loading career page classifier...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        # Expanded career page patterns
        self.career_patterns = {
            # Original patterns
            'careers': 0.95,
            'career': 0.95,
            'jobs': 0.90,
            'job': 0.85,
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
            'were-hiring': 0.90,
            'apply': 0.75,
            'openings': 0.85,
            # New patterns
            'join': 0.90,
            'team': 0.60,  # Lower confidence as it could be "our team" page
            'work': 0.70,
            'positions': 0.85,
            'recruiter': 0.90,
            'recruiting': 0.90,
            'job-board': 0.95,
            'job-postings': 0.95,
            'careers-at': 0.95,
            'work-here': 0.85,
            'join-the-team': 0.90,
            'come-work': 0.85,
            'now-hiring': 0.90,
            'job-listings': 0.95,
            'career-portal': 0.95,
            'employment-opportunities': 0.95,
        }
        
    def extract_url_features(self, url: str) -> Dict:
        """Extract features from URL with better pattern matching"""
        parsed = urlparse(url)
        path = parsed.path.strip('/').lower()
        
        # Clean the path for better matching
        path_clean = path.replace('-', ' ').replace('_', ' ').replace('/', ' ')
        
        # Extract path components
        path_parts = [p for p in path.split('/') if p]
        
        # Check for career patterns
        url_score = 0
        found_patterns = []
        
        # Check each pattern
        for pattern, weight in self.career_patterns.items():
            pattern_words = pattern.replace('-', ' ').split()
            
            # Check if all words in pattern appear in path
            if all(word in path_clean for word in pattern_words):
                url_score += weight
                found_patterns.append(pattern)
            # Also check individual path parts
            else:
                for part in path_parts:
                    part_clean = part.replace('-', ' ').replace('_', ' ')
                    if all(word in part_clean for word in pattern_words):
                        url_score += weight * 0.8  # Slightly lower score for partial matches
                        found_patterns.append(f"{pattern}(partial)")
                        break
        
        # Special case for "join our team" type patterns
        if 'join' in path_clean and ('team' in path_clean or 'us' in path_clean):
            if 'join-our-team' not in found_patterns:
                url_score += 0.85
                found_patterns.append('join-team-pattern')
        
        # Check subdomain
        if parsed.netloc.startswith(('careers.', 'jobs.', 'hiring.', 'recruitment.')):
            url_score += 0.9
            found_patterns.append('career-subdomain')
        
        # Normalize score
        url_score = min(url_score, 1.0)
        
        return {
            'path': path,
            'path_clean': path_clean,
            'path_parts': path_parts,
            'url_score': url_score,
            'found_patterns': list(set(found_patterns))  # Remove duplicates
        }
    
    def classify_url(self, url: str, page_title: str = None, meta_description: str = None) -> Dict:
        """Classify a single URL as career page or not"""
        # Extract URL features
        url_features = self.extract_url_features(url)
        
        # Generate default title/description if not provided
        if not page_title:
            # Create title from URL path
            if url_features['path_parts']:
                page_title = ' '.join(url_features['path_parts'][-1].replace('-', ' ').title() for part in url_features['path_parts'])
            else:
                page_title = "Page"
        
        if not meta_description:
            meta_description = f"Page content from {url_features['path_clean']}"
        
        # Prepare input text
        input_text = f"URL: {url_features['path_clean']} Title: {page_title} Description: {meta_description}"
        
        # Get model prediction
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=128, 
            truncation=True, 
            padding='max_length'
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get prediction
        model_pred_career = probabilities[0][1].item() > probabilities[0][0].item()
        model_confidence = probabilities[0][1].item() if model_pred_career else probabilities[0][0].item()
        
        # Smart combination of URL patterns and model prediction
        if url_features['url_score'] >= 0.8:
            # Strong URL pattern - likely career page
            if model_confidence < 0.5:
                # Model disagrees strongly
                final_prediction = True  # Trust URL pattern
                final_confidence = url_features['url_score'] * 0.8
            else:
                # Model agrees or is uncertain
                final_prediction = True
                final_confidence = max(url_features['url_score'], model_confidence)
        elif url_features['url_score'] >= 0.5:
            # Moderate URL pattern
            if model_pred_career:
                final_prediction = True
                final_confidence = (url_features['url_score'] + model_confidence) / 2
            else:
                # Need both to agree for career page
                final_prediction = False
                final_confidence = model_confidence
        else:
            # Weak or no URL pattern - trust model
            final_prediction = model_pred_career
            final_confidence = model_confidence
        
        return {
            'url': url,
            'is_career_page': final_prediction,
            'confidence': final_confidence,
            'model_confidence': model_confidence,
            'model_prediction': model_pred_career,
            'url_pattern_score': url_features['url_score'],
            'found_patterns': url_features['found_patterns'],
            'title_used': page_title,
            'description_used': meta_description,
            'requires_content': model_confidence < 0.6 and url_features['url_score'] < 0.7
        }
    
    def rank_career_pages(self, urls: List[Dict]) -> List[Dict]:
        """Rank multiple URLs to find the most likely career page"""
        scored_urls = []
        
        for url_data in urls:
            if isinstance(url_data, str):
                url_data = {'url': url_data}
            
            result = self.classify_url(
                url_data['url'],
                url_data.get('title'),
                url_data.get('description')
            )
            
            # Calculate final score with bonuses
            if result['is_career_page']:
                # Base score
                final_score = result['confidence']
                
                # Pattern-specific bonuses
                if any(p in ['careers', 'career', 'job-listings', 'career-portal'] 
                      for p in result['found_patterns']):
                    final_score += 0.1
                elif any(p in ['jobs', 'openings', 'positions'] 
                        for p in result['found_patterns']):
                    final_score += 0.05
                
                # URL structure bonus
                if 'career-subdomain' in result['found_patterns']:
                    final_score += 0.05
                
                # Penalize overly complex URLs (likely not main career page)
                url_parts = urlparse(url_data['url']).path.strip('/').split('/')
                if len(url_parts) > 3:
                    final_score -= 0.1
            else:
                final_score = result['confidence'] * 0.1  # Heavily penalize non-career pages
            
            result['final_score'] = max(0, min(final_score, 1.0))
            scored_urls.append(result)
        
        # Sort by score
        scored_urls.sort(key=lambda x: x['final_score'], reverse=True)
        
        return scored_urls

# Test function
def test_career_classifier():
    classifier = CareerPageClassifier()
    
    print("\n🧪 Testing Enhanced Career Page Classifier")
    print("=" * 80)
    
    # Test cases including the ones you mentioned
    test_cases = [
        # Magnolia Bakery case
        {
            'url': 'https://www.magnoliabakery.com/pages/join-our-team',
            'title': 'Join Our Team',
            'description': 'Career opportunities at Magnolia Bakery'
        },
        {
            'url': 'http://www.magnoliabakery.com/pages/join-our-team',
            'title': None,  # Test without title
            'description': None
        },
        # Standard patterns
        {
            'url': 'https://www.company.com/careers',
            'title': 'Careers',
            'description': 'Join our growing team'
        },
        {
            'url': 'https://www.company.com/about/careers',
            'title': 'Career Opportunities',
            'description': 'Work with us'
        },
        # Edge cases
        {
            'url': 'https://www.company.com/team',
            'title': 'Our Team',
            'description': 'Meet our team members'
        },
        {
            'url': 'https://www.company.com/join',
            'title': 'Join Us',
            'description': 'Become part of our community'
        }
    ]
    
    print("Individual URL Tests:")
    for test in test_cases:
        result = classifier.classify_url(test['url'], test.get('title'), test.get('description'))
        
        emoji = "✅" if result['is_career_page'] else "❌"
        print(f"\n{emoji} URL: {test['url']}")
        if test.get('title'):
            print(f"   Title: {test['title']}")
        print(f"   Is Career Page: {result['is_career_page']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   URL Pattern Score: {result['url_pattern_score']:.2%}")
        print(f"   Model Confidence: {result['model_confidence']:.2%}")
        print(f"   Patterns Found: {', '.join(result['found_patterns']) if result['found_patterns'] else 'None'}")

if __name__ == "__main__":
    test_career_classifier()