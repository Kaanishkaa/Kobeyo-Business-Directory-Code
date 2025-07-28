"""
skillTagger.py - Enhanced skill ID and name mapping for businesses
Optimized for main.py workflow integration
Automatically loads business groups from CSV and finds corresponding skill files
"""

import os
import re
import json
import time
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set

# Fuzzy matching for better skill detection
try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    print("⚠️ FuzzyWuzzy not available. Using exact matching only.")

# AI components for advanced skill classification
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from openai import RateLimitError
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configuration - Replace with your actual API key
OPENAI_API_KEY = "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"


class EnhancedSkillTagger:
    """Enhanced skill ID and name mapper with dynamic file loading from your data structure"""
    
    def __init__(self, 
                 business_groups_file: str = "Extracted_Business_Groups_and_Types.csv",
                 skill_files_folder: str = "Skill-tag csv",
                 enable_ai_classification: bool = True,
                 fuzzy_threshold: int = 85):
        
        self.business_groups_file = business_groups_file
        self.skill_files_folder = skill_files_folder
        self.enable_ai_classification = enable_ai_classification
        self.fuzzy_threshold = fuzzy_threshold
        
        # Data storage
        self.business_groups_mapping = {}  # business_group -> list of business types
        self.skill_mappings = {}  # business_group -> skill data
        self.business_type_to_group = {}  # business_type -> business_group (for quick lookup)
        self.ai_available = False
        
        print("🎯 Initializing Enhanced SkillTagger...")
        
        # Initialize AI components if enabled
        if enable_ai_classification:
            self.init_ai_components()
        
        # Load business groups and types mapping
        self.load_business_groups_mapping()
        
        # Load all available skill files
        self.load_all_skill_files()
        
        print(f"✅ SkillTagger ready! Loaded {len(self.business_groups_mapping)} groups, {len(self.skill_mappings)} skill files")

    def init_ai_components(self):
        """Initialize AI components for advanced skill classification"""
        if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-proj-") and len(OPENAI_API_KEY) < 50:
            print("⚠️ OpenAI API key not properly set. AI classification disabled.")
            return
        
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        
        if LANGCHAIN_AVAILABLE:
            try:
                self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
                
                self.prompt_template = ChatPromptTemplate.from_template("""
You are an expert business classification analyst. Analyze the website text based on the provided rules.

**PRIORITY 1: GLOBAL SPECIAL RULES**
{special_rules}

**PRIORITY 2: INDUSTRY-SPECIFIC RULES**
{dynamic_rules}

**IMPORTANT INSTRUCTIONS:**
- Read the website text carefully
- Apply the rules in order of priority
- Only return skills that have clear evidence in the text
- Be conservative - if unsure, don't include the skill

**OUTPUT FORMAT:**
Return a single JSON object with one key: "applied_skills". The value must be a list of strings, with each string being an EXACT match from the rule names provided above. If no skills apply, return an empty list.

Example:
{{"applied_skills": ["skill1", "skill2"]}}

**WEBSITE TEXT TO ANALYZE:**
{context}
""")
                
                self.chain = self.prompt_template | self.llm | StrOutputParser()
                self.ai_available = True
                print("✅ AI classification enabled")
                
            except Exception as e:
                print(f"⚠️ AI initialization failed: {e}")
                
        elif OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
                self.ai_available = True
                print("✅ OpenAI client enabled")
            except Exception as e:
                print(f"⚠️ OpenAI initialization failed: {e}")

    def load_business_groups_mapping(self):
        """Load business groups and types from CSV file"""
        try:
            if not os.path.exists(self.business_groups_file):
                print(f"❌ Business groups file not found: {self.business_groups_file}")
                return
            
            print(f"📚 Loading business groups from {self.business_groups_file}")
            
            # Try different encodings
            for encoding in ['utf-8', 'cp1252', 'latin1']:
                try:
                    df = pd.read_csv(self.business_groups_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                df = pd.read_csv(self.business_groups_file)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Validate required columns
            if 'Business Group' not in df.columns or 'Business Type' not in df.columns:
                print(f"❌ Required columns not found. Expected: 'Business Group', 'Business Type'")
                print(f"Found columns: {list(df.columns)}")
                return
            
            # Process each row
            for _, row in df.iterrows():
                business_group = str(row['Business Group']).strip()
                business_types_str = str(row['Business Type']).strip()
                
                if not business_group or business_group == 'nan':
                    continue
                
                # Parse business types (comma-separated)
                business_types = []
                if business_types_str and business_types_str != 'nan':
                    raw_types = [t.strip() for t in business_types_str.split(',')]
                    business_types = [t for t in raw_types if t and t != 'nan']
                
                # Store mapping
                self.business_groups_mapping[business_group] = business_types
                
                # Create reverse mapping for quick lookup
                for business_type in business_types:
                    self.business_type_to_group[business_type.lower()] = business_group
            
            print(f"✅ Loaded {len(self.business_groups_mapping)} business groups")
                
        except Exception as e:
            print(f"❌ Error loading business groups mapping: {e}")

    def find_skill_file(self, business_group: str) -> Optional[str]:
        """Find the skill file for a business group, trying multiple naming variations"""
        if not os.path.exists(self.skill_files_folder):
            print(f"❌ Skill files folder not found: {self.skill_files_folder}")
            return None
        
        available_files = os.listdir(self.skill_files_folder)
        
        # Try different naming patterns
        patterns = [
            f"skill tags  {business_group}.csv",
            f"skill_tags_{business_group.replace(' ', '_')}.csv",
            f"{business_group.replace(' ', '_').lower()}.xlsx",
            f"{business_group.replace(' & ', '_and_').replace(' ', '_').lower()}.xlsx"
        ]
        
        for pattern in patterns:
            if pattern in available_files:
                return os.path.join(self.skill_files_folder, pattern)
        
        # Try fuzzy matching with available files
        if FUZZYWUZZY_AVAILABLE:
            best_match = None
            best_score = 0
            
            for filename in available_files:
                if filename.endswith(('.xlsx', '.csv')):
                    file_base = filename.rsplit('.', 1)[0]
                    file_base = file_base.replace('skill tags  ', '').replace('skill_tags_', '')
                    
                    score = fuzz.ratio(business_group.lower(), file_base.lower())
                    if score > best_score and score > 60:
                        best_score = score
                        best_match = filename
            
            if best_match:
                return os.path.join(self.skill_files_folder, best_match)
        
        return None

    def load_all_skill_files(self):
        """Load skill files for all business groups"""
        if not self.business_groups_mapping:
            print("⚠️ No business groups loaded. Cannot load skill files.")
            return
        
        successful_loads = 0
        for business_group in self.business_groups_mapping.keys():
            skill_file_path = self.find_skill_file(business_group)
            
            if skill_file_path:
                if self.load_single_skill_file(skill_file_path, business_group):
                    successful_loads += 1

    def load_single_skill_file(self, filepath: str, business_group: str) -> bool:
        """Load skill mappings from a single skill file"""
        try:
            # Read Excel or CSV file
            if filepath.endswith(".xlsx"):
                df = pd.read_excel(filepath)
            else:
                for encoding in ['utf-8', 'cp1252', 'latin1']:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    df = pd.read_csv(filepath)
            
            # Clean column names and standardize
            df.columns = df.columns.str.strip()
            
            column_mapping = {
                'Skills Tags': ['Skills Tags', 'Skill Tags', 'Tag', 'Skills Tag'],
                'Skills IDs': ['Skills IDs', 'Skill IDs', 'ID', 'Skills ID', 'Skill ID'],
                'Skills Names': ['Skills Names', 'Skill Names', 'Name', 'Skills Name', 'Skill Name'],
                'Prompt Rule': ['Prompt Rule', 'Rule', 'Prompt', 'AI Rule']
            }
            
            for standard_col, variations in column_mapping.items():
                for variation in variations:
                    if variation in df.columns and standard_col not in df.columns:
                        df = df.rename(columns={variation: standard_col})
                        break
            
            # Build skill mapping
            skill_mapping = {}
            ai_rules = {}
            header_notes = {}
            
            for _, row in df.iterrows():
                skill_tag = str(row.get('Skills Tags', '')).strip()
                skill_ids = str(row.get('Skills IDs', '')).strip()
                skill_names = str(row.get('Skills Names', '')).strip()
                prompt_rule = str(row.get('Prompt Rule', '')).strip()
                
                # Handle special header rows
                if skill_tag.upper() in ['BUSINESS TYPES', 'SPECIAL TAGS', 'IMPORTANT NOTES']:
                    header_notes[skill_tag.upper()] = prompt_rule
                    continue
                
                # Skip invalid rows
                if (not skill_tag or skill_tag.startswith('#') or 
                    skill_tag in ['nan', 'None', ''] or not pd.notna(skill_tag)):
                    continue
                
                # Clean up the data
                clean_ids = skill_ids if skill_ids not in ['nan', '', 'None'] and pd.notna(skill_ids) else ''
                clean_names = skill_names if skill_names not in ['nan', '', 'None'] and pd.notna(skill_names) else skill_tag
                clean_prompt = prompt_rule if prompt_rule not in ['nan', '', 'None'] and pd.notna(prompt_rule) else ''
                
                # Store mapping
                skill_key = skill_tag.lower().strip()
                skill_mapping[skill_key] = {
                    'original_tag': skill_tag,
                    'skill_ids': clean_ids,
                    'skill_names': clean_names,
                    'prompt_rule': clean_prompt
                }
                
                if clean_prompt:
                    ai_rules[skill_tag] = clean_prompt
            
            if skill_mapping:
                self.skill_mappings[business_group] = {
                    'skills': skill_mapping,
                    'ai_rules': ai_rules,
                    'header_notes': header_notes,
                    'file_path': filepath
                }
                return True
                
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return False

    def find_business_group_for_type(self, business_type: str) -> Optional[str]:
        """Find the business group for a given business type"""
        if not business_type:
            return None
        
        business_type_clean = business_type.lower().strip()
        
        # Direct lookup
        if business_type_clean in self.business_type_to_group:
            return self.business_type_to_group[business_type_clean]
        
        # Fuzzy matching
        if FUZZYWUZZY_AVAILABLE:
            best_match = None
            best_score = 0
            
            for known_type, group in self.business_type_to_group.items():
                score = fuzz.partial_ratio(business_type_clean, known_type)
                if score > best_score and score > self.fuzzy_threshold:
                    best_score = score
                    best_match = group
            
            if best_match:
                return best_match
        
        # Word-based matching
        business_words = set(business_type_clean.split())
        for known_type, group in self.business_type_to_group.items():
            known_words = set(known_type.split())
            if business_words.intersection(known_words):
                return group
        
        return None

    def find_matching_skills_basic(self, business_type: str, business_group: str = None) -> Tuple[str, str, str]:
        """Basic skill matching using fuzzy string matching"""
        if not business_type:
            return "", "", ""
        
        # Find business group if not provided
        if not business_group:
            business_group = self.find_business_group_for_type(business_type)
            if not business_group:
                return "", "", ""
        
        # Check if we have skills for this business group
        if business_group not in self.skill_mappings:
            return "", "", business_group
        
        skill_mapping = self.skill_mappings[business_group]['skills']
        business_type_clean = business_type.lower().strip()
        
        # Try exact match first
        if business_type_clean in skill_mapping:
            mapping = skill_mapping[business_type_clean]
            return mapping['skill_ids'], mapping['skill_names'], business_group
        
        # Try fuzzy matching
        if FUZZYWUZZY_AVAILABLE:
            best_match = None
            best_score = 0
            
            for skill_tag, mapping in skill_mapping.items():
                score = max(
                    fuzz.partial_ratio(business_type_clean, skill_tag),
                    fuzz.partial_ratio(skill_tag, business_type_clean)
                )
                if score > best_score and score > self.fuzzy_threshold:
                    best_score = score
                    best_match = mapping
            
            if best_match:
                return best_match['skill_ids'], best_match['skill_names'], business_group
        
        # Try word-based matching
        business_words = set(business_type_clean.split())
        for skill_tag, mapping in skill_mapping.items():
            skill_words = set(skill_tag.split())
            if business_words.intersection(skill_words):
                return mapping['skill_ids'], mapping['skill_names'], business_group
        
        return "", "", business_group

    def classify_with_ai(self, website_content: str, business_group: str) -> Tuple[List[str], str]:
        """AI-powered skill classification using website content"""
        if not self.ai_available or not website_content.strip():
            return [], ""
        
        if business_group not in self.skill_mappings:
            return [], ""
        
        group_data = self.skill_mappings[business_group]
        ai_rules = group_data['ai_rules']
        header_notes = group_data['header_notes']
        
        if not ai_rules:
            return [], ""
        
        # Format rules for AI prompt
        prompt_rules = []
        for tag, rule in ai_rules.items():
            prompt_rules.append(f"- {tag}:\n  {rule}")
        
        prompt_rules_str = "\n".join(prompt_rules)
        special_notes = header_notes.get('SPECIAL TAGS', header_notes.get('IMPORTANT NOTES', 'No special rules apply.'))
        
        # Truncate content if too long
        if len(website_content) > 10000:
            website_content = website_content[:10000] + "..."
        
        # Try AI classification with retries
        for attempt in range(3):
            try:
                if LANGCHAIN_AVAILABLE and hasattr(self, 'chain'):
                    response_str = self.chain.invoke({
                        "special_rules": special_notes,
                        "dynamic_rules": prompt_rules_str,
                        "context": website_content
                    })
                else:
                    response_str = self.classify_with_openai_direct(
                        website_content, prompt_rules_str, special_notes
                    )
                
                return self.parse_ai_response(response_str, ai_rules)
                
            except Exception as e:
                if attempt == 2:  # Last attempt
                    return [], f"AI Error: {str(e)}"
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return [], "Max retries exceeded"

    def classify_with_openai_direct(self, website_content: str, prompt_rules_str: str, special_notes: str) -> str:
        """Classify using OpenAI client directly"""
        prompt = f"""
You are an expert business classification analyst. Analyze the website text based on the provided rules.

**PRIORITY 1: GLOBAL SPECIAL RULES**
{special_notes}

**PRIORITY 2: INDUSTRY-SPECIFIC RULES**
{prompt_rules_str}

**OUTPUT FORMAT:**
Return a single JSON object with one key: "applied_skills". The value must be a list of strings, with each string being an EXACT match from the rule names provided above.

**WEBSITE TEXT TO ANALYZE:**
{website_content}
"""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        return response.choices[0].message.content

    def parse_ai_response(self, response_str: str, valid_skills: Dict[str, str]) -> Tuple[List[str], str]:
        """Parse AI response and validate skills"""
        try:
            # Clean up response
            if "```json" in response_str:
                response_str = response_str.split("```json")[1].split("```")[0].strip()
            elif "```" in response_str:
                response_str = response_str.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            result = json.loads(response_str)
            applied_skills = result.get("applied_skills", [])
            
            # Validate skills
            validated_skills = []
            for skill in applied_skills:
                if skill in valid_skills:
                    validated_skills.append(skill)
            
            return validated_skills, response_str
            
        except Exception:
            return [], response_str

    def map_skills_to_ids(self, skill_names: List[str], business_group: str) -> List[str]:
        """Map skill names to their corresponding IDs"""
        if business_group not in self.skill_mappings:
            return []
        
        skill_mapping = self.skill_mappings[business_group]['skills']
        matched_ids = []
        
        for skill in skill_names:
            skill_lower = skill.lower().strip()
            for skill_tag, mapping in skill_mapping.items():
                if (skill_tag == skill_lower or 
                    mapping['original_tag'].lower() == skill_lower or
                    mapping['skill_names'].lower() == skill_lower):
                    
                    ids_str = mapping['skill_ids']
                    if ids_str:
                        id_list = [s.strip() for s in str(ids_str).split(',') if s.strip()]
                        matched_ids.extend(id_list)
                    break
        
        return sorted(list(set(id for id in matched_ids if id and id != 'nan')))

    def add_skills_to_business_data(self, businesses_data: List[Dict]) -> List[Dict]:
        """
        MAIN INTEGRATION FUNCTION - Used by main.py
        Add skills to business data from promptParser
        """
        print(f"\n🎯 Adding skills to {len(businesses_data)} businesses...")
        
        enhanced_businesses = []
        successful_matches = 0
        ai_classifications = 0
        
        for i, business in enumerate(businesses_data, 1):
            # Create copy to avoid modifying original
            enhanced_business = business.copy()
            
            # Get business info - handle different field name variations from promptParser
            business_name = business.get('name', business.get('business_name', 'Unknown'))
            business_type = business.get('matched business type', business.get('business_type', ''))
            business_group = business.get('business group', business.get('business_group', ''))
            website_content = business.get('website_content', '')
            
            if i <= 5:  # Show details for first 5 businesses
                print(f"  📊 Processing {i}/{len(businesses_data)}: {business_name}")
            elif i % 10 == 0:  # Show progress every 10 businesses
                print(f"  📊 Processing {i}/{len(businesses_data)}...")
            
            matched_skills = []
            skill_source = "none"
            final_business_group = business_group
            
            # Try AI classification first if available
            if (self.ai_available and 
                website_content and 
                len(website_content.strip()) > 100 and
                business_group and 
                business_group in self.skill_mappings):
                
                ai_skills, ai_response = self.classify_with_ai(website_content, business_group)
                
                if ai_skills:
                    matched_skills = ai_skills
                    skill_source = "ai"
                    ai_classifications += 1
            
            # Fallback to basic matching
            if not matched_skills and business_type:
                skill_ids, skill_names, detected_group = self.find_matching_skills_basic(business_type, business_group)
                
                # Update business group if detected
                if detected_group and not business_group:
                    final_business_group = detected_group
                    enhanced_business['business group'] = detected_group
                
                if skill_names:
                    matched_skills = [s.strip() for s in skill_names.split(',') if s.strip()]
                    skill_source = "basic"
            
            # Map skills to IDs
            if matched_skills and final_business_group:
                skill_ids_list = self.map_skills_to_ids(matched_skills, final_business_group)
                enhanced_business['Skill IDs'] = ', '.join(skill_ids_list)
                enhanced_business['Skill Names'] = ', '.join(matched_skills)
                enhanced_business['Skill Source'] = skill_source
                successful_matches += 1
            else:
                enhanced_business['Skill IDs'] = ""
                enhanced_business['Skill Names'] = ""
                enhanced_business['Skill Source'] = "none"
            
            enhanced_businesses.append(enhanced_business)
        
        print(f"✅ Skill tagging complete! {successful_matches}/{len(businesses_data)} businesses matched")
        if ai_classifications > 0:
            print(f"🧠 AI: {ai_classifications}, 🔍 Basic: {successful_matches - ai_classifications}")
        
        return enhanced_businesses


# ================================
# MAIN FUNCTION FOR main.py INTEGRATION
# ================================

def add_skills_to_businesses(businesses_data: List[Dict], 
                           business_groups_file: str = "Extracted_Business_Groups_and_Types.csv",
                           skill_files_folder: str = "Skill-tag csv",
                           enable_ai: bool = True) -> List[Dict]:
    """
    Main function called by main.py workflow
    
    Args:
        businesses_data: List of business dictionaries from promptParser
        business_groups_file: Path to CSV file with business groups and types
        skill_files_folder: Path to folder containing skill files
        enable_ai: Whether to enable AI-powered skill classification
    
    Returns:
        List of enhanced business dictionaries with Skill IDs and Skill Names added
    """
    try:
        skill_tagger = EnhancedSkillTagger(
            business_groups_file=business_groups_file,
            skill_files_folder=skill_files_folder,
            enable_ai_classification=enable_ai
        )
        
        return skill_tagger.add_skills_to_business_data(businesses_data)
        
    except Exception as e:
        print(f"❌ Error in skill tagger: {e}")
        # Return original data with empty skill fields to avoid breaking pipeline
        for business in businesses_data:
            business['Skill IDs'] = ""
            business['Skill Names'] = ""
            business['Skill Source'] = "error"
        return businesses_data


# ================================
# UTILITY FUNCTIONS
# ================================

def get_skills_for_business_type(business_type: str,
                                business_groups_file: str = "Extracted_Business_Groups_and_Types.csv",
                                skill_files_folder: str = "Skill-tag csv") -> Tuple[str, str, str]:
    """
    Quick function to get skills for a single business type
    
    Returns:
        Tuple of (skill_ids, skill_names, business_group)
    """
    try:
        skill_tagger = EnhancedSkillTagger(
            business_groups_file=business_groups_file,
            skill_files_folder=skill_files_folder,
            enable_ai_classification=False
        )
        
        return skill_tagger.find_matching_skills_basic(business_type)
    except Exception as e:
        print(f"❌ Error in quick skill lookup: {e}")
        return "", "", ""


def test_skill_tagger():
    """Test function for the skill tagger"""
    print("🧪 Testing SkillTagger...")
    
    # Test with sample data that matches promptParser output format
    sample_businesses = [
        {
            'business group': 'Food And Beverage Establishments',
            'matched business type': 'restaurant',
            'name': 'Test Restaurant',
            'website': 'https://test-restaurant.com',
            'address': '123 Main St, Los Angeles, CA',
            'phone number': '(555) 123-4567',
            'website_content': 'We are a fine dining restaurant specializing in Italian cuisine.'
        }
    ]
    
    enhanced_businesses = add_skills_to_businesses(sample_businesses)
    
    print("\n📋 Test Results:")
    for business in enhanced_businesses:
        print(f"🏢 {business['name']}")
        print(f"  Skills: {business.get('Skill Names', 'None')}")
        print(f"  IDs: {business.get('Skill IDs', 'None')}")
        print(f"  Source: {business.get('Skill Source', 'Unknown')}")


if __name__ == "__main__":
    # Run test when file is executed directly
    test_skill_tagger()