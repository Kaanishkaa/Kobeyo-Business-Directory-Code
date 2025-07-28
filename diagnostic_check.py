"""
diagnostic_check.py - Find out why skills aren't being tagged
Run this to identify the exact issues
"""

import os
import pandas as pd
from typing import List, Dict

def check_file_structure():
    """Check if required files exist"""
    print("=" * 60)
    print("CHECKING FILE STRUCTURE")
    print("=" * 60)
    
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check business groups file
    business_groups_file = "Extracted_Business_Groups_and_Types.csv"
    if os.path.exists(business_groups_file):
        print(f"✅ Found: {business_groups_file}")
        try:
            df = pd.read_csv(business_groups_file)
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Sample rows: {len(df)} total")
            if len(df) > 0:
                print(f"   First group: {df.iloc[0].to_dict()}")
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    else:
        print(f"❌ Missing: {business_groups_file}")
    
    # Check skill files folder
    skill_folder = "Skill-tag csv"
    if os.path.exists(skill_folder):
        print(f"✅ Found: {skill_folder}/")
        files = os.listdir(skill_folder)
        print(f"   Files in folder: {len(files)}")
        print(f"   Sample files: {files[:5]}")
    else:
        print(f"❌ Missing: {skill_folder}/")
    
    # List all files in current directory
    print(f"\n📁 All files in current directory:")
    all_files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for file in sorted(all_files)[:10]:  # Show first 10
        print(f"   {file}")
    
    print(f"\n📂 All directories:")
    all_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    for dir_name in sorted(all_dirs):
        print(f"   {dir_name}/")


def test_prompt_parser():
    """Test if promptParser works and what it returns"""
    print("\n" + "=" * 60)
    print("TESTING PROMPT PARSER")
    print("=" * 60)
    
    try:
        from promptParser import PromptParser
        print("✅ promptParser imported successfully")
        
        # Test with small query
        parser = PromptParser()
        print("✅ PromptParser initialized")
        
        # Try to get just 2 businesses for testing
        print("🔍 Testing with 'restaurants in Los Angeles' (limited to 2 results)")
        businesses = parser.process_files("restaurants in Los Angeles")
        
        if businesses:
            print(f"✅ Got {len(businesses)} businesses from promptParser")
            
            # Show structure of first business
            if len(businesses) > 0:
                sample = businesses[0]
                print(f"\n📋 Sample business structure:")
                for key, value in sample.items():
                    display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    print(f"   {key}: {display_value}")
                
                # Check for key fields that skillTagger needs
                required_fields = ['name', 'matched business type', 'business group']
                missing_fields = [field for field in required_fields if field not in sample]
                if missing_fields:
                    print(f"\n❌ Missing required fields for skillTagger: {missing_fields}")
                    print(f"💡 Available fields: {list(sample.keys())}")
                else:
                    print(f"\n✅ All required fields present for skillTagger")
        else:
            print("❌ promptParser returned no businesses")
            
    except ImportError as e:
        print(f"❌ Cannot import promptParser: {e}")
    except Exception as e:
        print(f"❌ promptParser error: {e}")
        import traceback
        traceback.print_exc()


def test_skill_tagger():
    """Test skillTagger with sample data"""
    print("\n" + "=" * 60)
    print("TESTING SKILL TAGGER")
    print("=" * 60)
    
    try:
        from skillTagger import add_skills_to_businesses
        print("✅ skillTagger imported successfully")
        
        # Test with sample data that should work
        sample_businesses = [
            {
                'name': 'Test Restaurant',
                'matched business type': 'restaurant',
                'business group': 'Food And Beverage Establishments',
                'website': 'https://test-restaurant.com',
                'address': '123 Main St, Los Angeles, CA',
                'phone number': '(555) 123-4567',
                'website_content': 'We serve delicious Italian food and pizza.'
            }
        ]
        
        print("🔍 Testing skillTagger with sample restaurant data...")
        enhanced = add_skills_to_businesses(sample_businesses)
        
        if enhanced:
            print(f"✅ skillTagger returned {len(enhanced)} businesses")
            
            sample_result = enhanced[0]
            print(f"\n📋 Sample enhanced business:")
            
            # Check for skill fields
            skill_fields = ['Skill IDs', 'Skill Names', 'Skill Source']
            for field in skill_fields:
                value = sample_result.get(field, 'NOT_FOUND')
                print(f"   {field}: {value}")
            
            # Overall assessment
            if sample_result.get('Skill Names'):
                print(f"\n✅ Skills successfully added!")
            else:
                print(f"\n❌ No skills were added")
                print(f"💡 This indicates an issue with skill file loading or matching")
        else:
            print("❌ skillTagger returned empty results")
            
    except ImportError as e:
        print(f"❌ Cannot import skillTagger: {e}")
    except Exception as e:
        print(f"❌ skillTagger error: {e}")
        import traceback
        traceback.print_exc()


def test_full_pipeline():
    """Test the complete pipeline with minimal data"""
    print("\n" + "=" * 60)
    print("TESTING FULL PIPELINE")
    print("=" * 60)
    
    try:
        # Step 1: Get businesses from promptParser
        from promptParser import PromptParser
        parser = PromptParser()
        businesses = parser.process_files("restaurants in Los Angeles")
        
        if not businesses:
            print("❌ No businesses from promptParser - pipeline stops here")
            return
        
        print(f"✅ Step 1: Got {len(businesses)} businesses from promptParser")
        
        # Step 2: Add skills
        from skillTagger import add_skills_to_businesses
        enhanced_businesses = add_skills_to_businesses(businesses[:2])  # Test with just 2
        
        print(f"✅ Step 2: Enhanced {len(enhanced_businesses)} businesses with skills")
        
        # Check if skills were actually added
        businesses_with_skills = sum(1 for b in enhanced_businesses 
                                   if b.get('Skill Names') and b.get('Skill Names').strip())
        print(f"📊 Businesses with actual skills: {businesses_with_skills}/{len(enhanced_businesses)}")
        
        # Step 3: Test email classifier (optional)
        try:
            from emailClassifier import add_email_classification
            email_enhanced = add_email_classification(enhanced_businesses[:1])  # Test with 1
            print(f"✅ Step 3: Added email data to {len(email_enhanced)} businesses")
        except Exception as e:
            print(f"⚠️ Step 3 (email): {e}")
        
        # Show final result
        if enhanced_businesses:
            final_business = enhanced_businesses[0]
            print(f"\n📋 Final business structure:")
            for key, value in final_business.items():
                if 'skill' in key.lower() or 'email' in key.lower():
                    print(f"   {key}: {value}")
        
        print(f"\n🎯 DIAGNOSIS COMPLETE")
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()


def check_csv_output():
    """Check the existing CSV output"""
    print("\n" + "=" * 60)
    print("CHECKING CSV OUTPUT")
    print("=" * 60)
    
    csv_file = "restaurants_in_Los_Angeles_results.csv"
    if os.path.exists(csv_file):
        print(f"✅ Found output file: {csv_file}")
        
        try:
            df = pd.read_csv(csv_file)
            print(f"📊 CSV shape: {df.shape}")
            
            if len(df) > 0:
                print(f"📋 CSV columns: {list(df.columns)}")
                
                # Check for skill columns
                skill_columns = [col for col in df.columns if 'skill' in col.lower()]
                if skill_columns:
                    print(f"✅ Skill columns found: {skill_columns}")
                    
                    # Check if any businesses have skills
                    if 'Skill Names' in df.columns:
                        non_empty_skills = df[df['Skill Names'].notna() & (df['Skill Names'] != '')]
                        print(f"📊 Businesses with skills in CSV: {len(non_empty_skills)}/{len(df)}")
                else:
                    print(f"❌ No skill columns found in CSV")
                
                # Show sample row
                print(f"\n📋 Sample row from CSV:")
                sample_row = df.iloc[0].to_dict()
                for key, value in sample_row.items():
                    display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    print(f"   {key}: {display_value}")
                    
            else:
                print(f"❌ CSV is empty (0 rows)")
                
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
    else:
        print(f"❌ Output file not found: {csv_file}")


def main():
    """Run all diagnostic checks"""
    print("🔍 COMPREHENSIVE DIAGNOSTIC CHECK")
    print("Finding out why skills aren't being tagged...")
    
    check_file_structure()
    test_prompt_parser()
    test_skill_tagger()
    check_csv_output()
    test_full_pipeline()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print("1. Check the output above for any ❌ errors")
    print("2. Missing files need to be added to your directory")
    print("3. Field name mismatches need to be fixed")
    print("4. If skillTagger test works but full pipeline doesn't,")
    print("   the issue is in data flow between modules")
    print("\nRun this script to identify the exact problem!")


if __name__ == "__main__":
    main()