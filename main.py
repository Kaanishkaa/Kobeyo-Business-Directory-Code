"""
main.py - Clean Business Analysis Workflow (CLEAN VERSION)
Enhanced with Google Drive upload and skill debugging
Simple orchestrator that follows the correct flow:
promptParser → skillTagger → emailClassifier → careersPage → Google Drive
"""

import os
import sys
import pandas as pd
import time
from typing import List, Dict, Optional

def main(user_prompt: str = None):
    """
    Main workflow function - handles the complete pipeline
    """
    print("Starting Business Analysis Workflow (ENHANCED)")
    print("=" * 50)
    
    # Get user input if not provided
    if not user_prompt:
        user_prompt = input("Enter your search (e.g., 'restaurants in Los Angeles'): ").strip()
    
    if not user_prompt:
        print("No search query provided. Exiting.")
        return None
    
    print(f"Processing: '{user_prompt}'")
    print("=" * 50)
    
    # ========================================
    # STEP 1: GET BUSINESSES FROM PROMPT PARSER
    # ========================================
    print("\nSTEP 1: Scraping businesses...")
    
    try:
        from promptParser import PromptParser
        
        # Initialize prompt parser
        parser = PromptParser()
        
        # Process the user prompt and get businesses
        businesses = parser.process_files(user_prompt)
        
        if not businesses:
            print("No businesses found from prompt parser")
            return None
        
        print(f"Found {len(businesses)} businesses")
        
        # DEBUG: Check promptParser output structure
        if businesses:
            print(f"DEBUG: Sample business from promptParser:")
            sample_business = businesses[0]
            print(f"   Keys: {list(sample_business.keys())}")
            
            # Check for key fields
            for field in ['name', 'business_name', 'matched business type', 'business_type', 'business group', 'business_group']:
                if field in sample_business:
                    value = str(sample_business[field])[:100]
                    print(f"   {field}: {value}")
        
        # Convert to DataFrame for easier handling
        businesses_df = pd.DataFrame(businesses)
        print(f"Business data shape: {businesses_df.shape}")
        
    except ImportError:
        print("Could not import promptParser module")
        return None
    except Exception as e:
        print(f"Error in prompt parser: {e}")
        return None
    
    # ========================================
    # STEP 2: ADD SKILLS WITH SKILL TAGGER
    # ========================================
    print("\nSTEP 2: Adding skills...")
    
    try:
        from skillTagger import add_skills_to_businesses
        
        # Convert DataFrame back to list of dicts for skill tagger
        businesses_list = businesses_df.to_dict('records')
        
        print(f"DEBUG: Sending {len(businesses_list)} businesses to skillTagger")
        
        # Add skills to businesses
        enhanced_businesses = add_skills_to_businesses(
            businesses_list,
            business_groups_file="Extracted_Business_Groups_and_Types.csv",
            skill_files_folder="Skill-tag csv",
            enable_ai=True
        )
        
        if not enhanced_businesses:
            print("No businesses returned from skill tagger")
            enhanced_businesses = businesses_list
        
        print(f"Skills added to {len(enhanced_businesses)} businesses")
        
        # DEBUG: Check if skills were actually added
        if enhanced_businesses:
            print(f"DEBUG: Checking skillTagger output...")
            sample_enhanced = enhanced_businesses[0]
            
            # Check for skill fields
            skill_fields = ['Skill IDs', 'Skill Names', 'Skill Source']
            for field in skill_fields:
                if field in sample_enhanced:
                    value = sample_enhanced[field]
                    print(f"   {field}: {value}")
                else:
                    print(f"   {field}: NOT FOUND")
            
            # Count businesses with skills
            businesses_with_skills = sum(1 for b in enhanced_businesses 
                                       if b.get('Skill Names') and b.get('Skill Names').strip())
            print(f"   Businesses with skills: {businesses_with_skills}/{len(enhanced_businesses)}")
        
        # Convert back to DataFrame
        businesses_df = pd.DataFrame(enhanced_businesses)
        
        # DEBUG: Verify DataFrame has skill columns
        skill_columns = [col for col in businesses_df.columns if 'skill' in col.lower()]
        print(f"DEBUG: Skill columns in DataFrame: {skill_columns}")
        
    except ImportError:
        print("Could not import skillTagger module - continuing without skills")
        enhanced_businesses = businesses_df.to_dict('records')
    except Exception as e:
        print(f"Error in skill tagger: {e}")
        import traceback
        traceback.print_exc()
        enhanced_businesses = businesses_df.to_dict('records')
    
    # ========================================
    # STEP 3: EMAIL CLASSIFICATION
    # ========================================
    print("\nSTEP 3: Email classification...")
    
    try:
        from emailClassifier import add_email_classification
        
        # Convert to list for email classifier
        businesses_list = businesses_df.to_dict('records')
        
        # Add email classification
        email_enhanced_businesses = add_email_classification(businesses_list)
        
        print(f"Email classification added to {len(email_enhanced_businesses)} businesses")
        
        # Convert back to DataFrame
        businesses_df = pd.DataFrame(email_enhanced_businesses)
        
    except ImportError:
        print("Email classifier module not available - skipping")
    except Exception as e:
        print(f"Error in email classification: {e}")
    
    # ========================================
    # STEP 4: CAREERS PAGE EXTRACTION
    # ========================================
    print("\nSTEP 4: Careers page extraction...")
    
    try:
        from careersPage import add_careers_data
        
        # Convert to list for careers page extractor
        businesses_list = businesses_df.to_dict('records')
        
        # Add careers page data
        final_businesses = add_careers_data(businesses_list)
        
        print(f"Careers data added to {len(final_businesses)} businesses")
        
        # Convert to final DataFrame
        final_df = pd.DataFrame(final_businesses)
        
    except ImportError:
        print("Careers page module not available - skipping")
        final_df = businesses_df
    except Exception as e:
        print(f"Error in careers page extraction: {e}")
        final_df = businesses_df
    
    # ========================================
    # STEP 5: SAVE FINAL RESULTS (ENHANCED)
    # ========================================
    print("\nSTEP 5: Saving results...")
    
    try:
        # DEBUG: Final DataFrame analysis
        print(f"DEBUG: Final DataFrame analysis...")
        print(f"   Shape: {final_df.shape}")
        print(f"   Columns: {list(final_df.columns)}")
        
        # Check for skill columns specifically
        skill_columns = [col for col in final_df.columns if 'skill' in col.lower()]
        print(f"   Skill columns: {skill_columns}")
        
        # Check skill data
        if 'Skill Names' in final_df.columns:
            non_empty_skills = final_df[
                final_df['Skill Names'].notna() & 
                (final_df['Skill Names'] != '') & 
                (final_df['Skill Names'] != 'nan')
            ]
            print(f"   Businesses with skills in final DF: {len(non_empty_skills)}/{len(final_df)}")
            
            # Show sample of businesses with skills
            if len(non_empty_skills) > 0:
                print(f"   Sample businesses with skills:")
                for i, (_, row) in enumerate(non_empty_skills.head(3).iterrows()):
                    business_name = row.get('name', row.get('business_name', 'Unknown'))
                    skills = row.get('Skill Names', 'None')
                    skill_ids = row.get('Skill IDs', 'None')
                    print(f"     - {business_name}")
                    print(f"       Skills: {skills}")
                    print(f"       IDs: {skill_ids}")
        else:
            print(f"   'Skill Names' column not found in final DataFrame!")
        
        # Create output directory
        output_dir = "Output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate filename based on search query
        safe_filename = "".join(c for c in user_prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_filename = safe_filename.replace(' ', '_')[:50]  # Limit length
        
        output_file = os.path.join(output_dir, f"{safe_filename}_results.csv")
        
        # Save to local CSV first
        final_df.to_csv(output_file, index=False)
        print(f"Results saved locally to: {output_file}")
        
        # Try to upload to Google Drive
        drive_result = upload_to_google_drive(output_file, f"{safe_filename}_results.csv")
        
        print(f"Total businesses processed: {len(final_df)}")
        
        # Print summary statistics
        print_summary_stats(final_df)
        
        return output_file
        
    except Exception as e:
        print(f"Error saving results: {e}")
        import traceback
        traceback.print_exc()
        return None


def upload_to_google_drive(local_file_path, drive_filename):
    """
    Upload CSV to Google Drive - tries multiple methods
    """
    print(f"\nAttempting to upload to Google Drive...")
    
    # Method 1: Google Colab Drive mount (most common)
    try:
        from google.colab import drive
        
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
        
        # Copy file to mounted Drive
        import shutil
        drive_path = f'/content/drive/MyDrive/{drive_filename}'
        shutil.copy2(local_file_path, drive_path)
        
        print(f"Successfully uploaded to Google Drive: {drive_path}")
        print(f"You can find it in your Google Drive root folder as: {drive_filename}")
        
        return drive_path
        
    except ImportError:
        print("Not running in Google Colab - trying alternative methods...")
        
        # Method 2: PyDrive (if available)
        try:
            return upload_with_pydrive(local_file_path, drive_filename)
        except Exception as e:
            print(f"PyDrive upload failed: {e}")
            
            # Method 3: Manual instructions
            print(f"MANUAL UPLOAD REQUIRED:")
            print(f"   1. Download the file from: {local_file_path}")
            print(f"   2. Upload it manually to your Google Drive")
            print(f"   3. Rename it to: {drive_filename}")
            
            return None
    
    except Exception as e:
        print(f"Google Drive upload failed: {e}")
        return None


def upload_with_pydrive(local_file_path, drive_filename):
    """
    Upload using PyDrive (alternative method)
    """
    try:
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        from google.colab import auth
        from oauth2client.client import GoogleCredentials
        
        print("Authenticating with PyDrive...")
        
        # Authenticate
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        drive = GoogleDrive(gauth)
        
        # Upload file
        file_drive = drive.CreateFile({'title': drive_filename})
        file_drive.SetContentFile(local_file_path)
        file_drive.Upload()
        
        print(f"Successfully uploaded via PyDrive: {drive_filename}")
        print(f"File ID: {file_drive['id']}")
        
        return file_drive['id']
        
    except ImportError:
        raise Exception("PyDrive not available")
    except Exception as e:
        raise Exception(f"PyDrive upload failed: {e}")


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics of the final results"""
    print("\nSUMMARY STATISTICS:")
    print("=" * 30)
    
    # Basic stats
    print(f"Total businesses: {len(df)}")
    
    # Business groups
    if 'business group' in df.columns:
        groups = df['business group'].value_counts()
        print(f"Business groups: {len(groups)}")
        for group, count in groups.head(3).items():
            print(f"  - {group}: {count}")
    
    # Skills stats (ENHANCED)
    if 'Skill Names' in df.columns:
        businesses_with_skills = len(df[df['Skill Names'].notna() & 
                                      (df['Skill Names'] != '') & 
                                      (df['Skill Names'] != 'nan')])
        print(f"Businesses with skills: {businesses_with_skills}/{len(df)}")
        
        # Show skill sources
        if 'Skill Source' in df.columns:
            skill_sources = df['Skill Source'].value_counts()
            print(f"  Skill sources: {dict(skill_sources)}")
    else:
        print("No 'Skill Names' column found - skills not added!")
    
    # Websites
    if 'website' in df.columns:
        businesses_with_websites = len(df[df['website'].notna() & (df['website'] != '')])
        print(f"Businesses with websites: {businesses_with_websites}/{len(df)}")
    
    # Email classification (if available)
    if 'email_classification' in df.columns:
        businesses_with_emails = len(df[df['email_classification'].notna()])
        print(f"Businesses with email data: {businesses_with_emails}/{len(df)}")
    
    # Careers data (if available)
    if 'careers_data' in df.columns:
        businesses_with_careers = len(df[df['careers_data'].notna()])
        print(f"Businesses with careers data: {businesses_with_careers}/{len(df)}")


def debug_skill_pipeline():
    """
    Special debug function to test just the skill pipeline
    """
    print("DEBUGGING SKILL PIPELINE...")
    
    # Test sample data
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
    
    print(f"Input: {len(sample_businesses)} sample businesses")
    
    try:
        from skillTagger import add_skills_to_businesses
        
        result = add_skills_to_businesses(sample_businesses)
        
        print(f"Output: {len(result)} businesses returned")
        
        if result:
            business = result[0]
            print(f"\nSample result:")
            print(f"  Name: {business.get('name', 'Unknown')}")
            print(f"  Type: {business.get('matched business type', 'Unknown')}")
            print(f"  Group: {business.get('business group', 'Unknown')}")
            print(f"  Skill IDs: {business.get('Skill IDs', 'None')}")
            print(f"  Skill Names: {business.get('Skill Names', 'None')}")
            print(f"  Skill Source: {business.get('Skill Source', 'None')}")
            
            # Save debug result
            df = pd.DataFrame(result)
            df.to_csv("debug_skill_test.csv", index=False)
            print(f"\nDebug result saved to: debug_skill_test.csv")
        
    except Exception as e:
        print(f"Skill pipeline test failed: {e}")
        import traceback
        traceback.print_exc()


def test_workflow():
    """Test the workflow with sample data"""
    print("Testing workflow components...")
    
    # Test each module individually
    modules_to_test = [
        ('promptParser', 'PromptParser'),
        ('skillTagger', 'add_skills_to_businesses'),
        ('emailClassifier', 'add_email_classification'),
        ('careersPage', 'add_careers_data')
    ]
    
    for module_name, function_name in modules_to_test:
        try:
            module = __import__(module_name)
            if hasattr(module, function_name):
                print(f"✓ {module_name}.{function_name} - Available")
            else:
                print(f"? {module_name}.{function_name} - Function not found")
        except ImportError:
            print(f"✗ {module_name} - Module not found")
        except Exception as e:
            print(f"✗ {module_name} - Error: {e}")


def run_sample_test():
    """Run a quick test with sample data"""
    print("Running sample test...")
    
    sample_businesses = [
        {
            'business group': 'Food And Beverage Establishments',
            'matched business type': 'restaurant',
            'name': 'Sample Restaurant',
            'website': 'https://sample-restaurant.com',
            'address': '123 Main St, Los Angeles, CA',
            'phone number': '(555) 123-4567'
        }
    ]
    
    try:
        from skillTagger import add_skills_to_businesses
        result = add_skills_to_businesses(sample_businesses)
        print("Sample test passed")
        print(f"Sample result: {result[0].get('Skill Names', 'No skills found')}")
    except Exception as e:
        print(f"Sample test failed: {e}")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_workflow()
        elif sys.argv[1] == "sample":
            run_sample_test()
        elif sys.argv[1] == "debug":
            debug_skill_pipeline()
        else:
            # Use command line argument as search query
            main(" ".join(sys.argv[1:]))
    else:
        # Interactive mode
        main()