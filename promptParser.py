# Sample Input : animal services in LA

import os
import re
import pandas as pd
import requests
from typing import List, Dict
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import time
from openai import OpenAI

GOOGLE_API_KEY = "Place your google api key here"
OPENAI_API_KEY = "Place your openai api key here"


class PromptParser:
    def __init__(self, folder_name="Skill-tag csv", output_folder="Output"):
        self.folder_name = folder_name
        self.output_folder = output_folder
        self.drive = self.authenticate_drive()
        self.folder_id = self.get_folder_id(self.folder_name)
        self.output_folder_id = self.get_folder_id(self.output_folder)
        self.api_key = GOOGLE_API_KEY  

    def authenticate_drive(self):
        gauth = GoogleAuth()
        gauth.LoadCredentialsFile("mycreds.txt")
        if gauth.credentials is None:
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            gauth.Refresh()
        else:
            gauth.Authorize()
    
        # Save the current credentials for next run
        gauth.SaveCredentialsFile("mycreds.txt")
        return GoogleDrive(gauth)

    def get_folder_id(self, folder_name):
        folder_list = self.drive.ListFile({
            'q': f"mimeType='application/vnd.google-apps.folder' and title='{folder_name}' and trashed=false"
        }).GetList()
        if not folder_list:
            raise Exception(f"No folder named '{folder_name}' found.")
        print("ID for folder : ", folder_name, " is: ", folder_list[0]['id'])
        return folder_list[0]['id']
    
    def load_business_type_map(self) -> Dict[str, List[str]]:
        root_id = self.get_folder_id("Kobeyo Business Directory")
        # Temporarily override self.folder_id to reuse list_business_files()
        original_folder_id = self.folder_id
        self.folder_id = root_id
        files_in_root = self.list_business_files()
        self.folder_id = original_folder_id  # restore to original after use

        # Find the file
        file_obj = next((f for f in files_in_root if f['title'] == "Extracted_Business_Groups_and_Types.csv"),None)
        if not file_obj:
            raise FileNotFoundError("Extracted_Business_Groups_and_Types.csv not found in root.")

        # Download and load it
        file_path = self.download_file(file_obj)
        df = pd.read_csv(file_path)
        mapping = {}

        for _, row in df.iterrows():
            group = str(row["Business Group"]).strip().lower()
            types = [x.strip() for x in str(row["Business Type"]).split(",")]
            mapping[group] = types

        # print("The created mapping : ", mapping)
        return mapping

    def match_business_group(self, user_input: str, available_groups: List[str]) -> str:
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = (
            f"A user entered the business group: '{user_input}'.\n"
            f"Here is a list of valid business groups:\n"
            + "\n".join(f"- {g}" for g in available_groups)
            + "\n\nWhich of these is the best match for the user input? Respond with only one group name from the list above."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI group matching failed for '{user_input}': {e}")
            return user_input


    def list_business_files(self):
        return self.drive.ListFile({
            'q': f"'{self.folder_id}' in parents and trashed=false"
        }).GetList()

    # def infer_business_type(self, filename: str) -> str:
    #     name = os.path.splitext(filename)[0]
    #     return name.replace("_", " ").replace("-", " ").lower().strip()

    def download_file(self, file_obj):
        file_path = f"/tmp/{file_obj['title']}"
        file_obj.GetContentFile(file_path)
        return file_path

    def load_skill_tag_sheet(self, business_group: str) -> str:
        """
        Finds and downloads the correct skill tag CSV from 'Skill-tag csv ' folder
        based on the given business group name. Returns the local path of the file.
        """
        print(f"Looking for skill tag sheet for business group: {business_group}")

        # Convert to expected filename format
        group_slug = business_group.strip().title().replace(" ", "_")
        expected_filename = f"{group_slug}.xlsx"
        print(f"Constructed expected file name: {expected_filename}")

        # Get folder ID for "Skill-tag csv"
        skill_tag_folder_id = self.get_folder_id("Skill-tag csv")
        print(f"Folder ID for 'Skill-tag csv': {skill_tag_folder_id}")

        # List files in that folder
        skill_files = self.drive.ListFile({
            'q': f"'{skill_tag_folder_id}' in parents and trashed=false"
        }).GetList()

        # Try to find the matching file
        matching_file = next((f for f in skill_files if f['title'] == expected_filename), None)

        if not matching_file:
            raise FileNotFoundError(f"Skill tag sheet not found for group: {business_group}")

        print(f"Found skill tag file: {matching_file['title']} (ID: {matching_file['id']})")

        # Download it to /tmp and return path
        file_path = f"/tmp/{expected_filename}"
        matching_file.GetContentFile(file_path)
        print(f"Downloaded skill tag file to: {file_path}")
        return file_path

    def parse_location_and_type(self, prompt: str) -> Dict[str, str]:
        # Replace with your custom logic or OpenAI if needed
        match = re.search(r"(.*) in (.*)", prompt, re.IGNORECASE)
        return {"business_type": match.group(1).strip(), "location": match.group(2).strip()} if match else {}

    def search_all_places(self, query: str, location: str, limit: int = 20) -> List[Dict]:
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        all_results = []
        params = {"query": f"{query} in {location}", "key": self.api_key}

        while True:
            response = requests.get(url, params=params).json()
            all_results.extend(response.get("results", []))
            if len(all_results) >= limit:
                return all_results[:limit]
            next_page_token = response.get("next_page_token")
            if not next_page_token:
                break
            time.sleep(2)
            params = {"pagetoken": next_page_token, "key": self.api_key}

        return all_results[:limit]

    def get_place_details(self, place_id: str) -> Dict:
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {
            "place_id": place_id,
            "fields": "name,website,formatted_phone_number",
            "key": self.api_key
        }
        response = requests.get(url, params=params).json()
        return response.get("result", {})

    def match_business_type(self, place_name: str, type_options: List[str]) -> str:
        client = OpenAI(api_key=OPENAI_API_KEY)

        prompt = (
            f"From the list below, which business type best matches this place?\n\n"
            f"Place: {place_name}\n\n"
            f"Options:\n- " + "\n- ".join(type_options) +
            f"\n\nRespond with only the best matching business type from the list above."
        )

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI matching failed for '{place_name}': {e}")
            return ""

    def process_files(self, user_prompt: str):
        parsed_info = self.parse_location_and_type(user_prompt)
        if not parsed_info:
            raise ValueError("Could not extract business type and location from prompt.")
        # Treat parsed business_type as the business group
        business_group = parsed_info["business_type"]
        location = parsed_info["location"]

        # Load mapping from business group to business types
        type_map = self.load_business_type_map()
        available_groups = list(type_map.keys())
        matched_group = self.match_business_group(business_group, available_groups)

         # 🔁 Load the skill tag sheet for the matched group
        try:
            skill_tag_path = self.load_skill_tag_sheet(matched_group)
            print(f"Skill tag sheet loaded: {skill_tag_path}")
            # Optional: load as DataFrame
            # skill_df = pd.read_csv(skill_tag_path)
        except Exception as e:
            print(f"Could not load skill tag sheet for '{matched_group}': {e}")
            skill_tag_path = None

        business_type_options = type_map.get(matched_group.lower(), [])
        print("Matched business group is : ", matched_group)

        if not business_type_options:
            print(f"No business types found for group: {business_group}")

        # Call Google Places API just once
        places_data = self.search_all_places(business_group, location, limit=10)

        rows = []
        for place in places_data:
            name = place.get("name", "")
            address = place.get("formatted_address", "")
            place_id = place.get("place_id", "")
            details = self.get_place_details(place_id) if place_id else {}
            website = details.get("website", "")
            phone = details.get("formatted_phone_number", "")

            matched_type = self.match_business_type(name, business_type_options) if business_type_options else ""
            print("Predicted business type is : ", matched_type)

            rows.append({
                "business group": matched_group,
                "matched business type": matched_type,
                "name": name,
                "website": website,
                "address": address,
                "phone number": phone
            })

        combined_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        # output_path = "/tmp/temp_results.xlsx"
        # combined_df.to_excel(output_path, index=False)
        combined_df = self.enrich_with_skills(combined_df, skill_tag_path)
        # Pass raw new DataFrame directly
        self.upload_result_file(new_data=combined_df, title="results.csv")

    def enrich_with_skills(self, df: pd.DataFrame, skill_tag_path: str) -> pd.DataFrame:
        """
        Adds Skill IDs and Skill Names to the result DataFrame based on matched business types.
        """
        if not skill_tag_path or not os.path.exists(skill_tag_path):
            print("Skill tag sheet not found or missing. Skipping enrichment.")
            return df

        try:
            skill_df = pd.read_excel(skill_tag_path)
        except Exception as e:
            print(f"Could not read skill tag sheet: {e}")
            return df

        # Normalize column names
        skill_df.columns = [col.strip() for col in skill_df.columns]
        skill_df["Skills Tags"] = skill_df["Skills Tags"].astype(str).str.strip().str.lower()
        
        # Create lookup dict
        skill_lookup = {
            row["Skills Tags"]: (
                str(row.get("Skills IDs", "")).strip(),
                str(row.get("Skills Names", "")).strip()
            )
            for _, row in skill_df.iterrows()
        }

        # Normalize and map
        def get_skill_info(matched_type):
            key = str(matched_type).strip().lstrip("-").lower()  # remove leading dash too
            return skill_lookup.get(key, ("", ""))

        df["Skill IDs"] = df["matched business type"].map(lambda x: get_skill_info(x)[0])
        df["Skill Names"] = df["matched business type"].map(lambda x: get_skill_info(x)[1])

        print("Enriched results with Skill IDs and Skill Names.")
        return df


    def upload_result_file(self, new_data: pd.DataFrame, title="results.xlsx"):
        """
        Uploads or merges the given new_data DataFrame with the existing file in Drive.
        Deduplicates based on business name + address.
        Automatically handles CSV or XLSX based on the file extension in 'title'.
        """
        # Check for existing file in the output folder
        files = self.drive.ListFile({
        'q': f"'{self.output_folder_id}' in parents and trashed=false and title='{title}'"
        }).GetList()

        existing_df = pd.DataFrame()
        if files:
            existing_file = files[0]
            existing_path = f"/tmp/{title}"
            existing_file.GetContentFile(existing_path)
            try:
                if title.lower().endswith(".xlsx"):
                    existing_df = pd.read_excel(existing_path)
                else:
                    existing_df = pd.read_csv(existing_path)
                print(f"Loaded existing '{title}' from Drive with {len(existing_df)} rows")
            except Exception as e:
                print(f"Could not read existing file. Proceeding with new data only: {e}")
        else:
            print("No existing results file found. A new one will be created.")

        combined_df = pd.concat([existing_df, new_data], ignore_index=True)

        # Normalize for better deduplication
        combined_df["name"] = combined_df["name"].str.strip().str.lower()
        combined_df["address"] = combined_df["address"].str.strip().str.lower()

        # Deduplicate
        combined_df.drop_duplicates(subset=["name", "address"], inplace=True)

        # Count
        before = len(existing_df)
        after = len(combined_df)
        added = after - before

        print(f"Updated results: {after} unique businesses ({added} new added)")

        # Save to local disk (CSV or XLSX)
        final_output_path = f"/tmp/{title}"
        if title.lower().endswith(".xlsx"):
            combined_df.to_excel(final_output_path, index=False)
        else:
            combined_df.to_csv(final_output_path, index=False)
        # Delete existing file(s) with the same name
        for f in files:
            try:
                f.Delete()
                print(f"Deleted old file: {f['title']} (ID: {f['id']})")
            except Exception as e:
                print(f"Failed to delete old file '{f['title']}': {e}")

        # Upload to Google Drive
        file_to_upload = self.drive.CreateFile({
            "title": title,
            "parents": [{"id": self.output_folder_id}]
        })
        file_to_upload.SetContentFile(final_output_path)
        file_to_upload.Upload()
        print(f"File '{title}' uploaded to Google Drive folder '{self.output_folder}'.")
