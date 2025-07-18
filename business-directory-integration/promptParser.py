import os
import re
import pandas as pd
import requests
from typing import List, Dict
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import time
from openai import OpenAI

OPENAI_API_KEY = "PLACE YOUR OPENAI API KEY"
GOOGLE_API_KEY = "PLACE YOUR GOOGLE API KEY"

class PromptParser:
    def __init__(self, folder_name="Skill tag sheets", output_folder="Output"):
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
        business_type_options = type_map.get(matched_group.lower(), [])
        print("Matched business group is : ", matched_group)

        if not business_type_options:
            print(f"⚠️ No business types found for group: {business_group}")

        # Call Google Places API just once
        places_data = self.search_all_places(business_group, location, limit=100)

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
        output_path = "/tmp/temp_results.xlsx"
        combined_df.to_excel(output_path, index=False)
        self.upload_result_file(output_path, "results.xlsx")

    def upload_result_file(self, new_data_path, title="results.xlsx"):
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
                existing_df = pd.read_excel(existing_path)
            except Exception as e:
                print(f"⚠️ Could not read existing file, using only new data: {e}")
        else:
            print("📄 No existing results file found. A new one will be created.")

        # Load new data (CSV or XLSX depending on path)
        if new_data_path.endswith(".csv"):
            new_df = pd.read_csv(new_data_path)
        else:
            new_df = pd.read_excel(new_data_path)

        # Merge, deduplicate by name+address
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset=["name", "address"], inplace=True)

        # Save merged output as XLSX (always .xlsx for Google Drive view)
        final_path = f"/tmp/{title}"
        combined_df.to_excel(final_path, index=False)

        # Upload or overwrite
        file_metadata = {"title": title, "parents": [{"id": self.output_folder_id}]}
        output_file = self.drive.CreateFile(file_metadata)
        output_file.SetContentFile(final_path)
        output_file.Upload()
        print(f"Uploaded updated '{title}' to Drive ({len(combined_df)} unique records).")