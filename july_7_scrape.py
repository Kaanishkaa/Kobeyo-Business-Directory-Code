import os
import json
import openai
import pandas as pd
import requests
from typing import List, Dict
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io

# === CONFIG ===
GOOGLE_API_KEY = ""
OPENAI_API_KEY = ""
SERVICE_ACCOUNT_FILE = "turing-course-465323-e9-269dd65ad13f.json"
DRIVE_FOLDER_ID = "1kL8VN9cCSYO1aw_yKySIwP7zhKRqy9ZM"  # Where to save/load files
EXTRACTED_MAPPING_CSV = "Extracted_Business_Groups_and_Types.csv"


def list_drive_files(folder_id):
    query = f"'{folder_id}' in parents"
    results = (
        drive_service.files()
        .list(q=query, pageSize=100, fields="files(id, name)")
        .execute()
    )
    items = results.get("files", [])
    print("Files in Drive folder:")
    for item in items:
        print(f"{item['name']} (ID: {item['id']})")


# === SETUP ===
openai.api_key = OPENAI_API_KEY
SCOPES = ["https://www.googleapis.com/auth/drive"]
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
drive_service = build("drive", "v3", credentials=credentials)


# === HELPERS ===
def download_file_from_drive(file_name):
    query = f"name='{file_name}' and '{DRIVE_FOLDER_ID}' in parents"
    response = drive_service.files().list(q=query, spaces="drive").execute()
    files = response.get("files", [])
    if not files:
        raise FileNotFoundError(
            f"File '{file_name}' not found in folder ID {DRIVE_FOLDER_ID}"
        )
    file = files[0]
    request = drive_service.files().get_media(fileId=file["id"])
    fh = io.FileIO(file_name, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return file_name


def upload_to_drive(local_path, drive_folder_id):
    file_metadata = {"name": os.path.basename(local_path), "parents": [drive_folder_id]}
    media = MediaFileUpload(local_path, resumable=True)
    drive_service.files().create(
        body=file_metadata, media_body=media, fields="id"
    ).execute()


def get_google_places(query: str) -> List[Dict]:
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={query}&key={GOOGLE_API_KEY}"
    results = []
    while url:
        response = requests.get(url).json()
        results.extend(response.get("results", []))
        url = response.get("next_page_token", None)
        if url:
            import time

            time.sleep(2)
            url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?pagetoken={url}&key={GOOGLE_API_KEY}"
    return results


def map_to_business_group(prompt: str, business_df: pd.DataFrame) -> str:
    for _, row in business_df.iterrows():
        if any(
            bt.strip().lower() in prompt.lower()
            for bt in row["Business Type"].split(",")
        ):
            return row["Business Group"]
    return "Unknown"


def get_skill_tagging(
    prompt: str, description: str, skill_rules_df: pd.DataFrame
) -> List[str]:
    rule_texts = skill_rules_df["Prompt Rule"].dropna().tolist()
    prompt_input = (
        f"Prompt: {prompt}\nBusiness Description: {description}\nSkill Rules:\n"
        + "\n".join(rule_texts)
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "Tag businesses with relevant skills based on rules.",
            },
            {"role": "user", "content": prompt_input},
        ],
    )
    return response.choices[0].message.content.strip().split(",")


def deduplicate_and_append(new_df: pd.DataFrame, file_path: str):
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        combined_df = pd.concat([old_df, new_df]).drop_duplicates(
            subset="place_id", keep="first"
        )
    else:
        combined_df = new_df
    combined_df.to_csv(file_path, index=False)
    return file_path


# === MAIN ===
def run_tagging_workflow(prompt: str):
    # Step 1: Load business group/type mapping
    list_drive_files()
    download_file_from_drive(EXTRACTED_MAPPING_CSV)
    business_df = pd.read_csv(EXTRACTED_MAPPING_CSV)

    # Step 2: Get Places
    places = get_google_places(prompt)

    if not places:
        print("No places found.")
        return

    # Step 3: Map to Business Group
    group = map_to_business_group(prompt, business_df)
    skill_file_name = group.lower().replace("&", "and").replace(" ", "_") + ".xlsx"

    # Step 4: Load Skill Tagging Rules
    download_file_from_drive(skill_file_name)
    skill_rules_df = pd.read_excel(skill_file_name)

    # Step 5: Tag Places

    # Step 5: Tag Places (Limit to 10 to save OpenAI tokens)
    tagged_rows = []
    for p in places[:10]:
        place_id = p.get("place_id")
        name = p.get("name")
        desc = p.get("types", [])
        address = p.get("formatted_address", "")
        try:
            tags = get_skill_tagging(
                prompt, f"{name}, {desc}, {address}", skill_rules_df
            )
            tagged_rows.append(
                {
                    "place_id": place_id,
                    "name": name,
                    "address": address,
                    "tags": ",".join(tags),
                }
            )
        except Exception as e:
            print(f"Error tagging {name}: {e}")

    # Step 6: Save and Upload
    result_df = pd.DataFrame(tagged_rows)
    local_result_file = (
        group.lower().replace("&", "and").replace(" ", "_") + "_results.csv"
    )
    final_path = deduplicate_and_append(result_df, local_result_file)
    upload_to_drive(final_path, DRIVE_FOLDER_ID)
    print(f"Uploaded result to Drive: {final_path}")


def debug_list_files_in_folder(folder_id):
    response = (
        drive_service.files()
        .list(q=f"'{folder_id}' in parents", spaces="drive")
        .execute()
    )
    files = response.get("files", [])
    if not files:
        print("No files found in folder.")
    else:
        print("Files in folder:")
        for f in files:
            print(f["name"])


# === RUN ===
if __name__ == "__main__":
    user_prompt = input("Enter a prompt (e.g., 'restaurants in Los Angeles'): ")
    # run_tagging_workflow(user_prompt)
    debug_list_files_in_folder("1kL8VN9cCSYO1aw_yKySIwP7zhKRqy9ZM")
