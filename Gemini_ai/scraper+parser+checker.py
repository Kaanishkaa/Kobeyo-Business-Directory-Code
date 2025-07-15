"""this code is built on the scraper+parser file that identifies the skill tag sheet from the business group picked from the universal or extracted_business_group_and_tpes csv
this code checks the manager_jobs_rows to skip the existing businesses and skill tag new businesses and append it
"""

import requests
import google.generativeai as genai
import pandas as pd
import json
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from time import sleep

from scraping_helpers import extract_website_content

# API Keys
GEMINI_API_KEY = ""
GOOGLE_PLACES_API_KEY = ""

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Load existing jobs CSV
existing_csv_path = "manager_jobs_rows.csv"
existing_df = pd.read_csv(existing_csv_path)

# Load business group mapping
business_group_mapping = pd.read_csv("Extracted_Business_Groups_and_Types.csv")


# Parse user prompt into business type and location
def parse_prompt(prompt: str):
    model = genai.GenerativeModel("gemini-2.5-pro")
    res = model.generate_content(
        f"Extract business type and location from: '{prompt}'. Reply only in this JSON format:\n"
        '{ "business_type": "", "location": "" }'
    )
    text = res.text.strip()
    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


# Identify business group from mapping
def get_business_group(business_type):
    for _, row in business_group_mapping.iterrows():
        if business_type.lower() in str(row["business_type"]).lower():
            return row["business_group"]
    return None


# Load skill tagging sheet based on business group
def load_skill_rules(group):
    filename = f"{group.lower().replace(' ', '_')}.xlsx"
    filepath = os.path.join(".", filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Skill tagging file not found: {filepath}")
    df = pd.read_excel(filepath)
    return df[df["Skills IDs"].notna() & df["Skills Names"].notna()][
        ["Skills Tags", "Prompt Rule", "Skills IDs", "Skills Names"]
    ]


# Search businesses using Google Places API
def search_places(business_type: str, location: str):
    print("🔍 Searching places...")
    query = f"{business_type} in {location}"
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": GOOGLE_PLACES_API_KEY}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json().get("results", [])[:10]


# Get website and editorial summary
def get_place_details(place_id: str):
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "editorial_summary,website",
        "key": GOOGLE_PLACES_API_KEY,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    result = response.json().get("result", {})
    return {
        "description": result.get("editorial_summary", {}).get("overview", ""),
        "website": result.get("website", ""),
    }


# Gemini-generated business analysis
def analyze_with_gemini(
    name: str, types: list, address: str, website_context: str
) -> dict:
    context = f"Business types: {', '.join(types)}. Address: {address}. Website says: {website_context}"
    prompt = f"""
    Analyze the business below and describe what it likely does, what job roles it may hire for,
    and what types of skills would be required.

    Business Name: {name}
    Business Context: {context}

    Please respond strictly in this JSON format:
    {{
      "description": "...",
      "potential_jobs": ["..."],
      "required_skills": ["..."]
    }}
    """
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return json.loads(response.text.strip().replace("```json", "").replace("```", ""))


# Skill matcher
def match_skills(required_skills: list, rules_df):
    matched_ids, matched_names = set(), set()
    for _, row in rules_df.iterrows():
        rule = row["Prompt Rule"].lower()
        for skill in required_skills:
            if rule in skill.lower():
                matched_ids.update(str(row["Skills IDs"]).split(","))
                matched_names.update(str(row["Skills Names"]).split(","))
    return ", ".join(sorted(matched_ids)), ", ".join(sorted(matched_names))


# Run the full tagging process
def process_places(places, rules_df, existing_df):
    new_rows = []
    print("⏳ Processing in background...")
    for place in places:
        name = place.get("name")
        address = place.get("formatted_address", place.get("vicinity", ""))
        identifier = f"{name}|{address}"

        if any(
            (existing_df["location_name"] + "|" + existing_df["area_address"])
            == identifier
        ):
            print(f"✅ Skipping already processed: {name}")
            continue

        types = place.get("types", [])
        place_id = place.get("place_id")
        details = get_place_details(place_id)
        website = details["website"]

        try:
            website_context = extract_website_content(website)
        except Exception as e:
            website_context = ""

        try:
            ai_data = analyze_with_gemini(name, types, address, website_context)
        except:
            ai_data = {
                "description": "AI failed",
                "potential_jobs": [],
                "required_skills": [],
            }

        skill_ids, skill_names = match_skills(ai_data["required_skills"], rules_df)

        new_rows.append(
            {
                "location_name": name,
                "area_address": address,
                "skills": skill_ids,
                "skill_names": skill_names,
            }
        )
    return pd.DataFrame(new_rows)


# Main
if __name__ == "__main__":
    user_prompt = input(
        "Enter your prompt (e.g., 'Pet grooming services in Austin, Texas'): "
    )
    parsed = parse_prompt(user_prompt)
    group = get_business_group(parsed["business_type"])
    if not group:
        print("❌ Could not determine business group.")
        exit()

    rules_df = load_skill_rules(group)
    places = search_places(parsed["business_type"], parsed["location"])
    new_data = process_places(places, rules_df, existing_df)

    if not new_data.empty:
        updated_df = pd.concat([existing_df, new_data], ignore_index=True)
        updated_df.to_csv(existing_csv_path, index=False)
        print(f"✅ Appended {len(new_data)} new businesses to {existing_csv_path}")
    else:
        print("⚠️ No new businesses to add.")
