# Construct the updated Python script with business group-based skill tagging
import json
import re
import requests
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict
import spacy
from openai import OpenAI

# Load spaCy's NLP model
nlp = spacy.load("en_core_web_sm")

# === Configuration ===
OPENAI_API_KEY = ""
GOOGLE_API_KEY = ""

# === Load Business Group Type Mapping ===
business_group_df = pd.read_csv("Extracted_Business_Groups_and_Types.csv")

# Map Business Group to Excel sheet file
business_group_to_excel = {
    "Animal Care & Services": "animal_care_services.xlsx",
    "Food & Beverage Establishments": "food_beverage_services.xlsx",
    "Real Estate & Property Management": "real_estate_services.xlsx",
    "Cleaning & Remediation": "cleaning_services.xlsx",
    "Security": "security_services.xlsx",
}


def find_business_group(prompt: str) -> str:
    prompt_lower = prompt.lower()
    for _, row in business_group_df.iterrows():
        business_types = str(row["Business Type"]).lower().split(",")
        for bt in business_types:
            if bt.strip() in prompt_lower:
                return row["Business Group"]
    return None


@dataclass
class Business:
    name: str
    description: str
    business_type: List[str]
    potential_jobs: List[str]
    required_skills: List[str]
    skill_ids: List[int]
    skill_names: List[str]


class AnimalCareSkillTagger:
    def __init__(
        self, openai_api_key: str, google_api_key: str, rules_df: pd.DataFrame
    ):
        self.client = OpenAI(api_key=openai_api_key)
        self.google_api_key = google_api_key
        self.rules_df = rules_df

        self.keyword_map = []
        for _, row in rules_df.iterrows():
            keywords = re.findall(
                r"\\b[a-zA-Z ]{3,}\\b", str(row["Prompt Rule"]).lower()
            )
            ids = [
                int(i.strip())
                for i in str(row["Skills IDs"]).split(",")
                if i.strip().isdigit()
            ]
            self.keyword_map.append((keywords, ids))

        self.skill_id_to_name = {}
        for _, row in rules_df.iterrows():
            if isinstance(row["Skills Names"], str):
                matches = re.findall(
                    r"(\\d+):\\s*([^>]+(?:>[^>]+)*)", row["Skills Names"]
                )
                for skill_id, skill_name in matches:
                    self.skill_id_to_name[int(skill_id)] = skill_name.strip()

    def extract_query_location(self, prompt: str) -> (str, str):
        doc = nlp(prompt)
        location = None
        service_keywords = []

        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                location = ent.text

        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
                service_keywords.append(token.text)

        query = " ".join(service_keywords)
        return query.strip(), location

    def search_places(self, query: str, location: str) -> List[Dict]:
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {"query": f"{query} in {location}", "key": self.google_api_key}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json().get("results", [])
        return []

    def analyze_with_openai(self, name: str, types: List[str], address: str) -> Dict:
        context = f"Business types: {', '.join(types)}. Address: {address}."
        prompt = f"""
        Analyze this business and identify what it likely does, what job roles it may hire for,
        and what types of skills would be required.

        Business Name: {name}
        Business Context: {context}

        Please respond in this format:
        {{
          "description": "...",
          "potential_jobs": [...],
          "required_skills": [...]
        }}
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        return json.loads(content)

    def match_skill_ids_and_names(self, text: str) -> (List[int], List[str]):
        text = text.lower()
        matched_ids = set()
        for keywords, ids in self.keyword_map:
            if any(kw.strip() in text for kw in keywords):
                matched_ids.update(ids)
        matched_ids = sorted(matched_ids)
        matched_names = [
            self.skill_id_to_name[i] for i in matched_ids if i in self.skill_id_to_name
        ]
        print("Matching text:", text)
        return matched_ids, matched_names

    def process_place(self, place: Dict) -> Business:
        name = place.get("name", "Unknown")
        types = place.get("types", [])
        address = place.get("formatted_address", "")

        gpt_data = self.analyze_with_openai(name, types, address)
        description = gpt_data.get("description", "")
        print("GPT Output:", gpt_data)

        skill_ids, skill_names = self.match_skill_ids_and_names(description)

        return Business(
            name=name,
            description=description,
            business_type=types,
            potential_jobs=gpt_data.get("potential_jobs", []),
            required_skills=gpt_data.get("required_skills", []),
            skill_ids=skill_ids,
            skill_names=skill_names,
        )


if __name__ == "__main__":
    user_prompt = input(
        "Enter a prompt (e.g., 'Animal care services in Los Angeles'): "
    )

    business_group = find_business_group(user_prompt)
    if not business_group:
        print("Could not identify the business group from the prompt.")
        exit()

    excel_path = business_group_to_excel.get(business_group)
    if not excel_path:
        print(f"No skill tag sheet mapped for business group: {business_group}")
        exit()

    print(f"Identified business group: {business_group}")
    print(f"Using skill tag file: {excel_path}")

    rules_df = pd.read_excel(excel_path)
    tagger = AnimalCareSkillTagger(OPENAI_API_KEY, GOOGLE_API_KEY, rules_df)

    query, location = tagger.extract_query_location(user_prompt)
    if not query or not location:
        print("Could not determine service or location.")
        exit()

    print(f"🔍 Searching for: {query} in {location}")
    places = tagger.search_places(query, location)

    existing_keys = set()
    try:
        existing_df = pd.read_csv("tagged_animal_businesses.csv")
        if "Name" in existing_df.columns and "Address" in existing_df.columns:
            existing_keys = set(
                (row["Name"].strip().lower(), row["Address"].strip().lower())
                for _, row in existing_df.iterrows()
            )
            print(f"Loaded {len(existing_keys)} existing businesses from CSV.")
        else:
            existing_df = pd.DataFrame()
    except FileNotFoundError:
        print("📁 No existing CSV found. A new one will be created.")
        existing_df = pd.DataFrame()

    results = []
    for i, place in enumerate(places, 1):
        name = place.get("name", "").strip()
        address = place.get("formatted_address", "").strip()
        key = (name.lower(), address.lower())

        if key in existing_keys:
            print(f"⏭Skipping already scraped: {name} — {address}")
            continue

        print(f"Scraping {i}/{len(places)}: {name}")
        try:
            result = tagger.process_place(place)
            results.append(
                {
                    "Name": result.name,
                    "Address": address,
                    "Description": result.description,
                    "Business Types": ", ".join(result.business_type),
                    "Potential Jobs": ", ".join(result.potential_jobs),
                    "Required Skills": ", ".join(result.required_skills),
                    "Skill IDs": ", ".join(map(str, result.skill_ids)),
                    "Skill Names": ", ".join(result.skill_names),
                }
            )
        except Exception as e:
            print(f"Error processing {name}: {e}")

    if results:
        df_new = pd.DataFrame(results)
        final_df = pd.concat([existing_df, df_new], ignore_index=True)
        final_df.to_csv("tagged_animal_businesses.csv", index=False)
        print("Appended new businesses to tagged_animal_businesses.csv")
    else:
        print("No new businesses found to add.")
