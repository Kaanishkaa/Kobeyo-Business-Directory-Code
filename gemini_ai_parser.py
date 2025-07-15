# Save the updated version of the script using Gemini instead of OpenAI
import json
import re
import requests
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict
import spacy
import google.generativeai as genai


# === CONFIGURATION ===
GOOGLE_API_KEY = ""
GEMINI_API_KEY = ""
RULES_EXCEL_PATH = "animal_care_services.xlsx"

genai.configure(api_key=GEMINI_API_KEY)
nlp = spacy.load("en_core_web_sm")

# Load rules
rules_df = pd.read_excel(RULES_EXCEL_PATH)
rules_df = rules_df[rules_df["Prompt Rule"].notna() & rules_df["Skills IDs"].notna()][
    ["Skills Tags", "Prompt Rule", "Skills IDs", "Skills Names"]
].reset_index(drop=True)

# Map Skill ID to Name
skill_id_to_name = {}
for _, row in rules_df.iterrows():
    if isinstance(row["Skills Names"], str):
        matches = re.findall(r"(\\d+):\\s*([^>]+(?:>[^>]+)*)", row["Skills Names"])
        for skill_id, skill_name in matches:
            skill_id_to_name[int(skill_id)] = skill_name.strip()


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
    def __init__(self, google_api_key: str):
        self.google_api_key = google_api_key
        self.keyword_map = []
        for _, row in rules_df.iterrows():
            keywords = re.findall(r"\\b[a-zA-Z ]{3,}\\b", row["Prompt Rule"].lower())
            ids = [
                int(i.strip())
                for i in str(row["Skills IDs"]).split(",")
                if i.strip().isdigit()
            ]
            self.keyword_map.append((keywords, ids))

    # def extract_query_location(self, prompt: str) -> (str, str):
    #     doc = nlp(prompt)
    #     location = None
    #     service_keywords = []

    #     for ent in doc.ents:
    #         if ent.label_ in ["GPE", "LOC"]:
    #             location = ent.text

    #     for token in doc:
    #         if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
    #             service_keywords.append(token.text)

    #     query = " ".join(service_keywords)
    #     return query.strip(), location
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

        # Remove tokens that are part of the location to avoid duplicate wording
        cleaned_keywords = [
            kw
            for kw in service_keywords
            if location is None or location.lower() not in kw.lower()
        ]

        query = " ".join(cleaned_keywords)
        return query.strip(), location

    def search_all_places(self, query: str, location: str) -> List[Dict]:
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        all_results = []
        params = {"query": f"{query} in {location}", "key": self.google_api_key}

        while True:
            response = requests.get(url, params=params).json()
            all_results.extend(response.get("results", []))

            next_page_token = response.get("next_page_token")
            if not next_page_token:
                break
            import time

            time.sleep(2)
            params = {"pagetoken": next_page_token, "key": self.google_api_key}

        return all_results

    def analyze_with_gemini(self, name: str, types: List[str], address: str) -> Dict:
        context = f"Business types: {', '.join(types)}. Address: {address}."
        prompt = f"""
        Analyze this animal care business and identify what it likely does, what job roles it may hire for,
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
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return json.loads(response.text)

    def match_skill_ids_and_names(self, text: str) -> (List[int], List[str]):
        text = text.lower()
        matched_ids = set()
        for keywords, ids in self.keyword_map:
            if any(kw.strip() in text for kw in keywords):
                matched_ids.update(ids)
        matched_ids = sorted(matched_ids)
        matched_names = [
            skill_id_to_name[i] for i in matched_ids if i in skill_id_to_name
        ]
        return matched_ids, matched_names

    def process_place(self, place: Dict) -> Business:
        name = place.get("name", "Unknown")
        types = place.get("types", [])
        address = place.get("formatted_address", "")

        gpt_data = self.analyze_with_gemini(name, types, address)
        description = gpt_data.get("description", "")
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
    tagger = AnimalCareSkillTagger(GOOGLE_API_KEY)

    query, location = tagger.extract_query_location(user_prompt)
    if not query or not location:
        print("❌ Could not determine service or location.")
        exit()

    print(f"🔍 Searching for: {query} in {location}")
    places = tagger.search_all_places(query, location)

    existing_keys = set()
    try:
        existing_df = pd.read_csv("tagged_animal_businesses.csv")
        if "Name" in existing_df.columns and "Address" in existing_df.columns:
            existing_keys = set(
                (row["Name"].strip().lower(), row["Address"].strip().lower())
                for _, row in existing_df.iterrows()
            )
            print(f"📁 Loaded {len(existing_keys)} existing businesses from CSV.")
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
            print(f"⏭️ Skipping already scraped: {name} — {address}")
            continue

        print(f"➡️ Scraping {i}/{len(places)}: {name}")
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
            print(f"❌ Error processing {name}: {e}")

    if results:
        df_new = pd.DataFrame(results)
        final_df = pd.concat([existing_df, df_new], ignore_index=True)
        final_df.to_csv("tagged_animal_businesses.csv", index=False)
        print("✅ Appended new businesses to tagged_animal_businesses.csv")
    else:
        print("📭 No new businesses found to add.")
