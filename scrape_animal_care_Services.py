import json
import re
import requests
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
from openai import OpenAI


@dataclass
class Business:
    name: str
    description: str
    business_type: List[str]
    potential_jobs: List[str]
    required_skills: List[str]
    skill_ids: List[int]


class AnimalCareSkillTagger:
    def __init__(self, openai_api_key: str, google_api_key: str, rules_excel: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.google_api_key = google_api_key

        # Load and clean rules
        df = pd.read_excel(rules_excel)
        self.rules_df = df[df["Prompt Rule"].notna() & df["Skills IDs"].notna()][
            ["Skills Tags", "Prompt Rule", "Skills IDs"]
        ].reset_index(drop=True)

        # Extract keywords for matching
        self.keyword_map = []
        for _, row in self.rules_df.iterrows():
            keywords = re.findall(r"\b[a-zA-Z ]{3,}\b", row["Prompt Rule"].lower())
            ids = [
                int(i.strip())
                for i in str(row["Skills IDs"]).split(",")
                if i.strip().isdigit()
            ]
            self.keyword_map.append((keywords, ids))

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
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except Exception as e:
            print("GPT Parse Error:", e)
            return {}

    def match_skill_ids(self, text: str) -> List[int]:
        text = text.lower()
        matched_ids = set()
        for keywords, ids in self.keyword_map:
            if any(kw.strip() in text for kw in keywords):
                matched_ids.update(ids)
        return sorted(matched_ids)

    def process_place(self, place: Dict) -> Business:
        name = place.get("name", "Unknown")
        types = place.get("types", [])
        address = place.get("formatted_address", "")

        gpt_data = self.analyze_with_openai(name, types, address)
        description = gpt_data.get("description", "")
        skill_ids = self.match_skill_ids(description)

        return Business(
            name=name,
            description=description,
            business_type=types,
            potential_jobs=gpt_data.get("potential_jobs", []),
            required_skills=gpt_data.get("required_skills", []),
            skill_ids=skill_ids,
        )


# === Example usage ===
if __name__ == "__main__":
    tagger = AnimalCareSkillTagger(
        openai_api_key="",
        google_api_key="",
        rules_excel="animal_care_services.xlsx",
    )

    # Search for businesses
    places = tagger.search_places("animal care services", "San Francisco")

    # Tag top 3 businesses
    for place in places[:3]:
        result = tagger.process_place(place)
        print(json.dumps(result.__dict__, indent=2))
