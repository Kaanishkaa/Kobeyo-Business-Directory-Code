# import requests
# import openai
# import json
# import time
# import re
# from typing import List, Dict, Optional
# from dataclasses import dataclass
# import pandas as pd

# from openai import OpenAI

# client = OpenAI(api_key="your-key")

# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[{"role": "user", "content": prompt}],
#     temperature=0.2,
# )
# content = response.choices[0].message.content


# @dataclass
# class Business:
#     name: str
#     address: str
#     description: str
#     place_id: str
#     rating: Optional[float]
#     business_type: List[str]
#     potential_jobs: List[str]
#     required_skills: List[str]
#     job_categories: List[str]
#     skill_ids: List[int]


# class BusinessSkillTagger:
#     def __init__(self, openai_api_key: str, skill_rules_csv: str):
#         openai.api_key = openai_api_key
#         self.skill_dictionary = {
#             "food_service": [
#                 "cooking",
#                 "food preparation",
#                 "dishwashing",
#                 "bartending",
#             ],
#             "retail": ["sales", "cash register", "inventory"],
#             "hospitality": ["housekeeping", "guest services", "reservations"],
#             "office_admin": ["data entry", "receptionist", "scheduling"],
#             "manual_labor": ["construction", "delivery", "warehouse"],
#             "creative": ["graphic design", "marketing", "social media"],
#         }
#         # Load skill rules CSV
#         self.rules_df = pd.read_csv(skill_rules_csv)
#         self.rules_df = self.rules_df[
#             self.rules_df["Prompt Rule"].notna() & self.rules_df["Skills IDs"].notna()
#         ][["Skills Tags", "Prompt Rule", "Skills IDs"]]

#     def analyze_business_with_ai(
#         self, name: str, types: List[str], reviews: List[str], summary: str = ""
#     ) -> Dict:
#         review_text = " ".join(reviews[:3])
#         business_context = f"Business types: {', '.join(types)}. Summary: {summary}. Reviews: {review_text}"

#         prompt = f"""
#         Analyze this business and identify potential job opportunities and required skills:

#         Business Name: {name}
#         Business Context: {business_context}

#         Skill Dictionary:
#         {json.dumps(self.skill_dictionary, indent=2)}

#         Please respond in this format:
#         {{
#           "description": "...",
#           "potential_jobs": [...],
#           "required_skills": [...],
#           "job_categories": [...]
#         }}
#         """

#         # response = openai.ChatCompletion.create(
#         #     model="gpt-3.5-turbo",
#         #     messages=[{"role": "user", "content": prompt}],
#         #     temperature=0.2,
#         # )

#         # content = response.choices[0].message["content"]
#         client = OpenAI(
#             api_key=""
#         )

#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.2,
#         )
#         content = response.choices[0].message.content
#         try:
#             return json.loads(content)
#         except json.JSONDecodeError:
#             print("Failed to parse response:", content)
#             return {}

#     def tag_business_with_skill_ids(self, text: str) -> List[int]:
#         matched_ids = set()
#         for _, row in self.rules_df.iterrows():
#             keywords = re.split(r",|\n|;", row["Prompt Rule"].lower())
#             if any(kw.strip() and kw.strip() in text.lower() for kw in keywords):
#                 ids = [
#                     int(i.strip())
#                     for i in str(row["Skills IDs"]).split(",")
#                     if i.strip().isdigit()
#                 ]
#                 matched_ids.update(ids)
#         return sorted(matched_ids)

#     def process_business(
#         self, name, address, types, reviews, summary, place_id, rating
#     ) -> Business:
#         analysis = self.analyze_business_with_ai(name, types, reviews, summary)
#         description = analysis.get("description", "")
#         skill_ids = self.tag_business_with_skill_ids(description)

#         return Business(
#             name=name,
#             address=address,
#             description=description,
#             place_id=place_id,
#             rating=rating,
#             business_type=types,
#             potential_jobs=analysis.get("potential_jobs", []),
#             required_skills=analysis.get("required_skills", []),
#             job_categories=analysis.get("job_categories", []),
#             skill_ids=skill_ids,
#         )


# # === Example Usage ===
# if __name__ == "__main__":
#     tagger = BusinessSkillTagger(
#         openai_api_key="",
#         skill_rules_csv="skill tags - Sheet1.csv",
#     )

#     example_business = tagger.process_business(
#         name="Perch LA",
#         address="448 S Hill St, Los Angeles, CA",
#         types=["restaurant", "bar", "rooftop"],
#         reviews=[
#             "Amazing food and great rooftop view!",
#             "Service was quick and staff were friendly.",
#             "A bit expensive but worth it for the ambiance.",
#         ],
#         summary="Trendy rooftop restaurant and bar in downtown LA.",
#         place_id="example123",
#         rating=4.5,
#     )

#     print(example_business)


import json
import re
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
from openai import OpenAI


@dataclass
class Business:
    name: str
    address: str
    description: str
    place_id: str
    rating: Optional[float]
    business_type: List[str]
    potential_jobs: List[str]
    required_skills: List[str]
    job_categories: List[str]
    skill_ids: List[int]


class BusinessSkillTagger:
    def __init__(self, openai_api_key: str, skill_rules_csv: str):
        self.client = OpenAI(api_key=openai_api_key)

        self.skill_dictionary = {
            "food_service": [
                "cooking",
                "food preparation",
                "dishwashing",
                "bartending",
            ],
            "retail": ["sales", "cash register", "inventory"],
            "hospitality": ["housekeeping", "guest services", "reservations"],
            "office_admin": ["data entry", "receptionist", "scheduling"],
            "manual_labor": ["construction", "delivery", "warehouse"],
            "creative": ["graphic design", "marketing", "social media"],
        }

        # Load tagging rules from CSV
        self.rules_df = pd.read_csv(skill_rules_csv)
        self.rules_df = self.rules_df[
            self.rules_df["Prompt Rule"].notna() & self.rules_df["Skills IDs"].notna()
        ][["Skills Tags", "Prompt Rule", "Skills IDs"]]

    def analyze_business_with_ai(
        self, name: str, types: List[str], reviews: List[str], summary: str = ""
    ) -> Dict:
        review_text = " ".join(reviews[:3])
        business_context = f"Business types: {', '.join(types)}. Summary: {summary}. Reviews: {review_text}"

        prompt = f"""
        Analyze this business and identify potential job opportunities and required skills:

        Business Name: {name}
        Business Context: {business_context}

        Skill Dictionary:
        {json.dumps(self.skill_dictionary, indent=2)}

        Please respond in this format:
        {{
          "description": "...",
          "potential_jobs": [...],
          "required_skills": [...],
          "job_categories": [...]
        }}
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print("Failed to parse response:", content)
            return {}

    def tag_business_with_skill_ids(self, text: str) -> List[int]:
        matched_ids = set()
        for _, row in self.rules_df.iterrows():
            keywords = re.split(r",|\n|;", row["Prompt Rule"].lower())
            if any(kw.strip() and kw.strip() in text.lower() for kw in keywords):
                ids = [
                    int(i.strip())
                    for i in str(row["Skills IDs"]).split(",")
                    if i.strip().isdigit()
                ]
                matched_ids.update(ids)
        return sorted(matched_ids)

    def process_business(
        self, name, address, types, reviews, summary, place_id, rating
    ) -> Business:
        analysis = self.analyze_business_with_ai(name, types, reviews, summary)
        description = analysis.get("description", "")
        skill_ids = self.tag_business_with_skill_ids(description)

        return Business(
            name=name,
            address=address,
            description=description,
            place_id=place_id,
            rating=rating,
            business_type=types,
            potential_jobs=analysis.get("potential_jobs", []),
            required_skills=analysis.get("required_skills", []),
            job_categories=analysis.get("job_categories", []),
            skill_ids=skill_ids,
        )


# === Example Usage ===
if __name__ == "__main__":
    tagger = BusinessSkillTagger(
        openai_api_key="",
        skill_rules_csv="skill tags - Sheet1.csv",
    )

    example_business = tagger.process_business(
        name="Perch LA",
        address="448 S Hill St, Los Angeles, CA",
        types=["restaurant", "bar", "rooftop"],
        reviews=[
            "Amazing food and great rooftop view!",
            "Service was quick and staff were friendly.",
            "A bit expensive but worth it for the ambiance.",
        ],
        summary="Trendy rooftop restaurant and bar in downtown LA.",
        place_id="example123",
        rating=4.5,
    )

    print(example_business)
