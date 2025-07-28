"""file that identifies the skill tag sheet from the
business group picked from the universal or extracted_business_group_and_tpes csv"""

import requests
import google.generativeai as genai
from bs4 import BeautifulSoup
import pandas as pd
import json
import csv
import os

# === API KEYS ===
# GEMINI_API_KEY = "AIzaSyA9TywTucTuH0W3iTXABOGjwOox-chb9F8"
GEMINI_API_KEY = ""
GOOGLE_PLACES_API_KEY = ""

genai.configure(api_key=GEMINI_API_KEY)


# === Load Skill Rules ===
def load_skill_rules_from_excel(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    return df[df["Skills IDs"].notna() & df["Skills Names"].notna()][
        ["Skills Tags", "Prompt Rule", "Skills IDs", "Skills Names"]
    ].reset_index(drop=True)


# === User Prompt Parsing ===
def parse_prompt(prompt: str):
    model = genai.GenerativeModel("gemini-2.5-pro")
    res = model.generate_content(
        f"Extract business type and location from: '{prompt}'. "
        'Reply in JSON format: { "business_type": "", "location": "" }'
    )
    text = res.text.strip()
    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


# === Google Places API Search ===
def search_places(business_type: str, location: str):
    query = f"{business_type} in {location}"
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": GOOGLE_PLACES_API_KEY}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json().get("results", [])


# === Google Places Details ===
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
        "description": result.get("editorial_summary", {}).get(
            "overview", "No description available."
        ),
        "website": result.get("website", "No website available."),
    }


# === Website Content Scraping ===
def scrape_meaningful_content(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        description = (
            meta_desc["content"].strip()
            if meta_desc and "content" in meta_desc.attrs
            else ""
        )
        headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2"])]
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        list_items = [li.get_text(strip=True) for li in soup.find_all("li")]

        return f"""
Title: {title}
Meta Description: {description}
Headings: {' | '.join(headings)}
Paragraphs: {' | '.join(paragraphs)}
List Items: {' | '.join(list_items)}
""".strip()

    except Exception as e:
        return f"Error scraping {url}: {e}"


# # === Gemini: Business Analysis ===
# def analyze_business_with_gemini(
#     name: str, types: list, address: str, website_content: str = ""
# ) -> dict:
#     context = f"""
# Business types: {', '.join(types)}
# Address: {address}
# Website content: {website_content[:4000]}
# """
#     prompt = f"""
# Analyze the business below and describe what it likely does, what job roles it may hire for,
# and what types of skills would be required.

# Business Name: {name}
# Business Context:
# {context}


# Respond in JSON format:
# {{
#   "description": "...",
#   "potential_jobs": ["..."],
#   "required_skills": ["..."]
# }}
# """
#     model = genai.GenerativeModel("gemini-2.5-pro")
#     response = model.generate_content(prompt)
#     return json.loads(response.text.strip().replace("```json", "").replace("```", ""))
def analyze_business_with_gemini(
    name: str, types: list, address: str, website_content: str = ""
) -> dict:
    context = f"""
Business types: {', '.join(types)}
Address: {address}
Website content: {website_content[:4000]}
"""
    prompt = f"""
Analyze the business below and describe what it likely does, what job roles it may hire for,
and what types of skills would be required. Pay special attention to whether the business involves workers traveling to client locations for services.

Business Name: {name}
Business Context:
{context}

Respond in JSON format:
{{
  "description": "...",
  "potential_jobs": ["..."],
  "required_skills": ["..."],
  "is_travel_job": "yes/no"  # Include this field if the business provides services at client locations
}}
"""
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return json.loads(response.text.strip().replace("```json", "").replace("```", ""))


# # === Gemini: Bulk Skill Matching ===
# def match_skills_bulk_with_gemini(business_name, context, rules_df):
#     model = genai.GenerativeModel("gemini-2.5-pro")

#     rule_texts = "\n".join(
#         f"- {row['Prompt Rule']} => [IDs: {row['Skills IDs']}] [Names: {row['Skills Names']}]"
#         for _, row in rules_df.iterrows()
#     )

#     prompt = f"""
# You are an expert at tagging job-related skills. Based on the business context below,
# identify which of the following rules apply.

# Business Name: {business_name}
# Context:
# {context}

# Skill Tagging Rules:
# {rule_texts}


# Respond ONLY in JSON like:
# [
#   {{
#     "skills_ids": "...",
#     "skills_names": "..."
#   }}
# ]
# """
#     response = model.generate_content(prompt)
#     cleaned = response.text.strip().replace("```json", "").replace("```", "")
#     return json.loads(cleaned)
def match_skills_bulk_with_gemini(business_name, context, rules_df, is_travel_job):
    model = genai.GenerativeModel("gemini-2.5-pro")

    rule_texts = "\n".join(
        f"- {row['Prompt Rule']} => [IDs: {row['Skills IDs']}] [Names: {row['Skills Names']}]"
        for _, row in rules_df.iterrows()
    )

    # If it's a travel job, add a rule to match travel-related skills
    if is_travel_job:
        rule_texts += "\n- Travel Job => [IDs: 194, 200] [Names: Cleaning & Remediation>Management>Housekeeping services, Cleaning & Remediation>Service Area>Housekeeping services]"

    prompt = f"""
You are an expert at tagging job-related skills. Based on the business context below,
identify which of the following rules apply.

Business Name: {business_name}
Context:
{context}

Skill Tagging Rules:
{rule_texts}

Respond ONLY in JSON like:
[
  {{
    "skills_ids": "...",
    "skills_names": "..."
  }}
]
"""
    response = model.generate_content(prompt)
    cleaned = response.text.strip().replace("```json", "").replace("```", "")
    return json.loads(cleaned)


def load_existing_businesses(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    return set((row["Business Name"], row["Address"]) for _, row in df.iterrows())


# === Collect and Save Business Data ===
# def collect_business_data(places, rules_df):
#     results = []

#     for place in places:
#         name = place.get("name")
#         address = place.get("formatted_address", place.get("vicinity", "N/A"))
#         rating = place.get("rating", "N/A")
#         types = place.get("types", [])
#         place_id = place.get("place_id")

#         details = get_place_details(place_id)
#         website = details["website"]
#         website_content = (
#             scrape_meaningful_content(website) if website.startswith("http") else ""
#         )

#         combined_context = f"{', '.join(types)}\n{address}\n{website_content[:4000]}"

#         try:
#             ai_summary = analyze_business_with_gemini(
#                 name, types, address, website_content
#             )
#         except Exception as e:
#             ai_summary = {
#                 "description": "N/A",
#                 "potential_jobs": [],
#                 "required_skills": [],
#             }
#             print(f"AI analysis failed: {e}")

#         try:
#             matched_skills = match_skills_bulk_with_gemini(
#                 name, combined_context, rules_df
#             )
#         except Exception as e:
#             matched_skills = []
#             print(f"Skill tagging failed: {e}")

#         results.append(
#             {
#                 "Business Name": name,
#                 "Address": address,
#                 "Rating": rating,
#                 "Website": website,
#                 "AI Description": ai_summary.get("description", "N/A"),
#                 "Potential Jobs": ", ".join(ai_summary.get("potential_jobs", [])),
#                 "Required Skills": ", ".join(ai_summary.get("required_skills", [])),
#                 "Matched Skill IDs": ", ".join(
#                     [m["skills_ids"] for m in matched_skills]
#                 ),
#                 "Matched Skill Names": ", ".join(
#                     [m["skills_names"] for m in matched_skills]
#                 ),
#             }
#         )

#     return results


# def collect_business_data(places, rules_df, existing_businesses):
#     results = []

#     for place in places:
#         name = place.get("name")
#         address = place.get("formatted_address", place.get("vicinity", "N/A"))

#         if (name, address) in existing_businesses:
#             print(f"⏩ Skipping already scraped: {name}, {address}")
#             continue

#         rating = place.get("rating", "N/A")
#         types = place.get("types", [])
#         place_id = place.get("place_id")

#         details = get_place_details(place_id)
#         website = details["website"]
#         website_content = (
#             scrape_meaningful_content(website) if website.startswith("http") else ""
#         )

#         combined_context = f"{', '.join(types)}\n{address}\n{website_content[:4000]}"

#         try:
#             ai_summary = analyze_business_with_gemini(
#                 name, types, address, website_content
#             )
#         except Exception as e:
#             ai_summary = {
#                 "description": "N/A",
#                 "potential_jobs": [],
#                 "required_skills": [],
#             }
#             print(f"AI analysis failed: {e}")

#         try:
#             matched_skills = match_skills_bulk_with_gemini(
#                 name, combined_context, rules_df
#             )
#         except Exception as e:
#             matched_skills = []
#             print(f"Skill tagging failed: {e}")

#         results.append(
#             {
#                 "Business Name": name,
#                 "Address": address,
#                 "Rating": rating,
#                 "Website": website,
#                 "AI Description": ai_summary.get("description", "N/A"),
#                 "Potential Jobs": ", ".join(ai_summary.get("potential_jobs", [])),
#                 "Required Skills": ", ".join(ai_summary.get("required_skills", [])),
#                 "Matched Skill IDs": ", ".join(
#                     [m["skills_ids"] for m in matched_skills]
#                 ),
#                 "Matched Skill Names": ", ".join(
#                     [m["skills_names"] for m in matched_skills]
#                 ),
#             }
#         )


#     return results
# def collect_business_data(places, rules_df, existing_businesses):
#     results = []

#     for place in places:
#         name = place.get("name")
#         address = place.get("formatted_address", place.get("vicinity", "N/A"))
#         business_type = ", ".join(place.get("types", []))  # Combine business types

#         # Check if this business-location pair already exists
#         if (name, address) in existing_businesses:
#             print(f"⏩ Skipping already scraped: {name}, {address}")
#             continue

#         # Collect business data
#         rating = place.get("rating", "N/A")
#         place_id = place.get("place_id")

#         # Get additional details from Google Places API
#         details = get_place_details(place_id)
#         website = details["website"]
#         website_content = (
#             scrape_meaningful_content(website) if website.startswith("http") else ""
#         )

#         combined_context = (
#             f"{', '.join(place.get('types', []))}\n{address}\n{website_content[:4000]}"
#         )

#         # Analyze business with Gemini AI
#         try:
#             ai_summary = analyze_business_with_gemini(
#                 name, place.get("types", []), address, website_content
#             )
#         except Exception as e:
#             ai_summary = {
#                 "description": "N/A",
#                 "potential_jobs": [],
#                 "required_skills": [],
#             }
#             print(f"AI analysis failed: {e}")

#         # Match skills based on the business context
#         try:
#             matched_skills = match_skills_bulk_with_gemini(
#                 name, combined_context, rules_df
#             )
#         except Exception as e:
#             matched_skills = []
#             print(f"Skill tagging failed: {e}")

#         # Store results as individual rows
#         results.append(
#             {
#                 "Business Name": name,
#                 "Address": address,
#                 "Business Type": business_type,
#                 "Rating": rating,
#                 "Website": website,
#                 "AI Description": ai_summary.get("description", "N/A"),
#                 "Potential Jobs": ", ".join(ai_summary.get("potential_jobs", [])),
#                 "Required Skills": ", ".join(ai_summary.get("required_skills", [])),
#                 "Matched Skill IDs": ", ".join(
#                     [m["skills_ids"] for m in matched_skills]
#                 ),
#                 "Matched Skill Names": ", ".join(
#                     [m["skills_names"] for m in matched_skills]
#                 ),
#                 "Is Travel Job": ai_summary.get("is_travel_job", "no"),
#             }
#         )


#     return results
def collect_business_data(places, rules_df, existing_businesses):
    results = []

    for place in places:
        name = place.get("name")
        address = place.get("formatted_address", place.get("vicinity", "N/A"))
        business_type = ", ".join(place.get("types", []))  # Combine business types

        if (name, address) in existing_businesses:
            print(f"⏩ Skipping already scraped: {name}, {address}")
            continue

        rating = place.get("rating", "N/A")
        place_id = place.get("place_id")

        # Get additional details from Google Places API
        details = get_place_details(place_id)
        website = details["website"]
        website_content = (
            scrape_meaningful_content(website) if website.startswith("http") else ""
        )

        combined_context = (
            f"{', '.join(place.get('types', []))}\n{address}\n{website_content[:4000]}"
        )

        # Analyze business with Gemini AI
        try:
            ai_summary = analyze_business_with_gemini(
                name, place.get("types", []), address, website_content
            )
            is_travel_job = (
                ai_summary.get("is_travel_job", "no") == "yes"
            )  # Extract the travel job info
        except Exception as e:
            ai_summary = {
                "description": "N/A",
                "potential_jobs": [],
                "required_skills": [],
            }
            is_travel_job = False  # If analysis fails, assume not a travel job
            print(f"AI analysis failed: {e}")

        # Match skills based on the business context and travel job status
        try:
            matched_skills = match_skills_bulk_with_gemini(
                name, combined_context, rules_df, is_travel_job
            )
        except Exception as e:
            matched_skills = []
            print(f"Skill tagging failed: {e}")

        # Store results as individual rows
        results.append(
            {
                "Business Name": name,
                "Address": address,
                "Business Type": business_type,
                "Rating": rating,
                "Website": website,
                "AI Description": ai_summary.get("description", "N/A"),
                "Potential Jobs": ", ".join(ai_summary.get("potential_jobs", [])),
                "Required Skills": ", ".join(ai_summary.get("required_skills", [])),
                "Matched Skill IDs": ", ".join(
                    [m["skills_ids"] for m in matched_skills]
                ),
                "Matched Skill Names": ", ".join(
                    [m["skills_names"] for m in matched_skills]
                ),
                "Is Travel Job": (
                    "Yes" if is_travel_job else "No"
                ),  # Add this to your output
            }
        )

    return results


# def save_business_data_to_csv(data, output_file="business_analysis.csv"):
#     with open(output_file, mode="w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=data[0].keys())
#         writer.writeheader()
#         writer.writerows(data)


# this new change to check and append
def save_business_data_to_csv(data, output_file="business_analysis.csv"):
    file_exists = os.path.exists(output_file)
    with open(
        output_file, mode="a" if file_exists else "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)


def identify_business_group_from_csv(business_type: str, csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        types = [t.strip().lower() for t in row["Business Type"].split(",")]
        if business_type.lower() in types:
            return row["Business Group"]
    raise ValueError(f"❌ No matching business group found for '{business_type}'")


from sentence_transformers import SentenceTransformer, util

# Load the model once globally
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")


def identify_business_group_semantic(business_type: str, df: pd.DataFrame):
    business_type = business_type.lower()
    best_score = 0.0
    best_group = None

    for _, row in df.iterrows():
        types = [
            t.strip().lower() for t in str(row["Business Type"]).split(",") if t.strip()
        ]
        embeddings = semantic_model.encode(types, convert_to_tensor=True)
        input_embedding = semantic_model.encode(business_type, convert_to_tensor=True)

        scores = util.cos_sim(input_embedding, embeddings)[0]
        max_score = scores.max().item()

        if max_score > best_score:
            best_score = max_score
            best_group = row["Business Group"]

    if best_score > 0.6:  # you can tune this threshold
        return best_group
    else:
        print(
            f"⚠️ No strong semantic match found for: {business_type} (best score: {best_score:.2f})"
        )
        return None


# # === Main ===
# if __name__ == "__main__":
#     user_prompt = input(
#         "Enter your prompt (e.g., 'Pet grooming services in Austin, Texas'): "
#     )

#     try:
#         parsed = parse_prompt(user_prompt)
#         places = search_places(parsed["business_type"], parsed["location"])
#         # rules_df = load_skill_rules_from_excel("animal_care_services.xlsx")

#         df = pd.read_csv("Extracted_Business_Groups_and_Types.csv")
#         group = identify_business_group_semantic(parsed["business_type"], df)
#         print(group)
#         # group = identify_business_group_from_csv(
#         #     parsed["business_type"], "Extracted_Business_Groups_and_Types.csv"
#         # )
#         skill_file = (
#             f"{group.replace(' & ', '_and_').replace(' ', '_').replace(',', '')}.xlsx"
#         )
#         print(skill_file)
#         if not os.path.exists(skill_file):
#             raise FileNotFoundError(f"❌ Skill tag sheet '{skill_file}' not found.")

#         rules_df = load_skill_rules_from_excel(skill_file)

#         business_data = collect_business_data(places, rules_df)
#         save_business_data_to_csv(business_data)
#         print("✅ Business data saved to business_analysis.csv")
#     except Exception as e:
#         print(f"❌ Error: {e}")


if __name__ == "__main__":
    user_prompt = input(
        "Enter your prompt (e.g., 'Pet grooming services in Austin, Texas'): "
    )

    try:
        parsed = parse_prompt(user_prompt)
        places = search_places(parsed["business_type"], parsed["location"])

        df = pd.read_csv("Extracted_Business_Groups_and_Types.csv")
        group = identify_business_group_semantic(parsed["business_type"], df)
        print(group)

        skill_file = (
            f"{group.replace(' & ', '_and_').replace(' ', '_').replace(',', '')}.xlsx"
        )
        print(skill_file)
        if not os.path.exists(skill_file):
            raise FileNotFoundError(f"❌ Skill tag sheet '{skill_file}' not found.")

        rules_df = load_skill_rules_from_excel(skill_file)

        # Load existing business entries before scraping
        existing_businesses = load_existing_businesses("business_analysis.csv")

        business_data = collect_business_data(places, rules_df, existing_businesses)

        if business_data:
            save_business_data_to_csv(business_data)
            print("✅ New business data appended to business_analysis.csv")
        else:
            print("✅ No new businesses to add. All already exist.")

    except Exception as e:
        print(f"❌ Error: {e}")
