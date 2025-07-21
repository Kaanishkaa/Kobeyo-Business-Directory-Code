"""file that identifies the skill tag sheet from the
business group picked from the universal or extracted_business_group_and_tpes csv"""

# import requests
# import google.generativeai as genai
# from bs4 import BeautifulSoup
# import json
# import pandas as pd

# # API Keys
# GEMINI_API_KEY = ""
# GOOGLE_PLACES_API_KEY = ""

# # Configure Gemini
# genai.configure(api_key=GEMINI_API_KEY)


# def parse_prompt(prompt: str):
#     model = genai.GenerativeModel("gemini-2.5-pro")
#     res = model.generate_content(
#         f"Extract business type and location from: '{prompt}'. "
#         "Reply only in this JSON format:\n"
#         '{ "business_type": "", "location": "" }'
#     )
#     text = res.text.strip()
#     if text.startswith("```json"):
#         text = text.replace("```json", "").replace("```", "").strip()
#     return json.loads(text)


# def search_places(business_type: str, location: str):
#     query = f"{business_type} in {location}"
#     url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
#     params = {"query": query, "key": GOOGLE_PLACES_API_KEY}
#     response = requests.get(url, params=params)
#     response.raise_for_status()
#     return response.json().get("results", [])[:5]


# def get_place_details(place_id: str):
#     url = "https://maps.googleapis.com/maps/api/place/details/json"
#     params = {
#         "place_id": place_id,
#         "fields": "editorial_summary,website",
#         "key": GOOGLE_PLACES_API_KEY,
#     }
#     response = requests.get(url, params=params)
#     response.raise_for_status()
#     result = response.json().get("result", {})
#     return {
#         "description": result.get("editorial_summary", {}).get(
#             "overview", "No description available."
#         ),
#         "website": result.get("website", "No website available."),
#     }


# def scrape_meaningful_content(url: str) -> str:
#     try:
#         headers = {"User-Agent": "Mozilla/5.0 (compatible; GPT-4 WebScraper/1.0)"}
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.text, "html.parser")
#         for tag in soup(["script", "style", "noscript"]):
#             tag.decompose()

#         title = soup.title.string.strip() if soup.title else ""
#         meta_desc = soup.find("meta", attrs={"name": "description"})
#         description = (
#             meta_desc["content"].strip()
#             if meta_desc and "content" in meta_desc.attrs
#             else ""
#         )

#         headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2"])]
#         paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
#         list_items = [li.get_text(strip=True) for li in soup.find_all("li")]

#         content = f"""
# Title:
# {title}

# Meta Description:
# {description}

# Headings:
# {chr(10).join('- ' + h for h in headings)}

# Paragraphs:
# {chr(10).join('- ' + p for p in paragraphs)}

# List Items:
# {chr(10).join('- ' + li for li in list_items)}
# """
#         return content.strip()
#     except Exception as e:
#         return f"Error scraping {url}: {e}"


# # def analyze_business_with_gemini(
# #     name: str, types: list, address: str, website_content: str = ""
# # ) -> dict:
# #     context = f"""
# # Business types: {', '.join(types)}
# # Address: {address}

# # Website content:
# # {website_content[:4000]}
# # """
# #     prompt = f"""
# # Analyze the business below and describe what it likely does, what job roles it may hire for,
# # and what types of skills would be required.

# # Business Name: {name}
# # Business Context:
# # {context}

# # Respond in JSON format:
# # {{
# #   "description": "...",
# #   "potential_jobs": ["..."],
# #   "required_skills": ["..."]
# # }}
# # """
# #     model = genai.GenerativeModel("gemini-2.5-pro")
# #     response = model.generate_content(prompt)
# #     try:
# #         return json.loads(
# #             response.text.strip().replace("```json", "").replace("```", "")
# #         )
# #     except Exception as e:
# #         raise ValueError(f"❌ Gemini response error: {response.text}") from e


# # def print_places(places):
# #     for place in places:
# #         name = place.get("name")
# #         address = place.get("formatted_address", place.get("vicinity", "N/A"))
# #         rating = place.get("rating", "N/A")
# #         types = place.get("types", [])
# #         place_id = place.get("place_id")

# #         details = get_place_details(place_id)
# #         website = details["website"]
# #         print(f"\n📍 {name}")
# #         print(f"   Address: {address}")
# #         print(f"   Rating: {rating}")
# #         print(f"   Website: {website}")

# #         website_content = ""
# #         if website.startswith("http"):
# #             print(f"🔍 Scraping {website} ...")
# #             website_content = scrape_meaningful_content(website)

# #         try:
# #             ai_summary = analyze_business_with_gemini(
# #                 name, types, address, website_content
# #             )
# #         except Exception as e:
# #             print(f"❌ Gemini error: {e}")
# #             ai_summary = {
# #                 "description": "N/A",
# #                 "potential_jobs": [],
# #                 "required_skills": [],
# #             }

# #         print(f"   📄 Description: {ai_summary['description']}")
# #         print(f"   💼 Jobs: {', '.join(ai_summary['potential_jobs']) or 'N/A'}")
# #         print(f"   🧠 Skills: {', '.join(ai_summary['required_skills']) or 'N/A'}")


# # Integrate skill tagging logic with the existing print_places flow


# def analyze_business_with_gemini(
#     name: str, types: list, address: str, website_content: str = ""
# ) -> dict:
#     context = f"""
# Business types: {', '.join(types)}
# Address: {address}

# Website content:
# {website_content[:4000]}
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
#     try:
#         return json.loads(
#             response.text.strip().replace("```json", "").replace("```", "")
#         )
#     except Exception as e:
#         raise ValueError(f"❌ Gemini response error: {response.text}") from e


# # this commented code send the prompt rule one by one for skill match a very long method
# # def match_skills_with_gemini(business_name, context, rules_df):
# #     model = genai.GenerativeModel("gemini-2.5-pro")
# #     matched_skills = []

# #     for _, row in rules_df.iterrows():
# #         rule_description = row["Prompt Rule"]
# #         skill_ids = row["Skills IDs"]
# #         skill_names = row["Skills Names"]

# #         prompt = f"""
# # You are an expert at tagging job-related skills. Based on the business context below,
# # determine whether the following skill tagging rule applies.

# # Business Name: {business_name}
# # Business Context:
# # {context}

# # Skill Tag Rule:
# # {rule_description}

# # Reply only with "YES" or "NO".
# # """
# #         try:
# #             response = model.generate_content(prompt)
# #             answer = response.text.strip().lower()
# #             if "yes" in answer:
# #                 matched_skills.append(
# #                     {
# #                         "Skills IDs": skill_ids,
# #                         "Skills Names": skill_names,
# #                         "Tag Description": rule_description[:100],
# #                     }
# #                 )
# #         except Exception as e:
# #             print(f"❌ Error evaluating rule: {e}")

# #     return matched_skills


# def match_skills_bulk_with_gemini(business_name, context, rules_df):
#     model = genai.GenerativeModel("gemini-2.5-pro")

#     # Build the prompt with all rules
#     rule_descriptions = []
#     for _, row in rules_df.iterrows():
#         rule_descriptions.append(
#             f"- {row['Prompt Rule'].strip()} => [IDs: {row['Skills IDs']}] [Names: {row['Skills Names']}]"
#         )

#     rules_text = "\n".join(rule_descriptions)

#     prompt = f"""
# You are an expert at tagging job-related skills. Based on the business context below,
# identify which of the following skill tagging rules apply.

# Business Name: {business_name}

# Business Context:
# {context}

# Skill Tagging Rules:
# {rules_text}

# Respond ONLY with the matched skill names and IDs in a JSON format like:
# [
#   {{
#     "skills_ids": "...",
#     "skills_names": "..."
#   }},
#   ...
# ]
# """

#     try:
#         response = model.generate_content(prompt)
#         cleaned_text = response.text.strip().replace("```json", "").replace("```", "")
#         return json.loads(cleaned_text)
#     except Exception as e:
#         print(f"❌ Error in bulk skill tagging: {e}")
#         return []


# def print_places_with_skills(places, rules_df):
#     for place in places:
#         name = place.get("name")
#         address = place.get("formatted_address", place.get("vicinity", "N/A"))
#         rating = place.get("rating", "N/A")
#         types = place.get("types", [])
#         place_id = place.get("place_id")

#         details = get_place_details(place_id)
#         website = details["website"]

#         print(f"\n📍 {name}")
#         print(f"   Address: {address}")
#         print(f"   Rating: {rating}")
#         print(f"   Website: {website}")

#         website_content = ""
#         if website.startswith("http"):
#             print(f"🔍 Scraping {website} ...")
#             website_content = scrape_meaningful_content(website)

#         # Combine for analysis
#         combined_context = f"Business types: {', '.join(types)}\nAddress: {address}\n\nWebsite Content:\n{website_content[:4000]}"

#         try:
#             ai_summary = analyze_business_with_gemini(
#                 name, types, address, website_content
#             )
#         except Exception as e:
#             print(f"❌ Gemini error: {e}")
#             ai_summary = {
#                 "description": "N/A",
#                 "potential_jobs": [],
#                 "required_skills": [],
#             }

#         print(f"   📄 Description: {ai_summary['description']}")
#         print(f"   💼 Jobs: {', '.join(ai_summary['potential_jobs']) or 'N/A'}")
#         print(f"   🧠 Skills: {', '.join(ai_summary['required_skills']) or 'N/A'}")

#         # Skill Tagging
#         matched_skills = match_skills_bulk_with_gemini(name, combined_context, rules_df)
#         if matched_skills:
#             print(f"   🏷️ Matched Skill Tags:")
#             for match in matched_skills:
#                 print(f"     - {match['skills_names']} (IDs: {match['skills_ids']})")
#         else:
#             print("   🏷️ No skill tags matched.")

#         # matched_skills = match_skills_with_gemini(name, combined_context, rules_df)
#         # if matched_skills:
#         #     print(f"   🏷️ Matched Skill Tags:")
#         #     for match in matched_skills:
#         #         print(f"     - {match['Skills Names']} (IDs: {match['Skills IDs']})")
#         # else:
#         #     print("   🏷️ No skill tags matched.")


# def load_skill_rules_from_excel(filepath: str) -> pd.DataFrame:
#     # Load the Excel sheet
#     df = pd.read_excel(filepath)

#     # Filter out rows missing Skill IDs or Skill Names
#     cleaned_df = df[df["Skills IDs"].notna() & df["Skills Names"].notna()][
#         ["Skills Tags", "Prompt Rule", "Skills IDs", "Skills Names"]
#     ].reset_index(drop=True)

#     return cleaned_df


# if __name__ == "__main__":
#     user_prompt = input(
#         "Enter your prompt (e.g., 'Pet grooming services in Austin, Texas'): "
#     )
#     try:
#         parsed = parse_prompt(user_prompt)
#         places = search_places(parsed["business_type"], parsed["location"])
#         # print_places(places)
#         rules_df = load_skill_rules_from_excel("animal_care_services.xlsx")
#         print_places_with_skills(places, rules_df)
#     except Exception as e:
#         print(f"❌ Error: {e}")


import requests
import google.generativeai as genai
from bs4 import BeautifulSoup
import pandas as pd
import json
import csv
import os

# === API KEYS ===
GEMINI_API_KEY = "AIzaSyA9TywTucTuH0W3iTXABOGjwOox-chb9F8"
GOOGLE_PLACES_API_KEY = "AIzaSyBnzR1YLd-SOwjoU1XYB2Rce6We0dvvmN4"

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


# === Gemini: Business Analysis ===
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
and what types of skills would be required.

Business Name: {name}
Business Context:
{context}

Respond in JSON format:
{{
  "description": "...",
  "potential_jobs": ["..."],
  "required_skills": ["..."]
}}
"""
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return json.loads(response.text.strip().replace("```json", "").replace("```", ""))


# === Gemini: Bulk Skill Matching ===
def match_skills_bulk_with_gemini(business_name, context, rules_df):
    model = genai.GenerativeModel("gemini-2.5-pro")

    rule_texts = "\n".join(
        f"- {row['Prompt Rule']} => [IDs: {row['Skills IDs']}] [Names: {row['Skills Names']}]"
        for _, row in rules_df.iterrows()
    )

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


# === Collect and Save Business Data ===
def collect_business_data(places, rules_df):
    results = []

    for place in places:
        name = place.get("name")
        address = place.get("formatted_address", place.get("vicinity", "N/A"))
        rating = place.get("rating", "N/A")
        types = place.get("types", [])
        place_id = place.get("place_id")

        details = get_place_details(place_id)
        website = details["website"]
        website_content = (
            scrape_meaningful_content(website) if website.startswith("http") else ""
        )

        combined_context = f"{', '.join(types)}\n{address}\n{website_content[:4000]}"

        try:
            ai_summary = analyze_business_with_gemini(
                name, types, address, website_content
            )
        except Exception as e:
            ai_summary = {
                "description": "N/A",
                "potential_jobs": [],
                "required_skills": [],
            }
            print(f"AI analysis failed: {e}")

        try:
            matched_skills = match_skills_bulk_with_gemini(
                name, combined_context, rules_df
            )
        except Exception as e:
            matched_skills = []
            print(f"Skill tagging failed: {e}")

        results.append(
            {
                "Business Name": name,
                "Address": address,
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
            }
        )

    return results


def save_business_data_to_csv(data, output_file="business_analysis.csv"):
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def identify_business_group_from_csv(business_type: str, csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        types = [t.strip().lower() for t in row["Business Type"].split(",")]
        if business_type.lower() in types:
            return row["Business Group"]
    raise ValueError(f"❌ No matching business group found for '{business_type}'")


# === Main ===
if __name__ == "__main__":
    user_prompt = input(
        "Enter your prompt (e.g., 'Pet grooming services in Austin, Texas'): "
    )

    try:
        parsed = parse_prompt(user_prompt)
        places = search_places(parsed["business_type"], parsed["location"])
        # rules_df = load_skill_rules_from_excel("animal_care_services.xlsx")

        group = identify_business_group_from_csv(
            parsed["business_type"], "Extracted_Business_Groups_and_Types.csv"
        )
        skill_file = (
            f"{group.replace(' & ', '_and_').replace(' ', '_').replace(',', '')}.xlsx"
        )
        print(skill_file)
        if not os.path.exists(skill_file):
            raise FileNotFoundError(f"❌ Skill tag sheet '{skill_file}' not found.")

        rules_df = load_skill_rules_from_excel(skill_file)

        business_data = collect_business_data(places, rules_df)
        save_business_data_to_csv(business_data)
        print("✅ Business data saved to business_analysis.csv")
    except Exception as e:
        print(f"❌ Error: {e}")
