import requests
import google.generativeai as genai
import json

# API Keys
GEMINI_API_KEY = ""
GOOGLE_PLACES_API_KEY = ""

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


# Parse user prompt into business type and location
def parse_prompt(prompt: str):
    model = genai.GenerativeModel("gemini-2.5-pro")
    res = model.generate_content(
        f"Extract business type and location from: '{prompt}'. "
        "Reply only in this JSON format:\n"
        '{ "business_type": "", "location": "" }'
    )
    text = res.text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"❌ Could not parse Gemini response: {res.text}") from e


# Search businesses using Google Places API
def search_places(business_type: str, location: str):
    query = f"{business_type} in {location}"
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": GOOGLE_PLACES_API_KEY}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json().get("results", [])[:2]


# Get website and editorial summary (fallbacks)
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

    description = result.get("editorial_summary", {}).get("overview")
    website = result.get("website")

    return {
        "description": description or "No description available.",
        "website": website or "No website available.",
    }


# Gemini-generated business analysis
def analyze_with_gemini(name: str, types: list, address: str) -> dict:
    context = f"Business types: {', '.join(types)}. Address: {address}."
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

    try:
        return json.loads(
            response.text.strip().replace("```json", "").replace("```", "")
        )
    except Exception as e:
        raise ValueError(f"❌ Gemini analysis error: {response.text}") from e


# Print business results
def print_places(places):
    if not places:
        print("No results found.")
        return
    for place in places:
        name = place.get("name")
        address = place.get("formatted_address", place.get("vicinity", "N/A"))
        rating = place.get("rating", "N/A")
        types = place.get("types", [])
        place_id = place.get("place_id")

        # Google details (website)
        details = get_place_details(place_id)

        # Gemini description, jobs, skills
        try:
            ai_summary = analyze_with_gemini(name, types, address)
        except Exception as e:
            ai_summary = {
                "description": "AI description failed.",
                "potential_jobs": [],
                "required_skills": [],
            }

        print(f"📍 {name}")
        print(f"   Address: {address}")
        print(f"   Rating: {rating}")
        print(f"   Website: {details['website']}")
        print(f"   Description: {ai_summary['description']}")
        print(f"   Jobs: {', '.join(ai_summary['potential_jobs']) or 'N/A'}")
        print(f"   Skills: {', '.join(ai_summary['required_skills']) or 'N/A'}\n")


# Main
if __name__ == "__main__":
    user_prompt = input(
        "Enter your prompt (e.g., 'Pet grooming services in Austin, Texas'): "
    )

    try:
        parsed = parse_prompt(user_prompt)
        places = search_places(parsed["business_type"], parsed["location"])
        print_places(places)
    except Exception as e:
        print(f"❌ Error: {e}")
