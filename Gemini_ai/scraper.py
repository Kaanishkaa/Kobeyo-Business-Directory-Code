import requests
from bs4 import BeautifulSoup

GOOGLE_PLACES_API_KEY = ""


def search_places(business_type, location):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": f"{business_type} in {location}", "key": GOOGLE_PLACES_API_KEY}
    response = requests.get(url, params=params)
    return response.json().get("results", [])[:10]


def get_place_details(place_id):
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {"place_id": place_id, "fields": "website", "key": GOOGLE_PLACES_API_KEY}
    response = requests.get(url, params=params)
    result = response.json().get("result", {})
    return result.get("website")


def scrape_meaningful_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; GPT-4 WebScraper/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Extract key content blocks
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

        # Format output cleanly
        content = f"""
                Title:
                {title}

                Meta Description:
                {description}

                Headings:
                {chr(10).join('- ' + h for h in headings)}

                Paragraphs:
                {chr(10).join('- ' + p for p in paragraphs)}

                List Items:
                {chr(10).join('- ' + li for li in list_items)}
                """

        return content.strip()

    except Exception as e:
        return f"Error scraping {url}: {e}"


# def scrape_meaningful_content(url):
#     try:
#         headers = {"User-Agent": "Mozilla/5.0 (compatible; GPT-4 WebScraper/1.0)"}
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.text, "html.parser")

#         for tag in soup(["script", "style", "noscript"]):
#             tag.decompose()

#         title = soup.title.string if soup.title else ""
#         meta_desc = soup.find("meta", attrs={"name": "description"})
#         description = (
#             meta_desc["content"] if meta_desc and "content" in meta_desc.attrs else ""
#         )

#         headings = " ".join(h.get_text(strip=True) for h in soup.find_all(["h1", "h2"]))
#         paragraphs = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
#         list_items = " ".join(li.get_text(strip=True) for li in soup.find_all("li"))

#         content = f"{title}\n{description}\n{headings}\n{paragraphs}\n{list_items}"
#         return content[:5000]
#     except Exception as e:
#         return f"Error scraping {url}: {e}"


if __name__ == "__main__":
    business_type = "Pet grooming services"
    location = "Austin, Texas"

    places = search_places(business_type, location)

    for place in places:
        name = place.get("name")
        place_id = place.get("place_id")
        address = place.get("formatted_address", "N/A")
        website = get_place_details(place_id)

        print(f"\n📍 {name} — {address}")
        if website:
            print(f"🌐 {website}")
            content = scrape_meaningful_content(website)
            print(f"🔍 Sample Content:\n{content[:500]}")
        else:
            print("⚠️ No website found.")
