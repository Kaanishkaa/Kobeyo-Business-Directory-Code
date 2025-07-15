import requests
from bs4 import BeautifulSoup
from typing import Dict

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def fetch_html(url: str, timeout: int = 10) -> str:
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"⚠️ Error scraping {url}: {e}")
        return ""


def extract_website_content(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.string.strip() if soup.title else ""
    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    meta_desc = (
        meta_desc_tag["content"].strip()
        if meta_desc_tag and meta_desc_tag.get("content")
        else ""
    )

    # Get first few useful headings
    headings = [tag.get_text(strip=True) for tag in soup.find_all(["h1", "h2", "h3"])]
    headings = list(dict.fromkeys(headings))[:5]  # remove duplicates, keep top 5

    # Get some visible paragraph text
    paragraphs = [
        p.get_text(strip=True)
        for p in soup.find_all("p")
        if len(p.get_text(strip=True)) > 40
    ]
    paragraphs = paragraphs[:5]

    return {
        "title": title,
        "meta_description": meta_desc,
        "headings": headings,
        "paragraphs": paragraphs,
    }


def summarize_context(context: Dict[str, str]) -> str:
    lines = []
    if context["title"]:
        lines.append(f"Title: {context['title']}")
    if context["meta_description"]:
        lines.append(f"Meta Description: {context['meta_description']}")
    if context["headings"]:
        lines.append("Headings:\n- " + "\n- ".join(context["headings"]))
    if context["paragraphs"]:
        lines.append("Paragraphs:\n- " + "\n- ".join(context["paragraphs"]))

    return "\n\n".join(lines) if lines else "No meaningful content found."


# Example usage:
# html = fetch_html("https://example.com")
# context = extract_website_context(html)
# print(summarize_context(context))
