import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

###############################################################################
# Configurable parameters
###############################################################################
CSV_IN = "business_directory.csv"
CSV_OUT = "business_with_keywords.csv"
MODEL = "claude-3-haiku-20240307"  # fast & inexpensive; use Sonnet/Opus if preferred
MAX_SITE_CHARS = 3000  # clip long pages to save tokens + cost
###############################################################################


anthropic_client = Anthropic(api_key="")


def scrape_site(url: str, max_chars: int = MAX_SITE_CHARS) -> str:
    """
    Download `url`, strip HTML, collapse whitespace, return up to `max_chars` chars.
    """
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        text = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text[:max_chars]
    except requests.RequestException:
        return ""  # blank lets the classifier fall back to NO


def classify_catering(page_text: str) -> bool:
    """
    Ask Claude whether the business is a caterer.  Returns True/False.
    """
    if not page_text.strip():
        return False  # empty scrape ⇒ can't decide ⇒ assume not catering

    prompt = (
        f"{HUMAN_PROMPT}You are an expert business analyst. "
        "Given the text of a company's website, answer STRICTLY with YES or NO:\n\n"
        "QUESTION: Does this company provide food *catering* services (event/wedding/banquet, "
        "corporate catering, meal-prep catering, etc.)?\n\n"
        f'WEBSITE TEXT:\n"""\n{page_text}\n"""\n\n'
        f"{AI_PROMPT}"
    )

    try:
        resp = anthropic_client.completions.create(
            model=MODEL,
            prompt=prompt,
            max_tokens_to_sample=1,
            temperature=0,  # deterministic
        )
        answer = resp.completion.strip().upper()
        return answer.startswith("Y")
    except Exception as e:
        print(f"[WARN] Anthropic API call failed: {e}")
        return False


# ────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_IN)

# Add empty columns if they don’t already exist
for col in ("scraped_keywords", "skill"):
    if col not in df.columns:
        df[col] = ""

for idx, row in df.iterrows():
    url = row.get("Company Website") or row.get("Website") or ""
    if not isinstance(url, str) or not url.startswith("http"):
        continue

    site_text = scrape_site(url)
    # quick & dirty keyword list, still useful for your own inspection
    df.at[idx, "scraped_keywords"] = ", ".join(
        re.findall(r"[A-Za-z]{4,}", site_text)[:20]
    )

    if classify_catering(site_text):
        df.at[idx, "skill"] = 11  # catering skill-ID

df.to_csv(CSV_OUT, index=False)
print(f"✅ Done. Updated file written to '{CSV_OUT}'.")
