import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
import re

df = pd.read_csv("business_directory.csv")


def extract_keywords_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        texts = soup.get_text(separator=" ", strip=True)
        # Clean text
        texts = re.sub(r"\s+", " ", texts)
        vectorizer = CountVectorizer(max_features=20, stop_words="english")
        X = vectorizer.fit_transform([texts])
        return vectorizer.get_feature_names_out().tolist()
    except:
        return []


catering_keywords = [
    "catering",
    "wedding",
    "banquet",
    "event",
    "buffet",
    "party",
    "chef",
    "menu",
    "food service",
]

df["scraped_keywords"] = ""
df["skill"] = ""

for idx, row in df.iterrows():
    url = row.get("Company Website") or row.get(
        "Website"
    )  # adapt if column name differs
    if pd.isna(url) or not isinstance(url, str) or not url.startswith("http"):
        continue
    keywords = extract_keywords_from_url(url)
    df.at[idx, "scraped_keywords"] = ", ".join(keywords)

    # Classify as catering-related
    if any(kw in keywords for kw in catering_keywords):
        df.at[idx, "skill"] = 11

# Save updated CSV
df.to_csv("business_with_keywords.csv", index=False)
print("Done. File saved as 'business_with_keywords.csv'")
