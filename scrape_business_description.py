import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

input_csv = "business_directory.csv"
output_csv = "business_with_descriptions.csv"

df = pd.read_csv(input_csv)

df.rename(columns={"Company Website": "website"}, inplace=True)


def scrape_description(url, max_chars=1000):
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text[:max_chars]
    except Exception as e:
        return f"Error: {str(e)}"


df["description"] = df["website"].apply(scrape_description)

df.to_csv(output_csv, index=False)

print(f"Descriptions added and saved to: {output_csv}")
