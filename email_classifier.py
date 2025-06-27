import pandas as pd, re, ollama

df = pd.read_csv("business_directory_old.csv")


def clean_emails(text):
    if pd.isna(text):
        return []
    return re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)


def pick_email(row):
    raw = (
        clean_emails(row.get("Best Email", ""))
        + clean_emails(row.get("Better Email", ""))
        + clean_emails(row.get("HR Email", ""))
    )
    if not raw:
        return ""  # nothing to grade

    prompt = f"""Company: "{row['Company Name']}"\nIndustry: restaurant/catering
    Emails: {', '.join(raw)}\nChoose exactly one BEST address ..."""
    choice = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    # return choice["message"].split()[0]  # first token is the e-mail
    return choice["message"]["content"].split()[0]


df["Chosen Email"] = df.apply(pick_email, axis=1)
