# import pandas as pd
# import re
# from ollama import Client

# ollama_client = Client(host="http://localhost:11434")

# df = pd.read_csv("business_directory.csv")

# email_pattern = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")


# def extract_emails(text):
#     if not isinstance(text, str):
#         return []
#     return email_pattern.findall(text)


# # Build prompt and query Ollama
# def choose_best_email(row):
#     all_text = " ".join(str(x) for x in row.values if isinstance(x, str))
#     emails = list(set(extract_emails(all_text)))
#     if not emails:
#         return ""

#     prompt = f"""
# You are an expert assistant helping a sales team identify the best contact email.

# Here is a business:
# Company: {row.get('Company Name', '')}
# Address: {row.get('Company Address', '')}

# Here are some email addresses found:
# {chr(10).join(emails)}

# Which one is the best email address to contact for a request for proposal or sales inquiry (such as for catering, gym partnerships, or construction services)?
# Reply with just one email. If none are suitable, reply with just: NONE.
# """
#     try:
#         response = ollama_client.chat(
#             model="llama3", messages=[{"role": "user", "content": prompt}]
#         )
#         result = response["message"]["content"].strip()
#         if "@" in result:
#             return result.split()[0]
#     except Exception as e:
#         print(f"Error: {e}")
#     return ""


# # Apply to each row
# df["Chosen Email"] = df.apply(choose_best_email, axis=1)

# # Save or display
# df[["Company Name", "Company Address", "Chosen Email"]].to_csv(
#     "ranked_emails_output.csv", index=False
# )
# print(" Done. Results saved to 'ranked_emails_output.csv'")


import pandas as pd
import re
from ollama import Client

# Initialize Ollama
ollama_client = Client(host="http://localhost:11434")

# Load your CSV
df = pd.read_csv("emails_summary.csv")

# Regex pattern to extract emails
email_pattern = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")

# Classification rules
BEST_PREFIXES = ["sales@", "orders@"]
BETTER_PREFIXES = [
    "info@",
    "contact@",
    "contactus@",
    "hello@",
    "admin@",
    "mail@",
    "support@",
    "hola@",
]
EXCLUDE_PREFIXES = ["invoices@", "billing@", "guestservices@", "estimates@"]


# Classify email by its prefix
def classify_email(email):
    email = email.lower()
    if any(email.startswith(p) for p in EXCLUDE_PREFIXES):
        return "EXCLUDE"
    elif any(email.startswith(p) for p in BEST_PREFIXES):
        return "BEST"
    elif any(email.startswith(p) for p in BETTER_PREFIXES):
        return "BETTER"
    elif re.match(r"[a-z]+\.[a-z]+@", email):  # likely a personal email
        return "EXCLUDE"
    return "EXCLUDE"


def choose_best_email(row):
    # Get all text and extract emails
    all_text = " ".join(str(x) for x in row.values if isinstance(x, str))
    emails = list(set(email_pattern.findall(all_text)))
    if not emails:
        return ""

    # Classify each email
    classified = {"BEST": [], "BETTER": []}
    for email in emails:
        label = classify_email(email)
        if label in classified:
            classified[label].append(email)

    # Priority: BEST > BETTER
    candidates = classified["BEST"] or classified["BETTER"]
    if not candidates:
        return ""

    # If only one candidate, return directly
    if len(candidates) == 1:
        return candidates[0]

    # Else, use Ollama to choose the best
    prompt = f"""
You are an expert assistant helping pick the most appropriate business contact email for RFP or sales purposes.

Company: {row.get('company_name', '')}
website: {row.get('company_website', '')}
Emails:
{chr(10).join(candidates)}

From these, choose the best one to send a proposal or sales message.
Reply with only one email address. If none are appropriate, reply: NONE.
"""
    try:
        response = ollama_client.chat(
            model="llama3", messages=[{"role": "user", "content": prompt}]
        )
        result = response["message"]["content"].strip()
        extracted = email_pattern.findall(result)
        return extracted[0] if extracted else ""
    except Exception as e:
        print(f"Error: {e}")
        return ""


BEST_PREFIXES_HR = [
    "joinus@",
    "hr@",
    "ta@",
    "talentaquisition@",
    "humanresources@",
    "apply@",
    "jobs@",
    "info@",
    "careers@",
    "hiring@",
    "recruiting@",
    "recruitment@",
    "talent@",
    "talentteam@",
    "people@",
    "peopleops@",
    "applications@",
    "submit@",
    "cv@",
    "resume@",
    "workwithus@",
    "jobshr@",
    "hrteam@",
    "recruiters@",
    "talentmgmt@",
    "hiringteam@",
    "teamhr@",
    "opportunities@",
    "team@",
    "staffing@",
    "onboarding@",
]
BETTER_PREFIXES_HR = [
    "contact@",
    "contactus@",
    "hello@",
    "admin@",
    "mail@",
    "store",
    "clinic",
    "hola@",
    "office@",
    "company_name@",
]
EXCLUDE_PREFIXES_HR = [
    "support@",
    "invoices@",
    "billing@",
    "guestservices@",
    "estimates@",
    "sales@",
    "orders@",
    "customerservice@",
    "press@",
    "accessibility@",
    "media@",
    "feedback@",
    "reservations@",
]


def classify_email_hr(email):
    email = email.lower()
    if any(email.startswith(p) for p in EXCLUDE_PREFIXES_HR):
        return "EXCLUDE"
    elif any(email.startswith(p) for p in BEST_PREFIXES_HR):
        return "BEST"
    elif any(email.startswith(p) for p in BETTER_PREFIXES_HR):
        return "BETTER"
    elif re.match(r"[a-z]+\.[a-z]+@", email):  # likely a personal email
        return "EXCLUDE"
    return "EXCLUDE"


def choose_best_email_hr(row):
    # Get all text and extract emails
    all_text = " ".join(str(x) for x in row.values if isinstance(x, str))
    emails = list(set(email_pattern.findall(all_text)))
    if not emails:
        return ""

    # Classify each email
    classified = {"BEST": [], "BETTER": []}
    for email in emails:
        label = classify_email_hr(email)
        if label in classified:
            classified[label].append(email)

    # Priority: BEST > BETTER
    candidates = classified["BEST"] or classified["BETTER"]
    if not candidates:
        return ""

    # If only one candidate, return directly
    if len(candidates) == 1:
        return candidates[0]

    # Else, use Ollama to choose the best
    prompt = f"""
You are an expert assistant helping pick the most appropriate business contact email for RFP or sales purposes.

Company: {row.get('company_name', '')}
website: {row.get('company_website', '')}
Emails:
{chr(10).join(candidates)}

From these, choose the best one to send a proposal or sales message.
Reply with only one email address. If none are appropriate, reply: NONE.
"""
    try:
        response = ollama_client.chat(
            model="llama3", messages=[{"role": "user", "content": prompt}]
        )
        result = response["message"]["content"].strip()
        extracted = email_pattern.findall(result)
        return extracted[0] if extracted else ""
    except Exception as e:
        print(f"Error: {e}")
        return ""


# Apply logic to each row
df["Chosen Email"] = df.apply(choose_best_email, axis=1)
df["Chosen Email HR"] = df.apply(choose_best_email_hr, axis=1)

# Save output
df[["company_name", "company_website", "Chosen Email", "Chosen Email HR"]].to_csv(
    "ranked_emails_output_emails_summary_new.csv", index=False
)
print("Done. Results saved to 'ranked_emails_output_emails_new.csv'")
