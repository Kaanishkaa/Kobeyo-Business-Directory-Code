import pandas as pd
from ollama import Client

# Load your CSV
df = pd.read_csv("business_with_descriptions.csv")

# Connect to the local Ollama server
ollama_client = Client(host="http://localhost:11434")


# Define the classification function
def classify_business(description):
    prompt = f"""
You are an expert assistant for classifying businesses.
Given the following business description, determine if it suggests a catering business.

If the description clearly indicates catering services (like food preparation, event catering, large meal service, etc.), respond with "11".
If not, respond with "null".

Business Description:
\"\"\"{description}\"\"\"

Respond with only the skill ID or "null".
"""
    try:
        response = ollama_client.chat(
            model="llama3", messages=[{"role": "user", "content": prompt}]
        )
        result = response["message"]["content"].strip()
        return result if result == "11" else None
    except Exception as e:
        return f"Error: {str(e)}"


# Apply the function
df["skill_id"] = df["description"].apply(classify_business)

# Save the result
df.to_csv("business_with_skills.csv", index=False)
print("Skill tagging complete. Results saved to 'business_with_skills.csv'")
