import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json

# Load model and tokenizer
print("🤖 Loading trained model...")
model_path = './models/distilbert_classifier'
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

def classify_email(email):
    """Classify a single email"""
    prefix = email.split('@')[0]
    inputs = tokenizer(prefix, return_tensors="pt", max_length=64, truncation=True, padding='max_length')
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    category = "HR" if predicted_class == 0 else "Sales"
    return category, confidence

# Test with various examples
print("\n📧 Testing Email Classifier")
print("=" * 50)

test_emails = [
    # Clear HR examples
    "careers@techcompany.com",
    "hiring.manager@startup.io",
    "recruitment@corp.com",
    "hr.department@business.com",
    "talent.acquisition@firm.com",
    
    # Clear Sales examples
    "sales.team@company.com",
    "orders@shop.com",
    "business.development@corp.com",
    "account.manager@business.com",
    "revenue@startup.com",
    
    # Ambiguous examples
    "info@company.com",
    "contact@business.com",
    "admin@corp.com",
    "support@company.com",
    "hello@startup.com",
    
    # Edge cases
    "careers.sales@company.com",  # Mixed signals
    "hiring.customers@corp.com",   # Mixed signals
    "talent.revenue@business.com", # Mixed signals
]

# Classify and display results
hr_count = 0
sales_count = 0

for email in test_emails:
    category, confidence = classify_email(email)
    
    # Emoji based on confidence
    if confidence > 0.8:
        emoji = "✅"
    elif confidence > 0.6:
        emoji = "🟡"
    else:
        emoji = "🔴"
    
    print(f"{emoji} {email:<35} -> {category:<5} ({confidence:.2%})")
    
    if category == "HR":
        hr_count += 1
    else:
        sales_count += 1

print("\n📊 Summary:")
print(f"   HR emails: {hr_count}")
print(f"   Sales emails: {sales_count}")

# Load test dataset and check accuracy
print("\n📈 Checking accuracy on test set sample...")
with open('./data/email_dataset_test.json', 'r') as f:
    test_data = json.load(f)[:20]  # Check first 20

correct = 0
for item in test_data:
    predicted_category, _ = classify_email(item['email'])
    if predicted_category == item['category']:
        correct += 1

accuracy = correct / len(test_data)
print(f"   Sample accuracy: {accuracy:.2%} ({correct}/{len(test_data)} correct)")
