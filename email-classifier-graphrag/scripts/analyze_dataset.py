import json
import pandas as pd
from collections import Counter

# Load the dataset
with open('./data/email_dataset_full.json', 'r') as f:
    data = json.load(f)

print("📊 Dataset Analysis")
print("=" * 50)

# Basic stats
df = pd.DataFrame(data)
print(f"\nTotal emails: {len(df)}")
print(f"HR emails: {len(df[df['category'] == 'HR'])}")
print(f"Sales emails: {len(df[df['category'] == 'Sales'])}")

# Confidence distribution
print(f"\n🎯 Confidence Distribution:")
print(f"High confidence (>0.8): {len(df[df['confidence'] > 0.8])}")
print(f"Medium confidence (0.6-0.8): {len(df[(df['confidence'] >= 0.6) & (df['confidence'] <= 0.8)])}")
print(f"Low confidence (<0.6): {len(df[df['confidence'] < 0.6])}")

# Top patterns
print(f"\n📧 Top 10 Email Patterns:")
pattern_counts = Counter(df['prefix_pattern'].values)
for pattern, count in pattern_counts.most_common(10):
    category = df[df['prefix_pattern'] == pattern]['category'].iloc[0]
    print(f"  {pattern}: {count} emails ({category})")

# Domain distribution
print(f"\n🌐 Top 5 Domains:")
domain_counts = Counter(df['domain'].values)
for domain, count in domain_counts.most_common(5):
    print(f"  {domain}: {count} emails")
