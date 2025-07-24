import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class CareerPageDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract features from URL
        url = item['url']
        pattern = item['pattern']
        title = item.get('page_title', '')
        description = item.get('meta_description', '')
        
        # Combine URL path, title, and description for richer context
        # Extract path from URL
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path.strip('/').replace('/', ' ').replace('-', ' ').replace('_', ' ')
        
        # Create input text with all available information
        input_text = f"URL: {path} Title: {title} Description: {description}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Label: 1 for career page, 0 for non-career
        label = 1 if item['is_career_page'] else 0
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    
    # Also calculate per-class metrics
    career_precision, non_career_precision = precision_recall_fscore_support(
        labels, predictions, average=None, labels=[1, 0]
    )[0]
    career_recall, non_career_recall = precision_recall_fscore_support(
        labels, predictions, average=None, labels=[1, 0]
    )[1]
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'career_precision': career_precision,
        'career_recall': career_recall,
        'non_career_precision': non_career_precision,
        'non_career_recall': non_career_recall
    }

# Load data
print("📚 Loading career page dataset...")
with open('./data/career_pages_train.json', 'r') as f:
    train_data = json.load(f)

with open('./data/career_pages_test.json', 'r') as f:
    test_data = json.load(f)

print(f"📊 Training samples: {len(train_data)}")
print(f"📊 Test samples: {len(test_data)}")

# Count distribution
train_career = sum(1 for d in train_data if d['is_career_page'])
train_non_career = len(train_data) - train_career
print(f"   Career pages: {train_career} ({train_career/len(train_data)*100:.1f}%)")
print(f"   Non-career pages: {train_non_career} ({train_non_career/len(train_data)*100:.1f}%)")

# Initialize tokenizer and model
print("\n🤖 Loading DistilBERT model...")
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Create datasets
train_dataset = CareerPageDataset(train_data, tokenizer)
test_dataset = CareerPageDataset(test_data, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./models/career_classifier',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs/career',
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=False,
    dataloader_num_workers=0,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train
print("\n🚀 Starting training...")
print("This will take about 5-10 minutes...")
trainer.train()

# Evaluate
print("\n📈 Evaluating model...")
results = trainer.evaluate()

print("\n✅ Training complete!")
print(f"   Accuracy: {results['eval_accuracy']:.3f}")
print(f"   F1 Score: {results['eval_f1']:.3f}")
print(f"\n📊 Per-class Performance:")
print(f"   Career Pages:")
print(f"     Precision: {results['eval_career_precision']:.3f}")
print(f"     Recall: {results['eval_career_recall']:.3f}")
print(f"   Non-Career Pages:")
print(f"     Precision: {results['eval_non_career_precision']:.3f}")
print(f"     Recall: {results['eval_non_career_recall']:.3f}")

# Save model
print("\n💾 Saving model...")
model.save_pretrained('./models/career_classifier')
tokenizer.save_pretrained('./models/career_classifier')

# Test with examples
print("\n🧪 Testing with examples:")
test_urls = [
    "https://www.company.com/careers",
    "https://www.company.com/jobs",
    "https://www.company.com/about",
    "https://www.company.com/contact",
    "https://www.company.com/work-with-us",
    "https://careers.company.com",
    "https://www.company.com/team",
    "https://www.company.com/join-our-team",
]

model.eval()
for url in test_urls:
    # Extract path
    from urllib.parse import urlparse
    parsed = urlparse(url)
    path = parsed.path.strip('/').replace('/', ' ').replace('-', ' ')
    
    # Simple title/description for testing
    input_text = f"URL: {path} Title: Page Title Description: Page description"
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True, padding='max_length')
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        is_career = predictions[0][1].item() > predictions[0][0].item()
        confidence = max(predictions[0][0].item(), predictions[0][1].item())
    
    result = "Career Page" if is_career else "Not Career"
    emoji = "✅" if is_career else "❌"
    print(f"  {emoji} {url} -> {result} (confidence: {confidence:.2%})")

print("\n🎉 Career page classifier ready!")
print(f"Model saved to: ./models/career_classifier")
