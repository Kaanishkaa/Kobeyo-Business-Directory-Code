import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

class EmailDataset(Dataset):
    def __init__(self, emails, labels, tokenizer, max_length=64):  # Reduced max_length
        self.emails = emails
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.emails)
    
    def __getitem__(self, idx):
        email = self.emails[idx]
        label = self.labels[idx]
        
        # Extract prefix
        prefix = email.split('@')[0]
        
        encoding = self.tokenizer(
            prefix,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
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
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Load data
print("📚 Loading dataset...")
with open('./data/email_dataset_train.json', 'r') as f:
    train_data = json.load(f)

with open('./data/email_dataset_test.json', 'r') as f:
    test_data = json.load(f)

# Use only a subset for faster training
train_data = train_data[:400]  # Use 400 samples
test_data = test_data[:100]    # Use 100 samples

# Prepare data
train_emails = [item['email'] for item in train_data]
train_labels = [0 if item['category'] == 'HR' else 1 for item in train_data]

test_emails = [item['email'] for item in test_data]
test_labels = [0 if item['category'] == 'HR' else 1 for item in test_data]

print(f"📊 Training samples: {len(train_emails)}")
print(f"📊 Test samples: {len(test_emails)}")

# Initialize tokenizer and model (using DistilBERT - smaller and faster)
print("\n🤖 Loading DistilBERT model (lighter than BERT)...")
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Create datasets
train_dataset = EmailDataset(train_emails, train_labels, tokenizer)
test_dataset = EmailDataset(test_emails, test_labels, tokenizer)

# Training arguments (optimized for low memory)
training_args = TrainingArguments(
    output_dir='./models/distilbert_classifier',
    num_train_epochs=2,  # Fewer epochs
    per_device_train_batch_size=8,  # Smaller batch size
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=2,  # Accumulate gradients
    fp16=False,  # Disable mixed precision
    dataloader_num_workers=0,  # Disable multiprocessing
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
print("\n🚀 Starting training (lightweight version)...")
print("This will take about 3-5 minutes...")
trainer.train()

# Evaluate
print("\n📈 Evaluating model...")
results = trainer.evaluate()

print("\n✅ Training complete!")
print(f"   Accuracy: {results['eval_accuracy']:.3f}")
print(f"   F1 Score: {results['eval_f1']:.3f}")

# Save model
print("\n💾 Saving model...")
model.save_pretrained('./models/distilbert_classifier')
tokenizer.save_pretrained('./models/distilbert_classifier')

# Test with a few examples
print("\n🧪 Testing with examples:")
test_examples = [
    "careers@company.com",
    "sales@company.com",
    "hiring@startup.io",
    "orders@shop.com",
    "info@business.com"
]

model.eval()
for email in test_examples:
    prefix = email.split('@')[0]
    inputs = tokenizer(prefix, return_tensors="pt", max_length=64, truncation=True, padding='max_length')
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    category = "HR" if predicted_class == 0 else "Sales"
    print(f"  {email} -> {category} (confidence: {confidence:.2f})")

print("\n🎉 Done! Model saved to ./models/distilbert_classifier")