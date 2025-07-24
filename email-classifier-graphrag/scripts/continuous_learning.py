import json
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datetime import datetime
import os
from collections import defaultdict
import shutil
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ContinuousLearningSystem:
    def __init__(self, 
                 model_path='./models/distilbert_classifier',
                 feedback_path='./data/graph/feedback_history.json',
                 retrain_threshold=50):  # Retrain after 50 feedback samples
        
        self.model_path = model_path
        self.feedback_path = feedback_path
        self.retrain_threshold = retrain_threshold
        self.models_history_path = './models/history'
        
        # Create history directory
        os.makedirs(self.models_history_path, exist_ok=True)
        
    def check_feedback_count(self):
        """Check how many feedback samples we have"""
        try:
            with open(self.feedback_path, 'r') as f:
                feedback = json.load(f)
            return len(feedback)
        except:
            return 0
    
    def prepare_training_data(self):
        """Prepare training data combining original + feedback"""
        print("📚 Preparing training data...")
        
        # Load original training data
        with open('./data/email_dataset_train.json', 'r') as f:
            original_data = json.load(f)
        
        # Load feedback data
        try:
            with open(self.feedback_path, 'r') as f:
                feedback_data = json.load(f)
        except:
            feedback_data = []
        
        # Convert feedback to training format
        feedback_formatted = []
        for item in feedback_data:
            feedback_formatted.append({
                'email': item['email'],
                'category': item['correct_category'],
                'confidence': 1.0,  # High confidence for human corrections
                'source': 'feedback'
            })
        
        # Combine data
        combined_data = original_data + feedback_formatted
        
        # Remove duplicates (keep feedback version)
        seen_emails = {}
        final_data = []
        
        # First add feedback data (higher priority)
        for item in feedback_formatted:
            email_key = item['email'].lower()
            if email_key not in seen_emails:
                seen_emails[email_key] = True
                final_data.append(item)
        
        # Then add original data
        for item in original_data:
            email_key = item['email'].lower()
            if email_key not in seen_emails:
                seen_emails[email_key] = True
                final_data.append(item)
        
        print(f"✅ Total training samples: {len(final_data)}")
        print(f"   - Original: {len(original_data)}")
        print(f"   - From feedback: {len(feedback_formatted)}")
        print(f"   - After deduplication: {len(final_data)}")
        
        return final_data
    
    def update_knowledge_graph(self, training_data):
        """Update the knowledge graph with new patterns"""
        print("\n🕸️ Updating knowledge graph...")
        
        # Load existing graph
        with open('./data/graph/enhanced_knowledge_graph.json', 'r') as f:
            graph = json.load(f)
        
        # Update pattern statistics
        pattern_updates = defaultdict(lambda: {'HR': 0, 'Sales': 0, 'total': 0})
        
        for item in training_data:
            if item.get('source') == 'feedback':
                email = item['email']
                category = item['category']
                prefix = email.split('@')[0].lower()
                
                # Clean prefix
                import re
                prefix_clean = re.sub(r'[0-9_.-]', '', prefix)
                
                # Update pattern counts
                for pattern in graph['pattern_mappings'].keys():
                    if pattern in prefix_clean:
                        pattern_updates[pattern][category] += 1
                        pattern_updates[pattern]['total'] += 1
        
        # Update graph with new statistics
        updates_made = 0
        for pattern, updates in pattern_updates.items():
            if updates['total'] >= 3:  # Need at least 3 examples to update
                hr_ratio = updates['HR'] / updates['total']
                sales_ratio = updates['Sales'] / updates['total']
                
                # Update pattern mapping if clear majority
                if hr_ratio > 0.7:
                    old_category = graph['pattern_mappings'].get(pattern, ('Unknown', 0))[0]
                    if old_category != 'HR':
                        print(f"   Updating '{pattern}': {old_category} -> HR (based on {updates['total']} feedback)")
                        graph['pattern_mappings'][pattern] = ('HR', min(0.9, hr_ratio))
                        updates_made += 1
                elif sales_ratio > 0.7:
                    old_category = graph['pattern_mappings'].get(pattern, ('Unknown', 0))[0]
                    if old_category != 'Sales':
                        print(f"   Updating '{pattern}': {old_category} -> Sales (based on {updates['total']} feedback)")
                        graph['pattern_mappings'][pattern] = ('Sales', min(0.9, sales_ratio))
                        updates_made += 1
        
        # Save updated graph
        if updates_made > 0:
            graph['last_updated'] = datetime.now().isoformat()
            graph['feedback_incorporated'] = len([d for d in training_data if d.get('source') == 'feedback'])
            
            with open('./data/graph/enhanced_knowledge_graph.json', 'w') as f:
                json.dump(graph, f, indent=2)
            
            print(f"✅ Graph updated with {updates_made} pattern changes")
        else:
            print("ℹ️  No significant pattern changes detected")
        
        return updates_made
    
    def retrain_model(self, force=False):
        """Retrain the model with feedback data"""
        feedback_count = self.check_feedback_count()
        
        if not force and feedback_count < self.retrain_threshold:
            print(f"ℹ️  Not enough feedback yet ({feedback_count}/{self.retrain_threshold})")
            return False
        
        print(f"\n🚀 Starting model retraining with {feedback_count} feedback samples...")
        
        # Backup current model
        backup_name = f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = os.path.join(self.models_history_path, backup_name)
        shutil.copytree(self.model_path, backup_path)
        print(f"📦 Current model backed up to: {backup_path}")
        
        # Prepare training data
        training_data = self.prepare_training_data()
        
        # Update knowledge graph
        self.update_knowledge_graph(training_data)
        
        # Create dataset class
        class EmailDataset(Dataset):
            def __init__(self, data, tokenizer, max_length=64):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                email = item['email']
                label = 0 if item['category'] == 'HR' else 1
                
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
        
        # Load tokenizer and model
        print("\n🤖 Loading model for retraining...")
        tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
        model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_data, val_data = train_test_split(training_data, test_size=0.1, random_state=42)
        
        # Create datasets
        train_dataset = EmailDataset(train_data, tokenizer)
        val_dataset = EmailDataset(val_data, tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./models/retrained_temp',
            num_train_epochs=2,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs/retraining',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
        )
        
        # Define metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted'
            )
            acc = accuracy_score(labels, predictions)
            
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        
        # Train
        print("\n📈 Training model...")
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        print(f"\n✅ Retraining complete!")
        print(f"   Accuracy: {eval_results['eval_accuracy']:.3f}")
        print(f"   F1 Score: {eval_results['eval_f1']:.3f}")
        
        # Save retrained model
        model.save_pretrained(self.model_path)
        tokenizer.save_pretrained(self.model_path)
        
        # Clean up temp directory
        shutil.rmtree('./models/retrained_temp', ignore_errors=True)
        
        # Archive processed feedback
        archive_path = f"./data/graph/feedback_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        shutil.move(self.feedback_path, archive_path)
        
        # Create new empty feedback file
        with open(self.feedback_path, 'w') as f:
            json.dump([], f)
        
        print(f"\n🎉 Model successfully retrained and deployed!")
        print(f"   Feedback archived to: {archive_path}")
        print(f"   Model backup saved to: {backup_path}")
        
        return True
    
    def get_learning_status(self):
        """Get current learning system status"""
        feedback_count = self.check_feedback_count()
        
        # Count model versions
        model_versions = len([d for d in os.listdir(self.models_history_path) 
                             if os.path.isdir(os.path.join(self.models_history_path, d))])
        
        # Load graph info
        try:
            with open('./data/graph/enhanced_knowledge_graph.json', 'r') as f:
                graph = json.load(f)
            last_updated = graph.get('last_updated', 'Never')
            feedback_incorporated = graph.get('feedback_incorporated', 0)
        except:
            last_updated = 'Never'
            feedback_incorporated = 0
        
        return {
            'current_feedback_count': feedback_count,
            'retrain_threshold': self.retrain_threshold,
            'progress_to_retrain': f"{feedback_count}/{self.retrain_threshold} ({feedback_count/self.retrain_threshold*100:.1f}%)",
            'model_versions': model_versions,
            'graph_last_updated': last_updated,
            'total_feedback_incorporated': feedback_incorporated,
            'ready_to_retrain': feedback_count >= self.retrain_threshold
        }

# Create monitoring script
if __name__ == "__main__":
    import sys
    
    learning_system = ContinuousLearningSystem()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        # Show status
        status = learning_system.get_learning_status()
        print("\n📊 Continuous Learning System Status")
        print("=" * 50)
        for key, value in status.items():
            print(f"{key}: {value}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == 'retrain':
        # Force retrain
        learning_system.retrain_model(force=True)
    
    else:
        # Check if ready to retrain
        status = learning_system.get_learning_status()
        print(f"\n🔄 Continuous Learning System")
        print(f"   Current feedback: {status['current_feedback_count']}")
        print(f"   Threshold: {status['retrain_threshold']}")
        
        if status['ready_to_retrain']:
            print("\n✅ Ready to retrain! Starting automatic retraining...")
            learning_system.retrain_model()
        else:
            print(f"\n⏳ Need {status['retrain_threshold'] - status['current_feedback_count']} more feedback samples before retraining")
