import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
import re
from typing import List, Dict, Tuple
from datetime import datetime

class FinalGraphRAGClassifier:
    def __init__(self, model_path='./models/distilbert_classifier', 
                 graph_path='./data/graph/enhanced_knowledge_graph.json'):
        # Load BERT model
        print("🤖 Loading BERT model...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        # Load enhanced knowledge graph
        print("🕸️ Loading enhanced knowledge graph...")
        with open(graph_path, 'r') as f:
            self.graph_data = json.load(f)
        
        self.pattern_mappings = self.graph_data['pattern_mappings']
        self.enhanced_patterns = self.graph_data['enhanced_patterns']
        
        # Learning history for continuous improvement
        self.learning_history = []
        
    def _get_bert_prediction(self, email: str) -> Tuple[str, float, Dict]:
        """Get BERT model prediction"""
        prefix = email.split('@')[0]
        inputs = self.tokenizer(prefix, return_tensors="pt", max_length=64, 
                               truncation=True, padding='max_length')
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            
        probs = predictions[0].cpu().numpy()
        category = "HR" if predicted_class == 0 else "Sales"
        confidence = float(probs[predicted_class])
        
        return category, confidence, {
            'HR': float(probs[0]),
            'Sales': float(probs[1])
        }
    
    def _get_enhanced_graph_prediction(self, email: str) -> Tuple[str, float, Dict, List]:
        """Get prediction using enhanced pattern matching"""
        prefix = email.split('@')[0].lower()
        prefix_clean = re.sub(r'[0-9_.-]', '', prefix)
        
        # Find all matching patterns
        patterns_found = []
        for pattern, (category, conf) in self.pattern_mappings.items():
            if pattern in prefix_clean:
                patterns_found.append({
                    'pattern': pattern,
                    'category': category,
                    'confidence': conf,
                    'match_type': 'exact' if pattern == prefix_clean else 'partial'
                })
        
        # Calculate weighted predictions
        weights = {'HR': 0, 'Sales': 0}
        total_confidence = 0
        
        for match in patterns_found:
            if match['category'] == 'Ambiguous':
                # Split ambiguous patterns 50/50
                weights['HR'] += 0.5 * match['confidence']
                weights['Sales'] += 0.5 * match['confidence']
            elif match['category'] in ['HR', 'Sales']:
                # Full weight for clear patterns
                weight_multiplier = 1.2 if match['match_type'] == 'exact' else 1.0
                weights[match['category']] += match['confidence'] * weight_multiplier
            
            total_confidence += match['confidence']
        
        # Normalize and determine prediction
        if total_confidence > 0:
            weights['HR'] /= total_confidence
            weights['Sales'] /= total_confidence
            
            category = 'HR' if weights['HR'] > weights['Sales'] else 'Sales'
            confidence = weights[category]
            
            # Boost confidence for strong patterns
            if len(patterns_found) == 1 and patterns_found[0]['confidence'] > 0.9:
                confidence = min(0.95, confidence + 0.1)
        else:
            category = 'Unknown'
            confidence = 0.0
            weights = {'HR': 0.5, 'Sales': 0.5}
        
        return category, confidence, weights, patterns_found
    
    def classify(self, email: str, explanation=True) -> Dict:
        """Classify email using enhanced GraphRAG approach"""
        # Get predictions from both components
        bert_category, bert_confidence, bert_probs = self._get_bert_prediction(email)
        graph_category, graph_confidence, graph_probs, patterns = self._get_enhanced_graph_prediction(email)
        
        # Smart combination strategy
        if graph_confidence > 0.8:
            # High confidence graph prediction - give it more weight
            weight_bert = 0.4
            weight_graph = 0.6
        elif graph_confidence < 0.3:
            # Low confidence or ambiguous - rely more on BERT
            weight_bert = 0.8
            weight_graph = 0.2
        else:
            # Normal case - balanced weights
            weight_bert = 0.6
            weight_graph = 0.4
        
        # Calculate combined probabilities
        combined_probs = {
            'HR': bert_probs['HR'] * weight_bert + graph_probs['HR'] * weight_graph,
            'Sales': bert_probs['Sales'] * weight_bert + graph_probs['Sales'] * weight_graph
        }
        
        # Final prediction
        final_category = 'HR' if combined_probs['HR'] > combined_probs['Sales'] else 'Sales'
        
        # Confidence calculation with agreement bonus
        base_confidence = combined_probs[final_category]
        agreement = (bert_category == graph_category == final_category)
        
        if agreement and bert_confidence > 0.7 and graph_confidence > 0.7:
            # Boost confidence for strong agreement
            final_confidence = min(0.95, base_confidence + 0.1)
        else:
            final_confidence = base_confidence
        
        result = {
            'email': email,
            'prediction': final_category,
            'confidence': final_confidence,
            'bert': {
                'prediction': bert_category,
                'confidence': bert_confidence,
                'weight_used': weight_bert
            },
            'graph': {
                'prediction': graph_category,
                'confidence': graph_confidence,
                'weight_used': weight_graph,
                'patterns': patterns
            },
            'agreement': agreement,
            'timestamp': datetime.now().isoformat()
        }
        
        if explanation:
            result['explanation'] = self._generate_explanation(result)
        
        return result
    
    def _generate_explanation(self, result: Dict) -> str:
        """Generate detailed explanation"""
        lines = []
        
        # Overall assessment
        conf = result['confidence']
        if conf > 0.85:
            emoji = "✅"
            assessment = "High confidence"
        elif conf > 0.65:
            emoji = "🟡"
            assessment = "Medium confidence"
        else:
            emoji = "🔴"
            assessment = "Low confidence"
        
        lines.append(f"{emoji} {assessment}: {result['prediction']} ({conf:.1%})")
        
        # Component analysis
        lines.append("\n📊 Component Analysis:")
        lines.append(f"   BERT: {result['bert']['prediction']} "
                    f"({result['bert']['confidence']:.1%}) - weight: {result['bert']['weight_used']:.0%}")
        lines.append(f"   Graph: {result['graph']['prediction']} "
                    f"({result['graph']['confidence']:.1%}) - weight: {result['graph']['weight_used']:.0%}")
        
        # Pattern details
        if result['graph']['patterns']:
            lines.append("\n🔍 Pattern Matches:")
            for p in result['graph']['patterns'][:3]:
                lines.append(f"   - '{p['pattern']}' → {p['category']} "
                           f"(confidence: {p['confidence']:.1%})")
        
        # Agreement status
        if result['agreement']:
            lines.append("\n✅ All components agree - high reliability")
        elif result['bert']['prediction'] != result['graph']['prediction']:
            lines.append("\n⚠️  Components disagree - using weighted combination")
        
        return '\n'.join(lines)
    
    def add_feedback(self, email: str, correct_category: str):
        """Add user feedback for continuous learning"""
        self.learning_history.append({
            'email': email,
            'correct_category': correct_category,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save feedback
        feedback_file = './data/graph/feedback_history.json'
        try:
            with open(feedback_file, 'r') as f:
                all_feedback = json.load(f)
        except:
            all_feedback = []
        
        all_feedback.extend(self.learning_history)
        
        with open(feedback_file, 'w') as f:
            json.dump(all_feedback, f, indent=2)
        
        return f"✅ Feedback recorded for {email} → {correct_category}"
    
    def batch_classify(self, emails: List[str]) -> List[Dict]:
        """Classify multiple emails efficiently"""
        results = []
        for email in emails:
            results.append(self.classify(email, explanation=False))
        return results

# Main execution
if __name__ == "__main__":
    print("🚀 Final GraphRAG Email Classifier")
    print("=" * 80)
    
    # Initialize classifier
    classifier = FinalGraphRAGClassifier()
    
    # Test cases
    test_emails = [
        # Clear cases
        ("careers@techcompany.com", "HR"),
        ("sales@company.com", "Sales"),
        ("hiring.manager@startup.io", "HR"),
        ("orders@shop.com", "Sales"),
        
        # Ambiguous cases
        ("info@business.com", None),
        ("contact@company.com", None),
        ("support@company.com", None),
        
        # Mixed signals
        ("talent.sales@company.com", None),
        ("hiring.customers@corp.com", None),
        
        # Edge cases
        ("hr.sales@company.com", None),
        ("sales.careers@firm.com", None),
    ]
    
    print("\n📧 Classification Results:")
    print("-" * 80)
    
    for email, expected in test_emails:
        result = classifier.classify(email)
        
        print(f"\nEmail: {email}")
        if expected:
            print(f"Expected: {expected}")
        print(f"Result: {result['prediction']} ({result['confidence']:.1%})")
        print(result['explanation'])
        print("-" * 80)
    
    # Evaluate on test set
    print("\n📈 Performance Evaluation")
    with open('./data/email_dataset_test.json', 'r') as f:
        test_data = json.load(f)[:50]  # Use 50 samples
    
    # Get predictions
    emails = [item['email'] for item in test_data]
    results = classifier.batch_classify(emails)
    
    # Calculate metrics
    correct = 0
    high_conf_correct = 0
    high_conf_total = 0
    
    for item, result in zip(test_data, results):
        if result['prediction'] == item['category']:
            correct += 1
            if result['confidence'] > 0.8:
                high_conf_correct += 1
        
        if result['confidence'] > 0.8:
            high_conf_total += 1
    
    accuracy = correct / len(test_data)
    high_conf_accuracy = high_conf_correct / high_conf_total if high_conf_total > 0 else 0
    
    print(f"\n📊 Results on {len(test_data)} test samples:")
    print(f"   Overall Accuracy: {accuracy:.1%}")
    print(f"   High-Confidence Predictions: {high_conf_total}/{len(test_data)} ({high_conf_total/len(test_data):.1%})")
    print(f"   High-Confidence Accuracy: {high_conf_accuracy:.1%}")
    
    # Save the classifier info
    print("\n💾 Saving classifier information...")
    classifier_info = {
        'version': '1.0',
        'created': datetime.now().isoformat(),
        'components': {
            'bert_model': './models/distilbert_classifier',
            'knowledge_graph': './data/graph/enhanced_knowledge_graph.json'
        },
        'performance': {
            'test_accuracy': accuracy,
            'high_confidence_ratio': high_conf_total/len(test_data),
            'high_confidence_accuracy': high_conf_accuracy
        }
    }
    
    with open('./data/graphrag_classifier_info.json', 'w') as f:
        json.dump(classifier_info, f, indent=2)
    
    print("✅ Final GraphRAG classifier ready for production!")
