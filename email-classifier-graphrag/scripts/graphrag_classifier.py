
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
import re
from typing import List, Dict, Tuple

class GraphRAGEmailClassifier:
    def __init__(self, model_path='./models/distilbert_classifier', graph_path='./data/graph/email_knowledge_graph.json'):
        # Load BERT model
        print("🤖 Loading BERT model...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        # Load knowledge graph
        print("🕸️ Loading knowledge graph...")
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
        
        self.nodes = graph_data['nodes']
        self.patterns = graph_data['patterns']
        
        # Build pattern statistics
        self._build_pattern_stats()
        
    def _build_pattern_stats(self):
        """Calculate pattern category probabilities"""
        self.pattern_probs = {}
        
        for pattern, emails in self.patterns.items():
            category_counts = {'HR': 0, 'Sales': 0}
            total_confidence = {'HR': 0, 'Sales': 0}
            
            for email in emails:
                if email in self.nodes:
                    node = self.nodes[email]
                    category = node['category']
                    category_counts[category] += 1
                    total_confidence[category] += node['confidence']
            
            total = sum(category_counts.values())
            if total > 0:
                self.pattern_probs[pattern] = {
                    'HR': category_counts['HR'] / total,
                    'Sales': category_counts['Sales'] / total,
                    'count': total,
                    'avg_confidence': {
                        'HR': total_confidence['HR'] / category_counts['HR'] if category_counts['HR'] > 0 else 0,
                        'Sales': total_confidence['Sales'] / category_counts['Sales'] if category_counts['Sales'] > 0 else 0
                    }
                }
    
    def _get_bert_prediction(self, email: str) -> Tuple[str, float, Dict]:
        """Get BERT model prediction"""
        prefix = email.split('@')[0]
        inputs = self.tokenizer(prefix, return_tensors="pt", max_length=64, truncation=True, padding='max_length')
        
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
    
    def _get_graph_prediction(self, email: str) -> Tuple[str, float, Dict, List]:
        """Get prediction based on knowledge graph"""
        prefix = email.split('@')[0].lower()
        prefix_clean = re.sub(r'[0-9_.-]', '', prefix)
        
        # Find matching patterns
        matches = []
        graph_probs = {'HR': 0, 'Sales': 0}
        total_weight = 0
        
        for pattern, stats in self.pattern_probs.items():
            if pattern in prefix_clean or prefix_clean in pattern:
                # Calculate match strength
                if pattern == prefix_clean:
                    weight = 1.0
                elif pattern in prefix_clean:
                    weight = 0.8
                else:
                    weight = 0.6
                
                matches.append({
                    'pattern': pattern,
                    'weight': weight,
                    'stats': stats
                })
                
                # Update probabilities
                graph_probs['HR'] += stats['HR'] * weight
                graph_probs['Sales'] += stats['Sales'] * weight
                total_weight += weight
        
        # Normalize probabilities
        if total_weight > 0:
            graph_probs['HR'] /= total_weight
            graph_probs['Sales'] /= total_weight
            
            category = 'HR' if graph_probs['HR'] > graph_probs['Sales'] else 'Sales'
            confidence = graph_probs[category]
        else:
            # No pattern match
            category = 'Unknown'
            confidence = 0.0
        
        return category, confidence, graph_probs, matches
    
    def classify(self, email: str) -> Dict:
        """Classify email using GraphRAG approach"""
        # Get BERT prediction
        bert_category, bert_confidence, bert_probs = self._get_bert_prediction(email)
        
        # Get Graph prediction
        graph_category, graph_confidence, graph_probs, pattern_matches = self._get_graph_prediction(email)
        
        # Combine predictions
        if graph_confidence > 0:
            # Weighted combination (BERT 60%, Graph 40%)
            combined_probs = {
                'HR': bert_probs['HR'] * 0.6 + graph_probs['HR'] * 0.4,
                'Sales': bert_probs['Sales'] * 0.6 + graph_probs['Sales'] * 0.4
            }
        else:
            # No graph info, use BERT only
            combined_probs = bert_probs
        
        # Final prediction
        final_category = 'HR' if combined_probs['HR'] > combined_probs['Sales'] else 'Sales'
        final_confidence = combined_probs[final_category]
        
        # Explanation
        explanation = self._generate_explanation(
            email, final_category, final_confidence,
            bert_category, bert_confidence,
            graph_category, graph_confidence,
            pattern_matches
        )
        
        return {
            'email': email,
            'prediction': final_category,
            'confidence': final_confidence,
            'bert_prediction': bert_category,
            'bert_confidence': bert_confidence,
            'graph_prediction': graph_category,
            'graph_confidence': graph_confidence,
            'combined_probs': combined_probs,
            'pattern_matches': pattern_matches,
            'explanation': explanation
        }
    
    def _generate_explanation(self, email, final_cat, final_conf, bert_cat, bert_conf, 
                            graph_cat, graph_conf, matches):
        """Generate human-readable explanation"""
        explanation = []
        
        if final_conf > 0.8:
            explanation.append(f"✅ High confidence: {email} is clearly {final_cat}")
        elif final_conf > 0.6:
            explanation.append(f"🟡 Medium confidence: {email} is likely {final_cat}")
        else:
            explanation.append(f"🔴 Low confidence: {email} might be {final_cat}")
        
        explanation.append(f"\n📊 Analysis:")
        explanation.append(f"   - BERT model: {bert_cat} ({bert_conf:.2%})")
        
        if graph_conf > 0:
            explanation.append(f"   - Graph analysis: {graph_cat} ({graph_conf:.2%})")
            if matches:
                explanation.append(f"   - Matched patterns: {', '.join([m['pattern'] for m in matches[:3]])}")
        else:
            explanation.append(f"   - Graph analysis: No pattern matches found")
        
        if bert_cat != graph_cat and graph_conf > 0:
            explanation.append(f"\n⚠️ Note: BERT and Graph disagree. Combined analysis favors {final_cat}.")
        
        return '\n'.join(explanation)

# Test the GraphRAG classifier
if __name__ == "__main__":
    print("🚀 Initializing GraphRAG Email Classifier...")
    classifier = GraphRAGEmailClassifier()
    
    # Test emails
    test_emails = [
        "careers@techcompany.com",
        "sales@company.com",
        "hiring@startup.io",
        "orders@shop.com",
        "info@business.com",
        "talent.acquisition@firm.com",
        "business.development@corp.com",
        "support@company.com",
        "hello@startup.com",
        "recruitment.sales@company.com",  # Ambiguous
    ]
    
    print("\n📧 Testing GraphRAG Classifier")
    print("=" * 80)
    
    for email in test_emails:
        result = classifier.classify(email)
        print(f"\nEmail: {email}")
        print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")
        print(f"BERT: {result['bert_prediction']} ({result['bert_confidence']:.2%}) | "
              f"Graph: {result['graph_prediction']} ({result['graph_confidence']:.2%})")
        if result['pattern_matches']:
            patterns = ', '.join([m['pattern'] for m in result['pattern_matches'][:3]])
            print(f"Patterns: {patterns}")
        print("-" * 80)
    
    # Test on sample from test set
    print("\n📈 Evaluating on test set sample...")
    with open('./data/email_dataset_test.json', 'r') as f:
        test_data = json.load(f)[:30]
    
    correct_bert = 0
    correct_graphrag = 0
    
    for item in test_data:
        # BERT only
        bert_cat, bert_conf, _ = classifier._get_bert_prediction(item['email'])
        if bert_cat == item['category']:
            correct_bert += 1
        
        # GraphRAG
        result = classifier.classify(item['email'])
        if result['prediction'] == item['category']:
            correct_graphrag += 1
    
    print(f"\n📊 Accuracy Comparison (on {len(test_data)} samples):")
    print(f"   BERT only: {correct_bert/len(test_data):.2%}")
    print(f"   GraphRAG:  {correct_graphrag/len(test_data):.2%}")
    
    improvement = ((correct_graphrag - correct_bert) / correct_bert) * 100 if correct_bert > 0 else 0
    if improvement > 0:
        print(f"   🎯 Improvement: +{improvement:.1f}%")
    else:
        print(f"   📊 Difference: {improvement:.1f}%")
