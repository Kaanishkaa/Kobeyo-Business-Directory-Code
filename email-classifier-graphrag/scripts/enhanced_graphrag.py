import json
from collections import defaultdict

# Define correct pattern mappings with confidence
PATTERN_MAPPINGS = {
    # Strong HR patterns (confidence > 0.9)
    'careers': ('HR', 0.95),
    'hiring': ('HR', 0.95),
    'recruitment': ('HR', 0.95),
    'hr': ('HR', 0.98),
    'talent': ('HR', 0.90),
    'jobs': ('HR', 0.90),
    'resume': ('HR', 0.95),
    'cv': ('HR', 0.95),
    'hiringteam': ('HR', 0.95),
    'recruiting': ('HR', 0.95),
    'talentacquisition': ('HR', 0.98),
    'peopleops': ('HR', 0.95),
    'humanresources': ('HR', 0.98),
    'workforce': ('HR', 0.85),
    'employee': ('HR', 0.85),
    'payroll': ('HR', 0.85),
    'onboarding': ('HR', 0.85),
    
    # Strong Sales patterns (confidence > 0.9)
    'sales': ('Sales', 0.95),
    'orders': ('Sales', 0.95),
    'purchase': ('Sales', 0.95),
    'deals': ('Sales', 0.95),
    'revenue': ('Sales', 0.90),
    'customer': ('Sales', 0.85),
    'client': ('Sales', 0.85),
    'account': ('Sales', 0.85),
    'pricing': ('Sales', 0.90),
    'quotes': ('Sales', 0.90),
    'leads': ('Sales', 0.90),
    'pipeline': ('Sales', 0.90),
    'crm': ('Sales', 0.85),
    'upsell': ('Sales', 0.90),
    'renewal': ('Sales', 0.85),
    'contract': ('Sales', 0.85),
    
    # Ambiguous patterns (lower confidence)
    'info': ('Ambiguous', 0.3),
    'contact': ('Ambiguous', 0.3),
    'hello': ('Ambiguous', 0.3),
    'admin': ('Ambiguous', 0.4),
    'support': ('Ambiguous', 0.4),
    'team': ('Ambiguous', 0.5),
    'manager': ('Ambiguous', 0.5),
    'business': ('Sales', 0.6),  # Slight Sales bias
    'development': ('Ambiguous', 0.5),
    'operations': ('Ambiguous', 0.5),
}

def enhance_graph_data():
    """Enhance the graph with better pattern categorization"""
    print("🔧 Enhancing knowledge graph...")
    
    # Load existing graph
    with open('./data/graph/email_knowledge_graph.json', 'r') as f:
        graph_data = json.load(f)
    
    # Update pattern statistics based on correct mappings
    enhanced_patterns = {}
    pattern_corrections = 0
    
    for pattern, emails in graph_data['patterns'].items():
        if pattern in PATTERN_MAPPINGS:
            correct_category, confidence = PATTERN_MAPPINGS[pattern]
            
            # Count actual categories in data
            category_counts = defaultdict(int)
            for email in emails:
                if email in graph_data['nodes']:
                    category_counts[graph_data['nodes'][email]['category']] += 1
            
            # Check if correction needed
            total = sum(category_counts.values())
            if total > 0:
                current_dominant = max(category_counts.items(), key=lambda x: x[1])[0]
                
                if correct_category != 'Ambiguous' and current_dominant != correct_category:
                    pattern_corrections += 1
                    print(f"   Correcting '{pattern}': {current_dominant} -> {correct_category}")
            
            enhanced_patterns[pattern] = {
                'expected_category': correct_category,
                'confidence': confidence,
                'emails': emails,
                'actual_distribution': dict(category_counts)
            }
        else:
            # Keep unknown patterns as is
            enhanced_patterns[pattern] = {
                'expected_category': 'Unknown',
                'confidence': 0.5,
                'emails': emails
            }
    
    # Save enhanced graph
    enhanced_graph = {
        'nodes': graph_data['nodes'],
        'edges': graph_data['edges'],
        'patterns': graph_data['patterns'],
        'enhanced_patterns': enhanced_patterns,
        'pattern_mappings': PATTERN_MAPPINGS
    }
    
    with open('./data/graph/enhanced_knowledge_graph.json', 'w') as f:
        json.dump(enhanced_graph, f, indent=2)
    
    print(f"✅ Enhanced graph created!")
    print(f"   Pattern corrections: {pattern_corrections}")
    print(f"   Total patterns: {len(enhanced_patterns)}")
    
    # Show ambiguous patterns
    ambiguous = [p for p, data in enhanced_patterns.items() 
                 if data['expected_category'] == 'Ambiguous']
    print(f"   Ambiguous patterns: {', '.join(ambiguous[:10])}")
    
    return enhanced_graph

# Create a confidence booster function
def calculate_graphrag_confidence(bert_conf, graph_conf, agreement):
    """Calculate final confidence with smart boosting"""
    if agreement:
        # Boost confidence when BERT and Graph agree
        boosted = max(bert_conf, graph_conf)
        if bert_conf > 0.7 and graph_conf > 0.8:
            boosted = min(0.95, boosted + 0.1)
        return boosted
    else:
        # Average when they disagree, with slight BERT preference
        return bert_conf * 0.7 + graph_conf * 0.3

# Test the enhancement
if __name__ == "__main__":
    enhance_graph_data()
    
    print("\n📊 Testing enhanced classification logic...")
    
    test_cases = [
        ("careers@company.com", "HR", "Should be high confidence HR"),
        ("sales@company.com", "Sales", "Should be high confidence Sales"),
        ("info@company.com", "Ambiguous", "Should be low confidence"),
        ("talent.sales@company.com", "Mixed", "Contains both patterns"),
    ]
    
    for email, expected, description in test_cases:
        prefix = email.split('@')[0].lower()
        patterns_found = []
        
        for pattern, (category, conf) in PATTERN_MAPPINGS.items():
            if pattern in prefix:
                patterns_found.append((pattern, category, conf))
        
        print(f"\n{email}: {description}")
        print(f"   Patterns found: {patterns_found}")
        
        if patterns_found:
            # Calculate weighted prediction
            weights = {'HR': 0, 'Sales': 0, 'Ambiguous': 0}
            for pattern, cat, conf in patterns_found:
                if cat != 'Ambiguous':
                    weights[cat] += conf
                else:
                    weights['HR'] += 0.5 * conf
                    weights['Sales'] += 0.5 * conf
            
            best_cat = max(weights.items(), key=lambda x: x[1])
            print(f"   Prediction: {best_cat[0]} (weight: {best_cat[1]:.2f})")
