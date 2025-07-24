from neo4j import GraphDatabase
import json
from collections import defaultdict
import os
from dotenv import load_dotenv

# For now, we'll use a simple in-memory graph structure
# Later we'll connect to actual Neo4j

class EmailKnowledgeGraph:
    def __init__(self):
        self.nodes = {}  # email -> properties
        self.edges = defaultdict(list)  # relationships
        self.patterns = defaultdict(list)  # pattern -> emails
        
    def add_email_node(self, email_data):
        """Add an email node to the graph"""
        email = email_data['email']
        self.nodes[email] = {
            'email': email,
            'category': email_data['category'],
            'confidence': email_data['confidence'],
            'prefix_pattern': email_data['prefix_pattern'],
            'domain': email_data['domain']
        }
        
        # Add to pattern index
        pattern = email_data['prefix_pattern']
        self.patterns[pattern].append(email)
        
    def add_similarity_edge(self, email1, email2, similarity):
        """Add similarity relationship between emails"""
        if similarity > 0.5:  # Only add if similar enough
            self.edges[email1].append({
                'to': email2,
                'type': 'SIMILAR_TO',
                'weight': similarity
            })
            
    def find_similar_emails(self, email, top_k=5):
        """Find similar emails based on patterns"""
        if email not in self.nodes:
            # For new emails, find by pattern
            prefix = email.split('@')[0].lower()
            # Remove numbers and special chars
            import re
            prefix_clean = re.sub(r'[0-9_.-]', '', prefix)
            
            similar = []
            for pattern, emails in self.patterns.items():
                if pattern in prefix_clean or prefix_clean in pattern:
                    for e in emails[:top_k]:
                        node = self.nodes[e]
                        similar.append({
                            'email': e,
                            'category': node['category'],
                            'confidence': node['confidence'],
                            'pattern_match': pattern
                        })
            return similar
        
        # For existing emails, use edges
        similar = []
        for edge in self.edges.get(email, [])[:top_k]:
            node = self.nodes[edge['to']]
            similar.append({
                'email': edge['to'],
                'category': node['category'],
                'confidence': node['confidence'],
                'similarity': edge['weight']
            })
        return similar
        
    def get_pattern_statistics(self):
        """Get statistics about patterns"""
        stats = {}
        for pattern, emails in self.patterns.items():
            categories = defaultdict(int)
            total_confidence = 0
            
            for email in emails:
                node = self.nodes[email]
                categories[node['category']] += 1
                total_confidence += node['confidence']
                
            stats[pattern] = {
                'count': len(emails),
                'categories': dict(categories),
                'avg_confidence': total_confidence / len(emails) if emails else 0,
                'dominant_category': max(categories.items(), key=lambda x: x[1])[0] if categories else None
            }
            
        return stats
    
    def save_to_file(self, filepath):
        """Save graph to JSON file"""
        data = {
            'nodes': self.nodes,
            'edges': {k: v for k, v in self.edges.items()},
            'patterns': {k: v for k, v in self.patterns.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_from_file(self, filepath):
        """Load graph from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.nodes = data['nodes']
            self.edges = defaultdict(list, data['edges'])
            self.patterns = defaultdict(list, data['patterns'])

# Build the knowledge graph
print("🏗️ Building Email Knowledge Graph...")

# Load training data
with open('./data/email_dataset_train.json', 'r') as f:
    train_data = json.load(f)

# Create graph
graph = EmailKnowledgeGraph()

# Add all emails to graph
print(f"📧 Adding {len(train_data)} emails to graph...")
for email_data in train_data:
    graph.add_email_node(email_data)

# Calculate simple similarities (based on same pattern)
print("🔗 Creating similarity relationships...")
for pattern, emails in graph.patterns.items():
    # Connect emails with same pattern
    for i, email1 in enumerate(emails):
        for email2 in emails[i+1:i+6]:  # Connect to next 5 emails
            graph.add_similarity_edge(email1, email2, 0.8)  # High similarity for same pattern

# Save graph
print("💾 Saving graph...")
os.makedirs('./data/graph', exist_ok=True)
graph.save_to_file('./data/graph/email_knowledge_graph.json')

# Display statistics
print("\n📊 Graph Statistics:")
print(f"   Total nodes (emails): {len(graph.nodes)}")
print(f"   Total patterns: {len(graph.patterns)}")
print(f"   Total edges: {sum(len(edges) for edges in graph.edges.values())}")

# Show top patterns
print("\n🔝 Top 10 Patterns:")
pattern_stats = graph.get_pattern_statistics()
sorted_patterns = sorted(pattern_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:10]

for pattern, stats in sorted_patterns:
    print(f"   {pattern}: {stats['count']} emails, "
          f"dominant category: {stats['dominant_category']}, "
          f"avg confidence: {stats['avg_confidence']:.2f}")

# Test similarity search
print("\n🔍 Testing similarity search:")
test_emails = [
    "careers@newcompany.com",
    "sales@newcompany.com",
    "hiring@newcompany.com"
]

for test_email in test_emails:
    print(f"\n   Similar to '{test_email}':")
    similar = graph.find_similar_emails(test_email, top_k=3)
    for sim in similar:
        print(f"   - {sim['email']} ({sim['category']}) - pattern: {sim.get('pattern_match', 'N/A')}")

print("\n✅ Knowledge graph built successfully!")
print("📁 Graph saved to: ./data/graph/email_knowledge_graph.json")
