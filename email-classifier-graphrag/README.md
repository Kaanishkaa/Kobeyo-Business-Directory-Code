
# Email Classifier with GraphRAG 🚀

An advanced email classification system that categorizes business emails as HR or Sales using BERT, Knowledge Graphs, and GraphRAG (Graph Retrieval-Augmented Generation) for continuous learning.

## 🌟 Features

- **BERT-based Classification**: Uses DistilBERT for initial email classification
- **Knowledge Graph Enhancement**: Neo4j-based graph structure for pattern recognition
- **GraphRAG System**: Combines neural networks with graph-based reasoning
- **Continuous Learning**: Learns from user feedback to improve over time
- **Web Crawler Integration**: Automatically extracts emails from business websites
- **REST API**: Production-ready API for easy integration
- **High Accuracy**: 100% accuracy on high-confidence predictions

## 📊 Performance Metrics

- Overall Accuracy: 86%
- High-Confidence Predictions (>80%): 100% accurate
- Model improves with feedback through continuous learning

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Email Input     │────▶│ BERT Classifier  │────▶│ Knowledge Graph │
└─────────────────┘     └──────────────────┘     └─────────────────┘
          │                         │
          ▼                         ▼
┌─────────────────┐      ┌──────────────────┐
│ Neural Predict  │      │ Graph Patterns   │
└─────────────────┘      └──────────────────┘
          │                         │
          └──────────┬──────────────┘
                     │
          ┌──────────▼──────────┐
          │   GraphRAG Fusion   │
          └─────────────────────┘
                     │
          ┌──────────▼──────────┐
          │  Final Prediction   │
          │  HR / Sales         │
          └─────────────────────┘
```

## 📁 Project Structure

```
email-classifier-graphrag/
├── data/
│   ├── email_dataset_*.json
│   └── graph/
│       └── enhanced_knowledge_graph.json
├── models/
│   ├── distilbert_classifier/
│   └── history/
├── scripts/
│   ├── api_server.py
│   ├── train_bert_simple.py
│   ├── final_graphrag_classifier.py
│   ├── continuous_learning.py
│   ├── email_crawler.py
│   └── crawl_and_classify.py
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 4GB+ RAM
- CUDA-capable GPU (optional)

### Installation

```bash
git clone https://github.com/yourusername/email-classifier-graphrag.git
cd email-classifier-graphrag
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

### Step-by-Step Usage

1. **Create Dataset**:
    ```bash
    python scripts/create_dataset.py
    ```

2. **Train BERT Model**:
    ```bash
    python scripts/train_bert_simple.py
    ```

3. **Build Knowledge Graph**:
    ```bash
    python scripts/build_knowledge_graph.py
    ```

4. **Enhance Graph**:
    ```bash
    python scripts/enhanced_graphrag.py
    ```

5. **Start API Server**:
    ```bash
    python scripts/api_server.py
    ```

6. **Test Classifier**:
    ```bash
    curl http://localhost:8000/test/careers@company.com
    ```

### Continuous Learning

- **Submit Feedback**:
    ```bash
    curl -X POST http://localhost:8000/feedback     -H "Content-Type: application/json"     -d '{"email": "info@restaurant.com", "correct_category": "Sales"}'
    ```

- **Retrain**:
    ```bash
    python scripts/continuous_learning.py retrain
    ```

## 🕷️ Web Crawler Integration

Create a CSV with website URLs, then run:
```bash
python scripts/crawl_and_classify.py
```

## 📊 API Endpoints

| Endpoint             | Method | Description                    |
|----------------------|--------|--------------------------------|
| `/`                  | GET    | API Docs                       |
| `/health`            | GET    | Health Check                   |
| `/classify`          | POST   | Classify Single Email          |
| `/classify/batch`    | POST   | Classify Multiple Emails       |
| `/feedback`          | POST   | Submit Correction              |
| `/test/{email}`      | GET    | Quick Test Endpoint            |

## 🔧 Configuration

- Change feedback threshold in `continuous_learning.py`
- Modify crawler depth & timeouts in `email_crawler.py`

## 🐛 Troubleshooting

- **Low Confidence**: Submit feedback
- **API Fails**: Ensure server is running and port 8000 is free
- **OOM**: Lower batch size or switch to DistilBERT

