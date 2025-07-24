from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
import sys
sys.path.append('.')
from scripts.final_graphrag_classifier import FinalGraphRAGClassifier

# Initialize FastAPI app
app = FastAPI(title="Email Classifier API", version="1.0")

# Initialize classifier (loaded once)
classifier = None

class EmailRequest(BaseModel):
    email: str
    include_explanation: bool = True

class BatchEmailRequest(BaseModel):
    emails: List[str]
    include_explanation: bool = False

class ClassificationResponse(BaseModel):
    email: str
    category: str
    confidence: float
    explanation: Optional[str] = None
    requires_review: bool

class FeedbackRequest(BaseModel):
    email: str
    correct_category: str

@app.on_event("startup")
async def startup_event():
    global classifier
    print("Loading GraphRAG classifier...")
    classifier = FinalGraphRAGClassifier()
    print("Classifier ready!")

@app.get("/")
def read_root():
    return {
        "message": "Email Classifier API",
        "endpoints": {
            "/classify": "Classify single email",
            "/classify/batch": "Classify multiple emails",
            "/feedback": "Submit correction feedback",
            "/health": "Check API health"
        }
    }

@app.post("/classify", response_model=ClassificationResponse)
def classify_email(request: EmailRequest):
    """Classify a single email"""
    try:
        result = classifier.classify(request.email, explanation=request.include_explanation)
        
        return ClassificationResponse(
            email=request.email,
            category=result['prediction'],
            confidence=result['confidence'],
            explanation=result.get('explanation'),
            requires_review=result['confidence'] < 0.8  # Flag low confidence for review
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/batch")
def classify_batch(request: BatchEmailRequest):
    """Classify multiple emails"""
    try:
        results = []
        for email in request.emails:
            result = classifier.classify(email, explanation=request.include_explanation)
            results.append({
                "email": email,
                "category": result['prediction'],
                "confidence": result['confidence'],
                "requires_review": result['confidence'] < 0.8
            })
        
        # Summary statistics
        hr_count = sum(1 for r in results if r['category'] == 'HR')
        sales_count = sum(1 for r in results if r['category'] == 'Sales')
        review_count = sum(1 for r in results if r['requires_review'])
        
        return {
            "results": results,
            "summary": {
                "total": len(results),
                "hr": hr_count,
                "sales": sales_count,
                "requires_review": review_count,
                "confidence_rate": f"{(len(results) - review_count) / len(results) * 100:.1f}%"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    """Submit feedback for continuous learning"""
    try:
        if request.correct_category not in ['HR', 'Sales']:
            raise ValueError("Category must be 'HR' or 'Sales'")
        
        message = classifier.add_feedback(request.email, request.correct_category)
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    """Check if API is healthy"""
    return {
        "status": "healthy",
        "classifier_loaded": classifier is not None
    }

# For testing individual emails quickly
@app.get("/test/{email}")
def quick_test(email: str):
    """Quick test endpoint for browser testing"""
    try:
        result = classifier.classify(email, explanation=True)
        return {
            "email": email,
            "category": result['prediction'],
            "confidence": f"{result['confidence']:.1%}",
            "confidence_level": "High" if result['confidence'] > 0.8 else "Medium" if result['confidence'] > 0.6 else "Low",
            "requires_review": result['confidence'] < 0.8
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting Email Classifier API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
