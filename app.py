""" Sentiment Analysis API using FastAPI and DistilBERT
This API provides endpoints for sentiment analysis using a pre-trained DistilBERT model."""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="DistilBERT-based sentiment analysis API",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

class TextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: List[str]

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]

@app.on_event("startup")
async def load_model():
    """Load model and tokenizer on startup"""
    global model, tokenizer, device
    
    try:
        model_path = "./data/sentiment_analysis"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully on {device}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

def predict_sentiment_internal(text: str) -> Dict:
    """Internal prediction function"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Tokenize the text
    encoded_text = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
        confidence_score = probabilities[0][prediction].item()
    
    # Add neutral category for low confidence predictions
    if confidence_score < 0.65:
        sentiment = 'neutral'
    else:
        sentiment = 'positive' if prediction.item() == 1 else 'negative'
    
    return {
        'text': text,
        'sentiment': sentiment,
        'confidence': round(confidence_score, 4)
    }

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Sentiment Analysis API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    model_loaded = model is not None and tokenizer is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "device": str(device) if device else None
    }

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(input_data: TextInput):
    """Predict sentiment for a single text"""
    try:
        if not input_data.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        result = predict_sentiment_internal(input_data.text)
        return SentimentResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchSentimentResponse)
async def predict_batch_sentiment(input_data: BatchTextInput):
    """Predict sentiment for multiple texts"""
    try:
        if not input_data.texts:
            raise HTTPException(status_code=400, detail="Texts list cannot be empty")
        
        if len(input_data.texts) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size cannot exceed 100")
        
        results = []
        for text in input_data.texts:
            if text.strip():  # Skip empty texts
                result = predict_sentiment_internal(text)
                results.append(SentimentResponse(**result))
        
        return BatchSentimentResponse(results=results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
