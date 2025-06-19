"""Test API endpoints for sentiment analysis service"""
import requests
from requests.exceptions import RequestException, Timeout

def predict_sentiment_api(text: str, timeout: int = 30):
    """Make API call with proper timeout and error handling"""
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"text": text},
            timeout=timeout  # Add timeout
        )
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()
    
    except Timeout:
        print(f"Request timed out after {timeout} seconds")
        return None
    except RequestException as e:
        print(f"Request failed: {e}")
        return None

result = predict_sentiment_api("This movie was amazing!", timeout=10)
if result:
    print(result)

def predict_batch_sentiment_api(texts: list, timeout: int = 60):
    """Make batch API call with proper timeout"""
    try:
        response = requests.post(
            "http://localhost:8000/predict/batch",
            json={"texts": texts},
            timeout=timeout  # Longer timeout for batch processing
        )
        response.raise_for_status()
        return response.json()
    
    except Timeout:
        print(f"Batch request timed out after {timeout} seconds")
        return None
    except RequestException as e:
        print(f"Batch request failed: {e}")
        return None

texts = [
    "This movie was amazing!",
    "Terrible waste of time", 
    "It was okay, nothing special"
]
result = predict_batch_sentiment_api(texts, timeout=30)
if result:
    print(result)
