"""Sentiment Analysis Interface using Streamlit and DistilBERT"""

import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        model_path = "./data/sentiment_analysis"
        
        if os.path.exists(model_path):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            
            return model, tokenizer, device
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def predict_sentiment(model, tokenizer, text, device):
    """Predict sentiment for given text"""
    model.eval()
    
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
        'sentiment': sentiment,
        'confidence': confidence_score
    }

def main():
    st.set_page_config(page_title="Sentiment Analysis", page_icon="🎭")
    
    st.title("🎭 Sentiment Analyzer")
    st.write("DistilBERT-based sentiment analysis for sentiment reviews")
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer()
    
    if model is None:
        st.error("❌ Model not found! Please train the model first by running the notebook.")
        st.stop()
    
    st.success(f"✅ Model loaded successfully! Running on: {device}")
    
    # Text input
    user_input = st.text_area(
        "Enter your review:",
        placeholder="Type your review here...",
        height=150
    )
    
    if st.button("Analyze Sentiment", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                result = predict_sentiment(model, tokenizer, user_input, device)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if result['sentiment'] == 'positive':
                        st.success(f"😊 Positive")
                    elif result['sentiment'] == 'negaative':
                        st.error(f"😞 Negative")
                    else:
                        st.info(f"😐 Neutral")
                
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.2%}")
        else:
            st.warning("Please enter a review.")
    
    # Sample reviews
    st.markdown("---")
    st.subheader("Try Sample Reviews")
    
    samples = [
        "This movie was really great! I enjoyed every minute of it.",
        "I wouldn't recommend this movie to anyone. It was terrible.",
        "The acting was okay but the plot could have been better."
    ]
    
    for i, sample in enumerate(samples):
        if st.button(f"Sample {i+1}: {sample[:40]}...", key=f"sample_{i}"):
            result = predict_sentiment(model, tokenizer, sample, device)
            
            st.write(f"**Review:** {sample}")
            if result['sentiment'] == 'positive':
                st.success(f"😊 Positive ({result['confidence']:.1%})")
            else:
                st.error(f"😞 Negative ({result['confidence']:.1%})")

if __name__ == "__main__":
    main()
