#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:16:57 2024

@author: kushalbasaula
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np

# Define the request model
class EmailText(BaseModel):
    email_text: str

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model and vectorizer
<<<<<<< HEAD
model: SVC = joblib.load('Trained_model.pkl')
vectorizer: TfidfVectorizer = joblib.load('transform.pkl')
=======
model: SVC = joblib.load('/Users/kushalbasaula/Documents/Individual Project/final/Trained_model.pkl')
vectorizer: TfidfVectorizer = joblib.load('/Users/kushalbasaula/Documents/Individual Project/final/transform.pkl')
>>>>>>> 6db976789bd06095368bce30f268e3d6cea33e7b

# Custom threshold (already defined)
THRESHOLD = 0.5856319934385817

@app.post("/predict/")
async def predict(email: EmailText):
    try:
        # Transform the email text to vector
        vectorized_text = vectorizer.transform([email.email_text])
        
        # Convert sparse matrix to dense array
        dense_input = vectorized_text.toarray()
        
        # Get confidence scores using predict_proba
        confidence_scores = model.predict_proba(dense_input)

        # Custom prediction logic based on the threshold
        prediction = 1 if confidence_scores[0][1] >= THRESHOLD else 0  # 1 for spam, 0 for ham

        # Prepare the response
        result = {
            "prediction": "spam" if prediction == 1 else "ham",
            "confidence_score": float(confidence_scores[0][1])  # Score for spam class
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Root endpoint for health check
@app.get("/")
async def read_root():
    return {"message": "Hello, this is the spam/ham classification API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
