from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np

# Define the request model
class EmailText(BaseModel):
    email_text: str

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model and vectorizer
model: MLPClassifier = joblib.load('neuralTrained_model.pkl')
vectorizer: TfidfVectorizer = joblib.load('neuraltransform.pkl')

# Custom threshold (already defined)
THRESHOLD = 0.5053181026693784

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


# Include a root endpoint for simple health check
@app.get("/")
async def read_root():
    return {"message": "Hello, this is the spam/ham classification API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
