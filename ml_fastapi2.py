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
model: SVC = joblib.load('lightweight.pkl')
vectorizer: TfidfVectorizer = joblib.load('ntransform.pkl')

@app.post("/predict/")
async def predict(email: EmailText):
    try:
        # Transform the email text to vector
        vectorized_text = vectorizer.transform([email.email_text])
        
        # Convert sparse matrix to dense array
        dense_input = vectorized_text.toarray()
        
        # Predict using the SVC model
        prediction = model.predict(dense_input)

        # Get confidence scores using predict_proba
        confidence_scores = model.predict_proba(dense_input)

        # Debug: Print raw confidence score to verify it's between 0 and 1
        print("Raw confidence score:", confidence_scores[0][1])

        # Prepare the response
        result = {
            "prediction": "spam" if prediction[0] == 1 else "ham",
            "confidence_score": float(confidence_scores[0][1])  # Keep confidence score between 0-1
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
