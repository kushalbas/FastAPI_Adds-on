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
model: MultinomialNB = joblib.load('mlpTrained_model.pkl')
vectorizer: TfidfVectorizer = joblib.load('mlptransform.pkl')

@app.post("/predict/")
async def predict(email: EmailText):
    try:
        # Transform the email text to vector
        vectorized_text = vectorizer.transform([email.email_text])
        
        # Convert sparse matrix to dense array
        dense_input = vectorized_text.toarray()
        
        # Get confidence scores using predict_proba
        confidence_scores = model.predict_proba(dense_input)

        # Model prediction (1 for spam, 0 for ham) based on maximum probability
        prediction = np.argmax(confidence_scores[0])  # Get the class with the highest probability

        # Prepare the response
        result = {
            "prediction": "spam" if prediction == 1 else "ham",
            "confidence_score": float(confidence_scores[0][prediction])  # Score for predicted class
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
