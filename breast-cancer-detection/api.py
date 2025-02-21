from fastapi import FastAPI
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(features: list):
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}
