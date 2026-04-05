# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from utils.predict import predict_animal
import os

port = int(os.environ.get("PORT", 10000))
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://miranzahotel.kesug.com"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_animal(image_bytes)
    return result

@app.get("/")
def home():
    return {"message": "Animal AI API работает!"}
