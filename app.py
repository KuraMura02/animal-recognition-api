# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from utils.predict import predict_animal

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # фронтенд с любого домена
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