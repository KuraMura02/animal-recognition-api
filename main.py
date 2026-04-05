# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from utils.predict import predict_animal
import uvicorn

app = FastAPI(title="Animal Classifier API")

# ✨ Ключевая настройка CORS для соединения с фронтендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # На время теста можно так. Потом укажите конкретный домен
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Animal Classification API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = predict_animal(image_bytes)
        return {"status": "success", "prediction": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
