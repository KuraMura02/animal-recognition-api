from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from utils.predict import predict_animal
import uvicorn

app = FastAPI(title="Animal Recognition API")

# CORS - разрешаем всем (для теста)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Проверка что API работает"""
    return {
        "message": "Animal Classification API is running",
        "status": "ok",
        "endpoints": {
            "predict": "/predict/ (POST)",
            "docs": "/docs"
        }
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Предсказание животного по фото"""
    try:
        # Проверка типа файла
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Файл должен быть изображением"}
            )
        
        # Читаем и предсказываем
        image_bytes = await file.read()
        result = predict_animal(image_bytes)
        
        return {
            "status": "success",
            "prediction": result,
            "file_name": file.filename
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/health")
async def health():
    """Health check для Railway"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
