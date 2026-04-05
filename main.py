# main.py
from fastapi import FastAPI, File, UploadFile
from utils.predict import predict_animal
import uvicorn

app = FastAPI(title="Animal Classifier API")


@app.get("/")
def root():
    return {"message": "Animal Classification API with ONNX"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Читаем изображение
        image_bytes = await file.read()

        # Предсказание
        result = predict_animal(image_bytes)

        return {
            "status": "success",
            "prediction": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
