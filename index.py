from fastapi import FastAPI, UploadFile, File
from tensorflow import keras
from fastapi import File, UploadFile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

app = FastAPI()
model = keras.models.load_model("mnist_model.h5")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert('L').resize((28, 28))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = 1.0 - img_array
    img_array = np.expand_dims(img_array, axis=(0, -1))
    prediction = model.predict(img_array)
    predicted_digit = int(np.argmax(prediction[0]))
    return { "predict": predicted_digit }

@app.get("/")
async def test():
    return {"message": "IAService is working!"}