import os
import tensorflow as tf
import warnings
import absl.logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
absl.logging.set_verbosity(absl.logging.ERROR)

from fastapi import FastAPI, UploadFile, File
from tensorflow import keras

from src.functions.identifyImage import identifyImage

app = FastAPI()
model = keras.models.load_model("mnist_model.h5")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    prediction = await identifyImage(model, file)
    return { "predict": prediction }

@app.get("/")
async def test():
    return {"message": "IAService is working!"}