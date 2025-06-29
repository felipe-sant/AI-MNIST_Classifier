import os
import tensorflow as tf
import warnings
import absl.logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
absl.logging.set_verbosity(absl.logging.ERROR)

from fastapi import FastAPI, UploadFile, File
from tensorflow import keras
from src.functions.indetifyImage import indenfityImage

app = FastAPI()
model = keras.models.load_model("mnist_model.h5")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    response = await indenfityImage(model, file)
    return response

@app.get("/")
async def teste():
    return "Teste"