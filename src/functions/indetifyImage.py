from fastapi import File, UploadFile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

async def indenfityImage(model, file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert('L').resize((28, 28))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = 1.0 - img_array
    img_array = np.expand_dims(img_array, axis=(0, -1))
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction[0])

    return predicted_digit