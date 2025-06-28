from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def indenfityImage(url):
    img = Image.open(url).convert('L')
    print(img)
    img = img.resize((28, 28))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = 1.0 - img_array
    img_array = np.expand_dims(img_array, axis=(0, -1))

    model = keras.models.load_model("mnist_model.h5")

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction[0])

    plt.imshow(img_array[0, :, :, 0], cmap='gray')
    plt.title(f"Previsto: {predicted_digit}")
    plt.axis('off')
    plt.show()