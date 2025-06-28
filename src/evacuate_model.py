import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Carregar modelo
    model = keras.models.load_model("mnist_model.h5")

    # Carregar dados de teste
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1)

    # Avaliar modelo
    loss, acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Acurácia no teste: {acc*100:.2f}%")

    # Visualizar algumas previsões
    predictions = model.predict(x_test)
    for i in range(5):
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"Verdadeiro: {y_test[i]}, Previsto: {np.argmax(predictions[i])}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()