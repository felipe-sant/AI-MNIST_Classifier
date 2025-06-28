import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def main():
    # Carregar dados MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Remodelar para [amostras, 28, 28, 1]
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Construir modelo simples
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Treinar modelo
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

    # Salvar modelo
    model.save("mnist_model.h5")
    print("Modelo treinado e salvo como mnist_model.h5")

if __name__ == "__main__":
    main()