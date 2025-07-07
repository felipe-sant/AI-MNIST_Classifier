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

    # Dividir treino e validação manualmente (10% validação)
    val_split = 0.1
    val_size = int(x_train.shape[0] * val_split)
    x_val = x_train[:val_size]
    y_val = y_train[:val_size]
    x_train_new = x_train[val_size:]
    y_train_new = y_train[val_size:]

    # Data augmentation apenas no treino
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.10,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(x_train_new)

    # Generator para treino
    train_gen = datagen.flow(x_train_new, y_train_new, batch_size=64)

    # Modelo melhorado
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    checkpoint = keras.callbacks.ModelCheckpoint("mnist_best_model.h5", monitor="val_accuracy", save_best_only=True)

    # Treinar modelo com data augmentation
    model.fit(
        train_gen,
        epochs=30,
        validation_data=(x_val, y_val),
        steps_per_epoch=x_train_new.shape[0] // 64,
        callbacks=[early_stop, checkpoint],
        verbose=2
    )

    # Avaliar no teste
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Acurácia em teste: {test_acc:.4f}")
    print("Modelo treinado e salvo como mnist_best_model.h5")

if __name__ == "__main__":
    main()