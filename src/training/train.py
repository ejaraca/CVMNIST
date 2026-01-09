"""
Descarga MNIST
Construye la red LeNet-5
Entrena y evalúa
Exporta el modelo en una carpeta outputs/model
"""
import os
import argparse
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models

def build_lenet(input_shape=(28, 28, 1), n_classes=10):
    model = models.Sequential([
        layers.Conv2D(6, kernel_size=(5,5), activation='relu', input_shape=input_shape),
        layers.AveragePooling2D(),
        layers.Conv2D(16, kernel_size=(5,5), activation='relu'),
        layers.AveragePooling2D(),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])
    return model

def main(args):
    # Cargar datos
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test  = x_test.reshape(-1, 28, 28, 1) / 255.0

    # Construir modelo
    model = build_lenet()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Entrenar
    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=(x_test, y_test))

    # Evaluar
    loss, acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {acc:.4f}")

    # Guardar modelo
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "model")
    model.save(model_path)
    print(f"Modelo guardado en: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default=os.environ.get('AZUREML_MODEL_DIR', 'outputs'))
    args = parser.parse_args()

    main(args)
