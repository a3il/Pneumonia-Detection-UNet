import os
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

def load_model(model_path):
    print("Loading the trained model...")
    model = tf.keras.models.load_model(model_path)
    return model

def calculate_accuracy(model, test_generator):
    print("Calculating accuracy on test data...")
    predictions = model.predict(test_generator)
    y_true = test_generator.classes
    y_pred = (predictions > 0.5).astype(int).flatten()  # Convert probabilities to binary classes
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def main():
    # Set paths
    model_path = "E:/VSCODE/RESULTS/cnn/trained_model.h5"
    test_data_path = "E:/dataset/2class"

    # Parameters
    batch_size = 32
    img_width = 512
    img_height = 512

    # Load the trained model
    model = load_model(model_path)

    # Create a test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_path,
        target_size=(img_width, img_height),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='binary',  # Use 'binary' for binary classification
        shuffle=False
    )

    # Calculate accuracy on test data
    test_accuracy = calculate_accuracy(model, test_generator)
    print("Test Accuracy:", test_accuracy)

if __name__ == "__main__":
    main()
