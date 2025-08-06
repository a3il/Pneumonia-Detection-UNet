import os
import numpy as np
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import shutil

def split_dataset(base_path, train_path, test_path, split_ratio=0.8):
    # Create train and test directories
    if not os.path.exists(train_path):
        os.makedirs(os.path.join(train_path, 'Normal'))
        os.makedirs(os.path.join(train_path, 'Pneumonia'))
    if not os.path.exists(test_path):
        os.makedirs(os.path.join(test_path, 'Normal'))
        os.makedirs(os.path.join(test_path, 'Pneumonia'))
    
    # Split dataset
    for category in ['Normal', 'Pneumonia']:
        category_path = os.path.join(base_path, category)
        images = os.listdir(category_path)
        train_images, test_images = train_test_split(images, train_size=split_ratio, random_state=42)
        
        for image in train_images:
            shutil.copy(os.path.join(category_path, image), os.path.join(train_path, category, image))
        
        for image in test_images:
            shutil.copy(os.path.join(category_path, image), os.path.join(test_path, category, image))

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
    return accuracy, y_true, y_pred

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def main():
    # Set paths
    dataset_path = r"E:\Chest X-Ray Classification\Dataset\Dataset-2class"
    train_data_path = r"E:\Chest X-Ray Classification\Dataset\train"
    test_data_path = r"E:\Chest X-Ray Classification\Dataset\test"
    model_path = r"E:\Chest X-Ray Classification\Paper\Output\CNN\trained_model.h5"

    # Split the dataset into training and test sets
    split_ratio = 0.8
    split_dataset(dataset_path, train_data_path, test_data_path, split_ratio)

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
    test_accuracy, y_true, y_pred = calculate_accuracy(model, test_generator)
    print("Test Accuracy:", test_accuracy)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", conf_matrix)

    # Plot and save confusion matrix
    plt.figure()
    plot_confusion_matrix(conf_matrix, classes=['Pneumonia', 'Normal'], title='CNN Binary Classifier Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()
