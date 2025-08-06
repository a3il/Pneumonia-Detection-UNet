import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
import gc
import psutil
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow_addons.callbacks import TimeStopping
from tensorflow_addons.layers import GradientReversal

class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32")

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

def setup_memory_optimization():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": False})

def clear_session_and_models():
    tf.keras.backend.clear_session()
    gc.collect()

def get_data_generators(dataset_path, img_width=512, img_height=512, batch_size=4):
    print("Creating data generators...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
        brightness_range=[0.5, 1.5],  # Additional data augmentation: brightness adjustment
        vertical_flip=True  # Additional data augmentation: vertical flip
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_width, img_height),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='binary',  # Use 'binary' for binary classification
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_width, img_height),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='binary',  # Use 'binary' for binary classification
        subset='validation',
        shuffle=False
    )

    # Reshape X_train to match the expected input shape for RandomOverSampler
    X_train = np.array(train_generator.filepaths)
    y_train = train_generator.classes

    ros = RandomOverSampler(sampling_strategy='not majority', random_state=None)

    X_resampled, y_resampled = ros.fit_resample(X_train.reshape(-1, 1), y_train)

    # Print the sample size before oversampling
    print("Sample size before oversampling:", len(X_train))
    # Print class distribution before oversampling
    print("Class distribution before oversampling:", np.bincount(y_train))

    # Print the sample size after oversampling
    print("Sample size after oversampling:", len(X_resampled))
    # Print class distribution after oversampling
    print("Class distribution after oversampling:", np.bincount(y_resampled))

    # Convert filepaths back to generator-friendly format
    X_resampled = X_resampled.flatten()

    # Convert class values to strings
    y_resampled = y_resampled.astype(str)

    # Create a new generator using the oversampled data
    train_generator = train_datagen.flow_from_dataframe(
        pd.DataFrame({'filename': X_resampled, 'class': y_resampled}),
        directory=None,
        x_col='filename',
        y_col='class',
        target_size=(img_width, img_height),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='binary',  # Use 'binary' for binary classification
        shuffle=True
    )

    return train_generator, validation_generator

def build_model(img_width, img_height):
    print("Building the model...")
    input_shape = (img_width, img_height, 1)

    cnn = Sequential()
    cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(64, (3, 3), activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(128, (3, 3), activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(BatchNormalization())
    cnn.add(Conv2D(256, (3, 3), activation="relu"))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(BatchNormalization())
    cnn.add(Flatten())
    cnn.add(Dense(512, activation='relu'))
    cnn.add(Dense(256, activation='relu'))
    cnn.add(Dense(1, activation='sigmoid'))  # Use sigmoid for binary classification

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    cnn.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return cnn

def train_model(model, train_generator, validation_generator, batch_size, epochs=100, results_path="E:/VSCODE/RESULTS/cnn"):
    print("Training the model...")
    checkpoint_path = os.path.join(results_path, "model_checkpoint.h5")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=5, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=5, restore_best_weights=True)

    steps_per_epoch = len(train_generator)
    validation_steps = len(validation_generator)
    time_stopping = TimeStopping(limit=86400)  # Stop if training runs for more than 24 hours

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stopping, time_stopping]
    )

    model_path = os.path.join(results_path, "trained_model.h5")
    model.save(model_path)

    return history

def evaluate_model(model, validation_generator):
    print("Evaluating the model...")
    evaluation = model.evaluate(validation_generator)
    return evaluation

def generate_predictions(model, validation_generator):
    print("Generating predictions...")
    predictions = model.predict(validation_generator)
    y_true = validation_generator.classes
    y_pred = np.round(predictions).flatten().astype(int)  # Convert probabilities to binary classes
    return y_true, y_pred

def plot_training_history(history, results_path="E:/VSCODE/RESULTS/cnn"):
    print("Plotting training history...")
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(results_path, 'accuracy_plot.png'))
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(results_path, 'loss_plot.png'))
    plt.show()

def memory_profiling():
    print("Profiling memory usage...")
    process = psutil.Process(os.getpid())
    memory_use = process.memory_info().rss / (1024 * 1024 * 1024)  # memory use in GB
    print('Memory use:', memory_use, 'GB')

def print_metrics(y_true, y_pred):
    print("Precision, Recall, F1 Score:")
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)

    # Specificity, PPV, NPV
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    print("Specificity:", specificity)
    print("PPV:", ppv)
    print("NPV:", npv)

def save_confusion_matrix(y_true, y_pred, results_path="E:/VSCODE/RESULTS/cnn"):
    print("Saving Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_path, 'confusion_matrix.png'))
    plt.show()

def plot_roc_auc(y_true, y_pred_proba, results_path="E:/VSCODE/RESULTS/cnn"):
    print("Plotting ROC-AUC curve...")
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_path, 'roc_auc_curve.png'))
    plt.show()

def main():
    setup_memory_optimization()
    memory_profiling()

    clear_session_and_models()
    memory_profiling()

    dataset_path = "E:/dataset/2class"
    results_path = "E:/VSCODE/RESULTS/cnn"

    batch_size = 4 # Reduced batch size
    epochs = 100
    img_width = 512
    img_height = 512

    train_generator, validation_generator = get_data_generators(dataset_path, batch_size=batch_size)

    model = build_model(img_width, img_height)

    history = train_model(model, train_generator, validation_generator, batch_size, epochs, results_path)

    # Plotting training history
    plot_training_history(history, results_path)

    evaluation = evaluate_model(model, validation_generator)
    print("Evaluation:", evaluation)

    y_true, y_pred = generate_predictions(model, validation_generator)

    print_metrics(y_true, y_pred)

    # Print and save confusion matrix
    save_confusion_matrix(y_true, y_pred, results_path)

    # Plot and save ROC-AUC curve
    y_pred_proba = model.predict(validation_generator)
    plot_roc_auc(y_true, y_pred_proba)

if __name__ == "__main__":
    main()
