#import required libraries
import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models

print('Modules imported')

#load and pre-process the data
def load_audio_files_from_folder(folder_path, sample_rate=16000):
    audio_files = []
    labels = []
    class_names = sorted(os.listdir(folder_path))  # Assuming folder names are the class labels

    for class_name in class_names:
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            for file_name in os.listdir(class_folder):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(class_folder, file_name)
                    audio, _ = librosa.load(file_path, sr=sample_rate)

                    # Ensure the audio is 1 second long
                    if len(audio) > sample_rate:
                        audio = audio[:sample_rate]
                    elif len(audio) < sample_rate:
                        padding = sample_rate - len(audio)
                        audio = np.pad(audio, (0, padding), 'constant')

                    # Normalize the audio
                    audio = audio / np.max(np.abs(audio))

                    audio_files.append(audio)
                    labels.append(class_name)

    return np.array(audio_files), np.array(labels)



train_dir = 'speech_commands_v0.02'
audios, labels = load_audio_files_from_folder(train_dir)
print("Audio data shape:", audios.shape)
print("Labels shape:", labels.shape)
print('Data loaded')

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

print("Class names:", label_encoder.classes_)

print("Labels encoded")

train_audios, val_audios, train_labels, val_labels = train_test_split(
    audios, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

print(f"Training set size: {train_audios.shape}")
print(f"Validation set size: {val_audios.shape}")


train_dataset = tf.data.Dataset.from_tensor_slices((train_audios, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_audios, val_labels))

BATCH_SIZE = 64
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

print("Converted to tensorflow dataset")


model = models.Sequential([
    layers.Input(shape=(16000,)),
    layers.Reshape((16000, 1)),

    # First convolutional layer
    layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=4),

    # Second convolutional layer
    layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=4),

    # Third convolutional layer
    layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=4),

    # Flattening
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    # Output layer
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.summary()

print("Model constructed")


EPOCHS = 10

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset
)
print("Model trained")


model.save('keyword_recognition_cnn_2.h5')
print("Model saved successfully!")

