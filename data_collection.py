import sounddevice as sd
import wavio
import os


Function to record audio
def record_audio(filename, duration=2, fs=16000):
    print(f"Recording {filename} for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until the recording is finished
    wavio.write(filename, recording, fs, sampwidth=2)
    print(f"Saved {filename}")


# Function to create directories and record for each label
def create_dataset(labels, num_samples_per_label=10, duration=2, dataset_dir="my_custom_dataset"):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    for label in labels:
        label_dir = os.path.join(dataset_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        for i in range(num_samples_per_label):
            filename = os.path.join(label_dir, f"{label}_sample_{i + 1}.wav")
            record_audio(filename, duration=duration)
            print(f"Recorded and saved {filename}")


# List of labels/words you want to record
labels = ['_background_noise_','backward','bed','bird','cat','dog','down','eight','five','follow', 'forward', 'four' ,'go', 'happy', 'house', 'learn', 'left','marvin' ,'nine' ,'no', 'off', 'on' ,'one', 'right', 'seven' ,'sheila', 'six','stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
create_dataset(labels, num_samples_per_label=30)  # Record 5 samples per word

import librosa
import numpy as np

def load_audio_files_from_folder(folder_path, sample_rate=16000):
    audio_files = []
    labels = []
    class_names = sorted(os.listdir(folder_path))

    for class_name in class_names:
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            for file_name in os.listdir(class_folder):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(class_folder, file_name)
                    audio, _ = librosa.load(file_path, sr=sample_rate)

                    # Ensure audio is 1 second long
                    if len(audio) > sample_rate:
                        audio = audio[:sample_rate]
                    elif len(audio) < sample_rate:
                        padding = sample_rate - len(audio)
                        audio = np.pad(audio, (0, padding), 'constant')

                    # Normalize audio
                    audio = audio / np.max(np.abs(audio))

                    audio_files.append(audio)
                    labels.append(class_name)

    return np.array(audio_files), np.array(labels)

# Load the custom dataset
dataset_dir = 'my_custom_dataset'
audios, labels = load_audio_files_from_folder(dataset_dir)

print("Audio data shape:", audios.shape)
print("Labels shape:", labels.shape)
print('Pre processing done')

from sklearn.preprocessing import LabelEncoder

# Encode the labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Check the label encoding
print("Class names:", label_encoder.classes_)
print('Labels encoded')

from sklearn.model_selection import train_test_split

train_audios, val_audios, train_labels, val_labels = train_test_split(
    audios, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

print(f"Training set size: {train_audios.shape}")
print(f"Validation set size: {val_audios.shape}")
print('splitted data')

import tensorflow as tf
from tensorflow.keras import layers, models

# Load the pretrained model
pretrained_model = tf.keras.models.load_model('keyword_recognition_cnn.h5')

# Freeze the layers except the last one
for layer in pretrained_model.layers[:-2]:
    layer.trainable = False

# Replace the final layers for your custom dataset
model = models.Sequential(pretrained_model.layers[:-2])  # Copy all except the last few layers

# Add new layers with unique names
model.add(layers.Flatten(name='new_flatten'))  # Give the flatten layer a new name
model.add(layers.Dense(128, activation='relu', name='new_dense_1'))  # New dense layer with unique name
model.add(layers.Dropout(0.3, name='new_dropout'))
model.add(layers.Dense(len(label_encoder.classes_), activation='softmax', name='new_output'))  # Final output layer

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

print('model trained')

# Convert the numpy arrays to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_audios, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_audios, val_labels))

# Batch and shuffle the datasets
BATCH_SIZE = 32
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

# Fine-tune the model
EPOCHS = 100

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset
)
print('Model trained')

# Save the fine-tuned model
model.save('keyword_recognition_finetuned_model.h5')
print("Model fine-tuned and saved successfully!")
