import numpy as np
import h5py
import tensorflow as tf
import glob
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

# === 🚀 Generator Function for Data Streaming ===
def data_generator(h5_files):
    """Generator function to yield spectrograms and labels one at a time."""
    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as hdf_file:
            book_name = list(hdf_file.keys())[0]  # Assume one group per file (book)
            book_group = hdf_file[book_name]

            for chunk_name in book_group.keys():
                chunk_data = book_group[chunk_name][()]
                label_data = book_group[chunk_name].attrs.get("label", "none")  # Default label
                yield chunk_data, label_data  # ✅ Streams one spectrogram & label at a time

# === 🚀 Load Spectrogram Data as Streaming Dataset ===
def load_spectrogram_data(h5_folder_path="../data/processed/*.h5"):
    """Loads spectrograms as a streaming dataset from multiple H5 files."""

    h5_files = sorted(glob.glob(h5_folder_path))
    
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(h5_files),
        output_signature=(
            tf.TensorSpec(shape=(64, 345), dtype=tf.float32),  # Adjust shape as needed
            tf.TensorSpec(shape=(), dtype=tf.string),  # Labels as string
        )
    )
    
    return dataset

# === 🚀 Convert Labels to One-Hot Encoding (Efficiently) ===
def encode_labels(dataset):
    """Converts labels in the dataset to one-hot encoding lazily."""

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(["none", "coughing", "clearingthroat", "smack", "stomach"])

    def one_hot_encode(label):
        """Encodes a single label into one-hot format."""
        integer_label = label_encoder.transform([label])[0]  # Convert label to integer
        one_hot_vector = keras.utils.to_categorical(integer_label, num_classes=len(label_encoder.classes_))
        return one_hot_vector

    # Map labels on-the-fly without loading everything into memory
    dataset = dataset.map(lambda x, y: (x, tf.numpy_function(one_hot_encode, [y], tf.float32)))

    return dataset

# === 🚀 Create Streaming Train & Eval Datasets ===
def prepare_datasets(dataset, batch_size=32, split_ratio=0.8):
    """Prepares training and evaluation datasets using a streaming pipeline."""

    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=1000).cache()
    
    # Compute dataset sizes
    dataset_size = sum(1 for _ in dataset)  # Count dataset size lazily
    train_size = int(split_ratio * dataset_size)

    # Split dataset into train & eval sets
    train_dataset = dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    eval_dataset = dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, eval_dataset
