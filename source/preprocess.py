import numpy as np
import h5py
import tensorflow as tf
import glob
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

# === Load Spectrogram Data from Multiple HDF5 Files ===
def load_spectrogram_data(h5_folder_path="../data/processed/*.h5"):
    """Load spectrogram data from multiple H5 files."""
    dataset = []
    labels = []

    h5_files = sorted(glob.glob(h5_folder_path))  # Get all H5 files

    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as hdf_file:
            book_name = list(hdf_file.keys())[0]  
            book_group = hdf_file[book_name]

            for chunk_name in book_group.keys():
                chunk_data = book_group[chunk_name][()]
                dataset.append(chunk_data)

                # Extract label from chunk attribute
                label_data = book_group[chunk_name].attrs.get("label", "none")
                labels.append(label_data)

    # Convert to TensorFlow dataset
    spectrogram_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    return spectrogram_dataset, labels


# === Convert Labels to One-Hot Encoding ===
def encode_labels(labels):
    """Convert raw label data to one-hot encoded tensors."""
    label_encoder = LabelEncoder()
    label_encoder.fit(["none", "coughing", "clearingthroat", "smack", "stomach"])

    def encode_fn(label):
        label = label.numpy().decode("utf-8")  
        integer_labels = label_encoder.transform([label])  
        one_hot_vector = tf.one_hot(integer_labels, depth=len(label_encoder.classes_))
        return tf.squeeze(one_hot_vector)

    # Convert labels into tf.data.Dataset
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels).map(
        lambda x: tf.py_function(encode_fn, [x], tf.float32)
    )

    return labels_dataset, label_encoder.classes_


# === Create Train & Eval Datasets ===
def prepare_datasets(spectrogram_dataset, labels_dataset, batch_size=32, split_ratio=0.8):
    """Create Dataset objects from spectrogram and labels tensors."""
    dataset_complete = tf.data.Dataset.zip((spectrogram_dataset, labels_dataset))

    dataset_size = sum(1 for _ in dataset_complete)  
    train_size = int(split_ratio * dataset_size)

    dataset_complete = dataset_complete.shuffle(buffer_size=4000).cache()

    train_dataset = dataset_complete.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    eval_dataset = dataset_complete.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, eval_dataset
