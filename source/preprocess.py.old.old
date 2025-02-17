import numpy as np
import h5py
import tensorflow as tf
import glob
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

# === Load Spectrogram Data from Multiple HDF5 Files ===
def load_spectrogram_data(h5_folder_path="../data/processed/*.h5"):
    """Load spectrogram data from multiple H5 files.

    Args:
        h5_folder_path (str, optional): Path to H5 files. Defaults to "../data/processed/*.h5".

    Returns:
        tf.data.Dataset: Spectrogram dataset
        dict: Book start indices
        dict: Book end indices
        dict: Chapter start indices
        dict: Chapter end indices
        list: Raw labels
    """
    dataset = []
    book_start = {}
    book_end = {}
    chapter_start = {}
    chapter_end = {}
    labels = []

    h5_files = sorted(glob.glob(h5_folder_path))  # Get all H5 files

    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as hdf_file:
            book_name = list(hdf_file.keys())[0]  # Assume only one book per file
            book_group = hdf_file[book_name]

            # Track book indices
            book_start[book_name] = len(dataset)

            for chunk_name in book_group.keys():
                chunk_data = book_group[chunk_name][()]
                dataset.append(chunk_data)

                # Extract label
                label_data = book_group[chunk_name].attrs.get("label", "none")
                labels.append(label_data)

                # Track chapter indices
                chapter_name = "_".join(chunk_name.split("_")[:-1])  # Remove chunk ID
                if chapter_name not in chapter_start:
                    chapter_start[chapter_name] = len(dataset) - 1
                chapter_end[chapter_name] = len(dataset) - 1

            # Track book end index
            book_end[book_name] = len(dataset) - 1

    # Convert dataset to TensorFlow dataset
    spectrogram_dataset = tf.data.Dataset.from_tensor_slices(dataset)

    return spectrogram_dataset, book_start, book_end, chapter_start, chapter_end, labels


# === Convert Labels to One-Hot Encoding ===
def encode_labels(labels):
    """Convert raw label data to one-hot encoded tensors.

    Args:
        labels (list): Raw labels list.

    Returns:
        tf.Tensor: One-hot encoded labels tensor.
        np.array: Label classes.
    """
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(["none", "coughing", "clearingthroat", "smack", "stomach"])

    one_hot_labels = []
    for label in labels:
        individual_labels = label.split(",")  # Handle multiple labels per sample
        integer_labels = label_encoder.transform(individual_labels)
        one_hot_vectors = keras.utils.to_categorical(integer_labels, num_classes=len(label_encoder.classes_))

        # Merge multiple labels into a single one-hot vector
        combined_one_hot = np.max(one_hot_vectors, axis=0)
        one_hot_labels.append(combined_one_hot)

    labels_tensor = tf.convert_to_tensor(one_hot_labels, dtype=tf.float32)
    return labels_tensor, label_encoder.classes_


# === Create Train & Eval Datasets ===
def prepare_datasets(spectrogram_dataset, labels_tensor, batch_size=32, split_ratio=0.8):
    """Create Dataset objects from spectrogram and labels tensors.

    Args:
        spectrogram_dataset (tf.data.Dataset): Spectrogram dataset
        labels_tensor (tf.Tensor): Labels tensor
        batch_size (int, optional): Batch size. Defaults to 32.
        split_ratio (float, optional): Ratio of train to eval set. Defaults to 0.8.

    Returns:
        tf.data.Dataset: Training dataset
        tf.data.Dataset: Evaluation dataset
    """
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels_tensor)

    dataset_complete = tf.data.Dataset.zip((spectrogram_dataset, labels_dataset))

    # Shuffle and batch the dataset
    dataset_size = len(labels_tensor)
    train_size = int(split_ratio * dataset_size)

    dataset_complete = dataset_complete.shuffle(buffer_size=4000).cache()

    # Split into train & eval
    train_dataset = dataset_complete.take(train_size).shuffle(buffer_size=4000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    eval_dataset = dataset_complete.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, eval_dataset
