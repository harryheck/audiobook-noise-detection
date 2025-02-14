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
        dict: Start index of each book
        dict: End index of each book
        dict: Start index of each chapter
        dict: End index of each chapter
        list: Labels (raw)
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
            book_name = list(hdf_file.keys())[0]  # Only one group per file (book)
            book_group = hdf_file[book_name]

            # Book Start Index
            book_start[book_name] = len(dataset)
            book_labels = []  
            book_chunks = []
            chunk_numbers = []

            for chunk_name in book_group.keys():
                chunk_name_no_ext = os.path.splitext(chunk_name)[0]
                chunk_parts = chunk_name_no_ext.split("_")

                # Extract chunk number safely
                if chunk_parts[-1].isdigit():
                    chunk_number = int(chunk_parts[-1])
                else:
                    continue  

                chunk_data = book_group[chunk_name][()]
                book_chunks.append(chunk_data)
                chunk_numbers.append(chunk_number)

                # Extract label from chunk attribute
                label_data = book_group[chunk_name].attrs.get("label", "none")  
                book_labels.append(label_data)

                # Track chapter start and end
                chapter_name = "_".join(chunk_parts[:-1])  
                if chapter_name not in chapter_start:
                    chapter_start[chapter_name] = len(dataset) + len(book_chunks) - 1
                chapter_end[chapter_name] = len(dataset) + len(book_chunks) - 1

            # Sort chunks in order
            sorted_data = sorted(zip(chunk_numbers, book_chunks, book_labels), key=lambda x: x[0])
            sorted_chunks = [x[1] for x in sorted_data]  
            sorted_labels = [x[2] for x in sorted_data]  

            dataset.extend(sorted_chunks)
            labels.extend(sorted_labels)

            # Book End Index
            book_end[book_name] = len(dataset) - 1

    # Convert dataset to tf.data.Dataset
    spectrogram_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    return spectrogram_dataset, book_start, book_end, chapter_start, chapter_end, labels


# === Convert Labels to One-Hot Encoding ===
def encode_labels(labels):
    """Convert raw label data to one-hot encoded tensors.

    Args:
        labels (list): Raw labels list.

    Returns:
        tf.data.Dataset: One-hot encoded labels dataset.
        np.array: Label classes.
    """
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(["none", "coughing", "clearingthroat", "smack", "stomach"])

    def process_label(label):
        """Encodes a label string into a one-hot vector."""
        individual_labels = tf.strings.split(label, ",")  
        integer_labels = label_encoder.transform(individual_labels.numpy())  
        one_hot_vectors = tf.one_hot(integer_labels, depth=len(label_encoder.classes_))
        combined_one_hot = tf.reduce_max(one_hot_vectors, axis=0)  
        return combined_one_hot

    # Convert labels to a dataset
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels).map(lambda x: tf.py_function(process_label, [x], tf.float32))

    return labels_dataset, label_encoder.classes_


# === Create Train & Eval Datasets ===
def prepare_datasets(spectrogram_dataset, labels_dataset, batch_size=32, split_ratio=0.8):
    """Create Dataset objects from spectrogram and labels tensors.

    Args:
        spectrogram_dataset (tf.data.Dataset): spectrogram dataset
        labels_dataset (tf.data.Dataset): labels dataset
        batch_size (int, optional): batch size. Defaults to 32.
        split_ratio (float, optional): ratio of train to eval set. Defaults to 0.8.

    Returns:
        tf.data.Dataset: Training dataset
        tf.data.Dataset: Evaluation dataset
    """
    dataset_complete = tf.data.Dataset.zip((spectrogram_dataset, labels_dataset))

    dataset_size = sum(1 for _ in dataset_complete)  
    train_size = int(split_ratio * dataset_size)

    dataset_complete = dataset_complete.shuffle(buffer_size=4000).cache()

    train_dataset = dataset_complete.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    eval_dataset = dataset_complete.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, eval_dataset
