import numpy as np
import h5py
import tensorflow as tf
import glob
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

def load_spectrogram_generator(h5_folder_path="data/processed/*.h5"):
    """Load spectrogram data lazily from HDF5 files and properly handle multi-label encoding."""
    
    h5_files = sorted(glob.glob(h5_folder_path))  # Get all H5 files
    label_encoder = LabelEncoder()
    label_classes = np.array(["none", "coughing", "clearingthroat", "smack", "stomach"])
    label_encoder.classes_ = label_classes  # Ensure known classes

    def generator():
        for h5_file in h5_files:
            with h5py.File(h5_file, "r") as hdf_file:
                book_name = list(hdf_file.keys())[0]
                book_group = hdf_file[book_name]

                for chunk_name in book_group.keys():
                    chunk_data = book_group[chunk_name][()]

                    # Handle multiple labels
                    label_data = book_group[chunk_name].attrs.get("label", "none")
                    individual_labels = label_data.split(",")  # Split multi-labels

                    # Convert labels to one-hot encoding and merge
                    integer_labels = label_encoder.transform(individual_labels)
                    one_hot_vectors = keras.utils.to_categorical(integer_labels, num_classes=len(label_classes))

                    # Merge multiple one-hot encodings (logical OR)
                    combined_one_hot = np.max(one_hot_vectors, axis=0)

                    yield chunk_data, combined_one_hot

    return tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(64, 345), dtype=tf.float32),  # Spectrogram shape
        tf.TensorSpec(shape=(5,), dtype=tf.float32)  # One-hot encoded multi-label
    ))

def prepare_datasets(batch_size=8, split_ratio=0.8):
    """Create a dataset pipeline without preloading everything into memory."""
    dataset = load_spectrogram_generator()

    dataset_size = sum(1 for _ in dataset)  # Count dataset size
    train_size = int(split_ratio * dataset_size)

    dataset = dataset.shuffle(buffer_size=500)

    train_dataset = dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    eval_dataset = dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, eval_dataset

