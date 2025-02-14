import numpy as np
import h5py
import tensorflow as tf
import glob
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

def load_spectrogram_data(h5_folder_path="data/test/*.h5"):
    """Load spectrogram data from multiple H5 files.

    Args:
        h5_folder_path (str): Path to H5 files.

    Returns:
        tf.data.Dataset: Spectrogram dataset.
        dict: Book start indices.
        dict: Book end indices.
        dict: Chapter start indices.
        dict: Chapter end indices.
        list: Raw labels.
    """
    dataset = []
    book_start = {}
    book_end = {}
    chapter_start = {}
    chapter_end = {}
    labels = []

    h5_files = sorted(glob.glob(h5_folder_path))

    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as hdf_file:
            book_name = list(hdf_file.keys())[0]  
            book_group = hdf_file[book_name]

            book_start[book_name] = len(dataset)
            book_labels = []
            book_chunks = []
            chunk_numbers = []

            for chunk_name in book_group.keys():
                chunk_name_no_ext = os.path.splitext(chunk_name)[0]
                chunk_parts = chunk_name_no_ext.split("_")

                if chunk_parts[-1].isdigit():
                    chunk_number = int(chunk_parts[-1])
                else:
                    continue  

                chunk_data = book_group[chunk_name][()]
                book_chunks.append(chunk_data)
                chunk_numbers.append(chunk_number)

                label_data = book_group[chunk_name].attrs.get("label", "none")
                book_labels.append(label_data)

                chapter_name = "_".join(chunk_parts[:-1])  
                if chapter_name not in chapter_start:
                    chapter_start[chapter_name] = len(dataset) + len(book_chunks) - 1
                chapter_end[chapter_name] = len(dataset) + len(book_chunks) - 1

            sorted_data = sorted(zip(chunk_numbers, book_chunks, book_labels), key=lambda x: x[0])
            sorted_chunks = [x[1] for x in sorted_data]  
            sorted_labels = [x[2] for x in sorted_data]  

            dataset.extend(sorted_chunks)
            labels.extend(sorted_labels)
            book_end[book_name] = len(dataset) - 1  

    spectrogram_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    return spectrogram_dataset, book_start, book_end, chapter_start, chapter_end, labels
