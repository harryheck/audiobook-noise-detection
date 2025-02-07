import numpy as np
import h5py
import tensorflow as tf
import glob
import csv
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from utils import config

def load_spectrogram_data(h5_file_path="dataset.h5"):
    """load the spectrogram data from the h5 file

    Args:
        h5_file_path (str, optional): file path of h5 file. Defaults to "dataset.h5".

    Returns:
        tf_tensor: spectrogram tensor
        dict: start index of each book
        dict: end index of each book
    """

    dataset = []
    book_start = {} # for keeping track of the start of each book in the dataset
    book_end = {} # for keeping track of the end of each book in the dataset

    # read dataset
    with h5py.File(h5_file_path, "r") as hdf_file:
        # sort books by name
        hdf_file = dict(sorted(hdf_file.items()))
        # get all books from the dataset
        for book_name in hdf_file.keys():
            book = hdf_file[book_name]

            # get start of the book
            book_start[book_name] = (len(dataset))

            # get all spectrograms from the book
            book_dataset = []
            book_dataset_numbering = [] # for sorting the dataset
            for key in book.keys():
                book_dataset.append(book[key][:])
                book_dataset_numbering.append(int(key))

            # sort spectrograms by key
            book_dataset = [x for _, x in sorted(zip(book_dataset_numbering, book_dataset))]
            dataset.extend(book_dataset)

            # get end of the book
            book_end[book_name] = len(dataset)-1

    # convert dataset to tensor
    spectrogram_tensor = tf.convert_to_tensor(dataset, dtype=tf.float32)

    print(f"Data successfully loaded and converted to tensor with size {spectrogram_tensor.shape}")
    return spectrogram_tensor, book_start, book_end


def load_labels(label_folder='dataset/*/*.csv'):
    """load the labels from the csv files

    Args:
        label_folder (str, optional): filepath to all csv files in dataset folder. Defaults to 'dataset/*/*.csv'.

    Returns:
        tf_tensor: labels tensor
        np.array: classes of labels
        dict: start index of each book
        dict: end index of each book
    """
    labels = []
    labels_book_start = {} # for keeping track of the start of each book in the labels
    labels_book_end = {} # for keeping track of the end of each book in the labels

    # get all csv files in dataset
    csv_files = glob.glob(label_folder)

    # sort csv files by book name
    csv_files = sorted(csv_files)

    # read labels from csv files
    for csv_file in csv_files:
        # get start of the book
        labels_book_start[os.path.basename(csv_file)] = (len(labels))
        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip the headers
            for row in reader:
                labels.append(row[1])
                
        # get end of the book
        labels_book_end[os.path.basename(csv_file)] = len(labels)-1

    # convert labels to one hot encoding

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(['none', 'coughing', 'clearingthroat', 'smack', 'stomach'])
    one_hot_labels = []
    for label in labels:
        # Split multi-label entries and encode each label
        individual_labels = label.split(',')  # Split by comma
        integer_labels = label_encoder.transform(individual_labels)  # Encode to integers
        one_hot_vectors = keras.utils.to_categorical(integer_labels, num_classes=len(label_encoder.classes_))  # One-hot encode

        # Combine one-hot vectors by taking element-wise maximum
        combined_one_hot = np.max(one_hot_vectors, axis=0)
        one_hot_labels.append(combined_one_hot)

    # convert labels to tensor
    labels_tensor = tf.convert_to_tensor(one_hot_labels, dtype=tf.float32)
    print("Label encoding (one-hot):")
    for i, v in enumerate(label_encoder.classes_):
        print(i, v)
    print()

    print(f"Labels successfully loaded and converted to tensor with size {labels_tensor.shape}")

    return labels_tensor, label_encoder.classes_, labels_book_start, labels_book_end


def prepare_datasets(spectrogram_tensor, labels_tensor, batch_size=32, split_ratio=0.8):
    """Create Dataset objects from spectrogram and labels tensors

    Args:
        spectrogram_tensor (tf_tensor): spectrogram tensor
        labels_tensor (tf_tensor): labels tensor
        batch_size (int, optional): batch size. Defaults to 32.
        split_ratio (float, optional): ratio of train to eval set. Defaults to 0.8.

    Returns:
        tf_tensor: complete training dataset
        tf_tensor: complete eval dataset
    """
    spectrogram_tensor = tf.expand_dims(spectrogram_tensor, axis=-1) # add channel dimension for CNN
    dataset_complete = tf.data.Dataset.from_tensor_slices((spectrogram_tensor, labels_tensor))


    # # Print dataset for verification
    # for data, label in dataset_complete.take(3):  # Preview the first 3 samples
    #     plt.imshow(data.numpy(), origin='lower')
    #     plt.show()
    #     print("Label:", label.numpy())


    # Shuffle and batch the dataset
    dataset_size = labels_tensor.shape[0]
    train_size = int(split_ratio * dataset_size)

    dataset_complete = dataset_complete.shuffle(buffer_size=4000)  # Shuffle with a buffer size
    dataset_complete = dataset_complete.cache()

    print(f"Dataset successfully created, shuffled and batched with batch size {batch_size}")
    print(f"Dataset size: {dataset_size}")

    # take training set from dataset
    train_dataset = dataset_complete.take(train_size)
    train_dataset = train_dataset.shuffle(buffer_size=4000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    print(f"Training set successfully created with size {train_size}")

    # take eval set samples from dataset 
    eval_dataset = dataset_complete.skip(train_size).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    print(f"Eval set successfully created with size {dataset_size-train_size}")

    return train_dataset, eval_dataset
