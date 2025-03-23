import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import h5py
import glob
from sklearn.preprocessing import LabelEncoder
from utils import config

class SpectrogramDataset(Dataset):
    def __init__(self, h5_folder_path="data/processed/*.h5"):
        """Lazy loading of spectrogram data from HDF5 files with multi-label encoding."""
        self.h5_files = sorted(glob.glob(h5_folder_path))  # Get all H5 files
        self.label_encoder = LabelEncoder()
        self.label_classes = np.array(["none", "coughing", "clearingthroat", "smack", "stomach"])
        self.label_encoder.classes_ = self.label_classes  # Ensure known classes
        self.data = self._load_data()
        self.index_map = self._build_index_map()  # Precompute index lookup

    def _load_data(self):
        """Loads all data references without actually loading the dataset into memory."""
        data_refs = {}
        for h5_file in self.h5_files:
            with h5py.File(h5_file, "r") as hdf_file:
                book_name = list(hdf_file.keys())[0]
                book_group = hdf_file[book_name]
                if book_name not in data_refs:
                    data_refs[book_name] = []
                for chunk_name in book_group.keys():
                    data_refs[book_name].append((h5_file, book_name, chunk_name))
        return data_refs

    def _build_index_map(self):
        """Precomputes a lookup table mapping global indices to (book_name, chunk_index)."""
        index_map = []
        for book_name, chunks in self.data.items():
            for chunk_index in range(len(chunks)):
                index_map.append((book_name, chunk_index))
        return index_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert global index to per-book index
        book_name, chunk_index = self.index_map[idx]
        h5_file, _, chunk_name = self.data[book_name][chunk_index]

        with h5py.File(h5_file, "r") as hdf_file:
            chunk_data = hdf_file[book_name][chunk_name][()]
            label_data = hdf_file[book_name][chunk_name].attrs.get("label", "none")
            individual_labels = label_data.split(",")  # Handle multiple labels
            integer_labels = self.label_encoder.transform(individual_labels)
            one_hot_vectors = np.eye(len(self.label_classes))[integer_labels]
            combined_one_hot = np.max(one_hot_vectors, axis=0)

        return torch.tensor(chunk_data, dtype=torch.float32), torch.tensor(combined_one_hot, dtype=torch.float32)


def prepare_dataloaders(batch_size=8, min_split_ratio=None):
    """Creates PyTorch DataLoaders for training and evaluation."""

    params = config.Params()
    if min_split_ratio is None:
        min_split_ratio = params['preprocess']['min_split_ratio']

    dataset = SpectrogramDataset()
    dataset_size = len(dataset)

    books = list(dataset.data.keys())
    np.random.shuffle(books)
    print("Book order:", books)
    train_books = []
    eval_books = []
    train_size = 0

    for book in books:
        if train_size / dataset_size < min_split_ratio:
            train_books.append(book)
            train_size += len(dataset.data[book])
        else:
            eval_books.append(book)

    # train_size = int(split_ratio * dataset_size)
    # eval_size = dataset_size - train_size

    # train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

    # Create index lists for Subset
    train_indices = []
    eval_indices = []
    current_index = 0  # Keeps track of the global index in the dataset

    for book in books:
        num_chunks = len(dataset.data[book])
        
        if book in train_books:
            for i in range(num_chunks):
                train_indices.append(current_index + i)
        else:
            for i in range(num_chunks):
                eval_indices.append(current_index + i)
        
        # Move global index forward by the number of chunks in this book
        current_index += num_chunks


    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, eval_loader