import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import glob
from sklearn.preprocessing import LabelEncoder

class SpectrogramDataset(Dataset):
    def __init__(self, h5_folder_path="data/processed/*.h5"):
        """Lazy loading of spectrogram data from HDF5 files with multi-label encoding."""
        self.h5_files = sorted(glob.glob(h5_folder_path))  # Get all H5 files
        self.label_encoder = LabelEncoder()
        self.label_classes = np.array(["none", "coughing", "clearingthroat", "smack", "stomach"])
        self.label_encoder.classes_ = self.label_classes  # Ensure known classes
        self.data = self._load_data()

    def _load_data(self):
        """Loads all data references without actually loading the dataset into memory."""
        data_refs = []
        for h5_file in self.h5_files:
            with h5py.File(h5_file, "r") as hdf_file:
                book_name = list(hdf_file.keys())[0]
                book_group = hdf_file[book_name]
                for chunk_name in book_group.keys():
                    data_refs.append((h5_file, book_name, chunk_name))
        return data_refs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        h5_file, book_name, chunk_name = self.data[idx]
        with h5py.File(h5_file, "r") as hdf_file:
            chunk_data = hdf_file[book_name][chunk_name][()]
            label_data = hdf_file[book_name][chunk_name].attrs.get("label", "none")
            individual_labels = label_data.split(",")  # Handle multiple labels
            integer_labels = self.label_encoder.transform(individual_labels)
            one_hot_vectors = np.eye(len(self.label_classes))[integer_labels]
            combined_one_hot = np.max(one_hot_vectors, axis=0)
        return torch.tensor(chunk_data, dtype=torch.float32), torch.tensor(combined_one_hot, dtype=torch.float32)


def prepare_dataloaders(batch_size=8, split_ratio=0.8):
    """Creates PyTorch DataLoaders for training and evaluation."""
    dataset = SpectrogramDataset()
    dataset_size = len(dataset)
    train_size = int(split_ratio * dataset_size)
    eval_size = dataset_size - train_size

    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, eval_loader