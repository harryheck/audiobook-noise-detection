import torch
import torchinfo
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from preprocess import prepare_dataloaders
from model import AudioCNN
from utils import config
from utils import logs
from pathlib import Path
import time
from tqdm import tqdm


# IMPLEMENT THIS AFTER TESTING NEW PYTORCH APPROACH
def get_class_weights(dataset):
    """
    Compute class weights based on dataset distribution.
    """
    labels = []
    for _, y in dataset:
        labels.extend(np.argmax(y.numpy(), axis=1))  # Convert one-hot to class indices

    unique_classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=labels)
    
    return {i: w for i, w in enumerate(weights)}


def train_epoch(dataloader, model, loss_fn, optimizer, device, writer, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0 
    model.train()
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        X, y = X.to(device), y.to(device)
        X = X.unsqueeze(1)  # Add channel dimension
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        writer.add_scalar("Batch_Loss/train", loss.item(), batch + epoch * len(dataloader))
        train_loss += loss.item()
        if batch % 100 == 0:
            loss_value = loss.item()
            current = (batch + 1) * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /=  num_batches
    return train_loss

def test_epoch(dataloader, model, loss_fn, device, writer):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            X = X.unsqueeze(1)  # Add channel dimension
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss

def main():
    params = config.Params()
    random_seed = params['general']['random_seed']
    epochs = params['train']['epochs']
    learning_rate = params['train']['learning_rate']
    batch_size = params['train']['batch_size']
    device_request = params['train']['device_request']
    dilation = params['model']['conv2d_dilation']
    
    # Create a SummaryWriter object to write the tensorboard logs
    #tensorboard_path = logs.return_tensorboard_path() # for remote logging
    tensorboard_path = Path('_logs') # for local running
    metrics = {'Epoch_Loss/train': None, 'Epoch_Loss/test': None, 'Batch_Loss/train': None}
    writer = logs.CustomSummaryWriter(log_dir=tensorboard_path, params=params, metrics=metrics, sync_interval=0) # remove sync_interval=0 for remote logging

    config.set_random_seeds(random_seed)

    device = config.prepare_device(device_request)

    if device.type == "cuda":
        print(torch.cuda.get_device_name())
        print(torch.cuda.current_device())
    
    print("Preparing datasets...")
    train_dataloader, eval_dataloader = prepare_dataloaders(batch_size=batch_size)
    print("Datasets prepared.")
    print("Training set size:", len(train_dataloader))
    print("Evaluation set size:", len(eval_dataloader))

    # Get a single batch from the DataLoader
    input_shape, output_length = None, None

    for batch_data, batch_labels in train_dataloader:
        input_shape = batch_data.shape[1:]  # Exclude batch dimension
        output_length = batch_labels.shape[1]  # Number of output classes
        print("Input data Shape:", input_shape)
        print("Labels Shape:", output_length)
        break  # Only need one batch
    

    if input_shape is None or output_length is None:
        raise ValueError("Error: input_shape or output_length is None! Dataset may be empty.")
    
    input_shape = (1, 1) + input_shape  # Add two channel dimensions

    print("Building model...")
    model = AudioCNN(input_shape, output_length, dilation).to(device)
    print("Model built.")
    summary = torchinfo.summary(model, input_size=input_shape, device=device)
    print(summary)

    # Add the model graph to the tensorboard logs
    sample_inputs = torch.randn(input_shape) 
    writer.add_graph(model, sample_inputs.to(device))

    # Define the loss function and the optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        epoch_loss_train = train_epoch(train_dataloader, model, loss_fn, optimizer, device, writer, epoch=t)
        epoch_loss_test = test_epoch(eval_dataloader, model, loss_fn, device, writer)
        writer.add_scalar("Epoch_Loss/train", epoch_loss_train, t)
        writer.add_scalar("Epoch_Loss/test", epoch_loss_test, t)     
        writer.step()  

    writer.close()

    # Save the model checkpoint
    modelname = time.strftime("model_%Y_%m_%d-%H_%M_%S") + ".pth"
    output_file_path = Path('models/checkpoints/' + modelname)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_file_path)
    print(f"Saved PyTorch Model State to {modelname}")

    print("Done with the training stage!")


if __name__ == "__main__":
    main()