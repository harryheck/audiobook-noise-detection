import torch
import torchinfo
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from preprocess import prepare_dataloaders
from model import AudioCNN
from utils import config
from utils import logs
from pathlib import Path
import time
from tqdm import tqdm


def get_class_weights(dataset, device):
    """
    Compute class weights based on dataset distribution.
    """
    labels = []
    for _, y in dataset:
        labels.append(np.argmax(y.numpy(), axis=-1))  # Convert one-hot to class indices

    unique_classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=labels)
    
    return torch.tensor(weights, dtype=torch.float32, device=device)

def compute_metrics(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    f1_scores = f1_score(y_true, y_pred, average=None)
    f1_none = f1_scores[0]
    f1_coughing = f1_scores[1]
    f1_clearingthroat = f1_scores[2]
    f1_smack = f1_scores[3]
    f1_stomach = f1_scores[4]
    
    return acc, f1_none, f1_coughing, f1_clearingthroat, f1_smack, f1_stomach


def train_epoch(dataloader, model, loss_fn, optimizer, device, writer, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    all_preds, all_labels = [], []

    model.train()
    # for batch, (X, y) in enumerate(tqdm(dataloader)):     # for more verbosity
    for batch, (X, y) in enumerate(dataloader):             # for less verbosity
        X, y = X.to(device), y.to(device)
        X = X.unsqueeze(1)  # Add channel dimension
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        writer.add_scalar("Batch_Loss/train", loss.item(), batch + epoch * len(dataloader))
        train_loss += loss.item()

        all_preds.append(pred.cpu().detach().numpy())
        all_labels.append(y.cpu().numpy())

        if batch % 100 == 0:
            loss_value = loss.item()
            current = (batch + 1) * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /=  num_batches

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc, f1_none, f1_coughing, f1_clearingthroat, f1_smack, f1_stomach = compute_metrics(all_labels, all_preds)
    f1_avg = np.mean([f1_none, f1_coughing, f1_clearingthroat, f1_smack, f1_stomach], axis=0)

    writer.add_scalar("Accuracy/train", acc, epoch)
    writer.add_scalar("F1_avg/train", f1_avg, epoch)
    writer.add_scalar("F1_none/train", f1_none, epoch)
    writer.add_scalar("F1_coughing/train", f1_coughing, epoch)
    writer.add_scalar("F1_clearingthroat/train", f1_clearingthroat, epoch)
    writer.add_scalar("F1_smack/train", f1_smack, epoch)
    writer.add_scalar("F1_stomach/train", f1_stomach, epoch)

    print(f"Train Error: \n Avg loss: {train_loss:>8f} \n")
    print(f"Accuracy: {acc:>8f} \n")
    print(f"F1 Average: {f1_avg:>8f}")
    print(f"F1 None: {f1_none:>8f}")
    print(f"F1 Coughing: {f1_coughing:>8f}")
    print(f"F1 Clearing Throat: {f1_clearingthroat:>8f}")
    print(f"F1 Smack: {f1_smack:>8f}")
    print(f"F1 Stomach: {f1_stomach:>8f} \n")

    return train_loss, f1_avg

def test_epoch(dataloader, model, loss_fn, device, writer, epoch):
    num_batches = len(dataloader)
    test_loss = 0
    all_preds, all_labels = [], []

    model.eval()
    with torch.no_grad():
        # for X, y in tqdm(dataloader):     # for more verbosity
        for X, y in dataloader:             # for less verbosity
            X, y = X.to(device), y.to(device)
            X = X.unsqueeze(1)  # Add channel dimension
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    test_loss /= num_batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc, f1_none, f1_coughing, f1_clearingthroat, f1_smack, f1_stomach = compute_metrics(all_labels, all_preds)
    f1_avg = np.mean([f1_none, f1_coughing, f1_clearingthroat, f1_smack, f1_stomach], axis=0)
    
    writer.add_scalar("Accuracy/test", acc, epoch)
    writer.add_scalar("F1_avg/test", f1_avg, epoch)
    writer.add_scalar("F1_none/test", f1_none, epoch)
    writer.add_scalar("F1_coughing/test", f1_coughing, epoch)
    writer.add_scalar("F1_clearingthroat/test", f1_clearingthroat, epoch)
    writer.add_scalar("F1_smack/test", f1_smack, epoch)
    writer.add_scalar("F1_stomach/test", f1_stomach, epoch)

    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    print(f"Accuracy: {acc:>8f} \n")
    print(f"F1 Average: {f1_avg:>8f}")
    print(f"F1 None: {f1_none:>8f}")
    print(f"F1 Coughing: {f1_coughing:>8f}")
    print(f"F1 Clearing Throat: {f1_clearingthroat:>8f}")
    print(f"F1 Smack: {f1_smack:>8f}")
    print(f"F1 Stomach: {f1_stomach:>8f} \n")

    return test_loss, f1_avg

def main():
    params = config.Params()
    random_seed = params['general']['random_seed']
    epochs = params['train']['epochs']
    learning_rate = params['train']['learning_rate']
    batch_size = params['train']['batch_size']
    device_request = params['train']['device_request']
    dilation = params['model']['conv2d_dilation']
    patience = params['train']['early_stopping_patience'] # Number of epochs to wait before stopping
 
    counter = 0
    best_f1_score = 0.0  # Track the best avg F1 score
    last_model = None
    
    # Create a SummaryWriter object to write the tensorboard logs
    tensorboard_path = logs.return_tensorboard_path() # for remote logging
    # tensorboard_path = Path("_logs") / time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) # for local running
    metrics = {'Epoch_Loss/train': None, 'Epoch_Loss/test': None, 'Batch_Loss/train': None,
               'Accuracy/train': None, 'Accuracy/test': None, 'Batch_Accuracy/train': None,
               'F1_avg/train': None, 'F1_avg/test': None,
               'F1_none/train': None, 'F1_none/test': None,
               'F1_coughing/train': None, 'F1_coughing/test': None,
               'F1_clearingthroat/train': None, 'F1_clearingthroat/test': None,
               'F1_smack/train': None, 'F1_smack/test': None,
               'F1_stomach/train': None, 'F1_stomach/test': None}
    
    writer = logs.CustomSummaryWriter(log_dir=tensorboard_path, params=params, metrics=metrics) # for remote logging
    # writer = logs.CustomSummaryWriter(log_dir=tensorboard_path, params=params, metrics=metrics, sync_interval=0) # for local running

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

    # Compute class weights
    print("Computing class weights...")
    class_weights = get_class_weights(train_dataloader.dataset, device)
    print("Class weights computed.")

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
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        epoch_loss_train, avg_train_f1_score = train_epoch(train_dataloader, model, loss_fn, optimizer, device, writer, epoch=t)
        epoch_loss_test, avg_test_f1_score = test_epoch(eval_dataloader, model, loss_fn, device, writer, epoch=t)
        writer.add_scalar("Epoch_Loss/train", epoch_loss_train, t)
        writer.add_scalar("Epoch_Loss/test", epoch_loss_test, t)     
        writer.step()


        # Early Stopping Check
        if avg_test_f1_score > best_f1_score:
            best_f1_score = avg_test_f1_score  # Update best score
            counter = 0  # Reset patience counter
            print(f"New best F1 score: {best_f1_score:.4f}, saving model.")

            # Save the best model
            modelname = time.strftime("model_%Y_%m_%d-%H_%M_%S") + ".pth"
            output_file_path = Path('models/checkpoints/' + modelname)
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_file_path)
            print(f"Saved PyTorch Model State to {modelname}")

            # delete previous best model
            if last_model is not None:
                last_model.unlink(missing_ok=True)
            last_model = output_file_path

        else:
            counter += 1
            print(f"No improvement in F1 score for {counter} epochs.")

        if counter >= patience:
            print("Early stopping triggered. Training stopped.")
            break  # Stop training  

    writer.close()

    print("Done with the training stage!")


if __name__ == "__main__":
    main()