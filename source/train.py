import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from preprocess import prepare_datasets
from model import build_model, compile_model
from utils import config
from utils import logs
import os
import time


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


def train_model(model, train_dataset, eval_dataset, epochs=10, callbacks=None):
    class_weights = get_class_weights(train_dataset)
    history = model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )
    return history

def main():
    gpus = tf.config.list_physical_devices('GPU')
    print("Checking available GPUs:", gpus)
    params = config.Params()
    random_seed = params['general']['random_seed']
    epochs = params['train']['epochs']
    learning_rate = params['train']['learning_rate']
    batch_size = params['train']['batch_size']
    dilation = params['model']['conv2d_dilation']


    config.set_random_seeds(random_seed)

    print("Preparing datasets...")
    train_dataset, eval_dataset = prepare_datasets(batch_size=batch_size)
    print("Datasets prepared.")

    input_shape, output_length = None, None
    for batch_data, batch_labels in train_dataset.take(1):
        input_shape = batch_data.shape[1:]
        output_length = batch_labels.shape[1]
        print("Batch Data Shape:", batch_data.shape)
        print("Batch Labels Shape:", batch_labels.shape)

    if input_shape is None or output_length is None:
        raise ValueError("Error: input_shape or output_length is None! Dataset may be empty.")

    print("Building model...")
    model = build_model(input_shape, output_length, dilation)
    model = compile_model(model, learning_rate)
    print("Model built.")
    model.summary()

    # uncomment for plotting of model architecture
    # plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=False, rankdir="LR", show_layer_activations=True)
    # return

    # Tensorboard logs
    def get_run_logdir():
        base_logdir = os.getenv("TENSORBOARD_LOGDIR", default="logs/tensorboard")
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(base_logdir, run_id)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir(), histogram_freq=1)

    print("Starting training...")
    train_model(model, train_dataset, eval_dataset, epochs, [tensorboard_cb])
    print("Training completed successfully.")

    modelname = time.strftime("model_%Y_%m_%d-%H_%M_%S") + ".keras"
    modelpath = "models"
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
    model.save(os.path.join("models", modelname))
    print("Model saved as", modelname)

if __name__ == "__main__":
    main()