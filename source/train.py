import tensorflow as tf
from preprocess import load_spectrogram_data, load_labels, prepare_datasets
from model import build_model, compile_model
from utils import config
import os
import time

def train_model(model, train_dataset, eval_dataset, epochs=10, callbacks=None):
    history = model.fit(train_dataset, 
                        validation_data=eval_dataset, 
                        epochs=epochs,
                        callbacks=callbacks)
    return history

def main():

    # Load config
    params = config.Params()
    random_seed = params['general']['random_seed']
    epochs = params['train']['epochs']
    learning_rate = params['train']['learning_rate']

    # Set a random seed for reproducibility across all devices. Add more devices if needed
    config.set_random_seeds(random_seed)

    # Load and preprocess data
    spectrogram_tensor, _, _ = load_spectrogram_data(os.path.join("data", "processed", "dataset.h5"))
    labels_tensor, label_classes, _, _ = load_labels(os.path.join("data", "processed", "*.csv"))

    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(spectrogram_tensor, labels_tensor)

    # Get input/output shapes
    for batch_data, batch_labels in train_dataset.take(1):
        input_shape = batch_data.shape[1:]
        output_length = batch_labels.shape[1]
        print("Batch Data Shape:", batch_data.shape)
        print("Batch Labels Shape:", batch_labels.shape)

    # Build and compile model
    model = build_model(input_shape, output_length)
    model = compile_model(model, learning_rate)

    # Tensorboard logs
    # create individual folder for tensorbaord logs
    def get_run_logdir():
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(os.curdir, '_logs', run_id)

    # tensorboard callback
    tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir())

    # early stopping callback
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='accuracy', 
                                                        min_delta=0, 
                                                        patience=2, 
                                                        verbose=0, 
                                                        mode='auto', 
                                                        baseline=None, 
                                                        restore_best_weights=True)
    


    print("Starting training...")
    train_model(model, train_dataset, eval_dataset, epochs, [tensorboard_cb, early_stopping_cb])
    print("Training completed successfully")
    modelname = time.strftime("model_%Y_%m_%d-%H_%M_%S") + ".keras"
    modelpath = "models"
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
    model.save(os.path.join("models", modelname))
    print("Model saved as", modelname)


if __name__ == "__main__":
    main()