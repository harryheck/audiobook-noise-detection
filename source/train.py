import tensorflow as tf
from preprocess import load_spectrogram_data, encode_labels, prepare_datasets
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
    # Print available GPUs
    
    gpus = tf.config.list_physical_devices('GPU')
    print("Checking available GPUs:", gpus) 
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)  # Allow memory growth
            print(f"GPU memory growth enabled for: {gpus}")
        except RuntimeError as e:
            print(e)

    # Load config
    params = config.Params()
    random_seed = params['general']['random_seed']
    epochs = params['train']['epochs']
    learning_rate = params['train']['learning_rate']
    batch_size = params['train']['batch_size']

    # Set random seed
    config.set_random_seeds(random_seed)

    # Load dataset
    print("Loading dataset. This may take a while...")
    spectrogram_tensor, _, _, _, _, labels_raw = load_spectrogram_data(os.path.join("data", "processed", "*.h5"))
    print("Dataset loaded.")

    # 🚨 Debugging: Check dataset validity
    print(f"Dataset Tensor Shape: {spectrogram_tensor.shape}")
    print(f"Labels Raw (first 5): {labels_raw[:5]}")

    # 🚨 Check for empty dataset
    if spectrogram_tensor.shape[0] == 0:
        raise ValueError("Error: The dataset is empty! Check preprocessing step.")

    if len(labels_raw) == 0:
        raise ValueError("Error: No labels found in dataset! Check preprocessing step.")

    labels_tensor, label_classes = encode_labels(labels_raw)

    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, eval_dataset = prepare_datasets(spectrogram_tensor, labels_tensor, batch_size=batch_size)
    print("Datasets prepared.")

    # Get input/output shapes
    input_shape, output_length = None, None
    for batch_data, batch_labels in train_dataset.take(1):
        input_shape = batch_data.shape[1:]
        output_length = batch_labels.shape[1]
        print("Batch Data Shape:", batch_data.shape)
        print("Batch Labels Shape:", batch_labels.shape)

    # 🚨 Ensure input shapes are valid
    if input_shape is None or output_length is None:
        raise ValueError("Error: input_shape or output_length is None! Dataset may be empty.")

    # Define training strategy
    # strategy = tf.distribute.MirroredStrategy()


    # Build and compile model
    print("Building model...")
    model = build_model(input_shape, output_length)
    model = compile_model(model, learning_rate)
    print("Model built.")

    # Enable mixed precision training
    # tf.keras.mixed_precision.set_global_policy('mixed_float32')
    # print("Using mixed precision training.")

    # Tensorboard logs
    def get_run_logdir():
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(os.curdir, 'logs', 'tensorboard', run_id)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir())

    # Early stopping callback
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        min_delta=0, 
        patience=8, 
        verbose=1, 
        mode='auto', 
        baseline=None, 
        restore_best_weights=True
    )

    print("Starting training...")
    train_model(model, train_dataset, eval_dataset, epochs, [tensorboard_cb, early_stopping_cb])
    print("Training completed successfully.")

    # Save trained model
    modelname = time.strftime("model_%Y_%m_%d-%H_%M_%S") + ".keras"
    modelpath = "models"
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
    model.save(os.path.join("models", modelname))
    print("Model saved as", modelname)

if __name__ == "__main__":
    main()
