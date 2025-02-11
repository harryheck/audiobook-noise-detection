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

    # ✅ Print available GPUs
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))

    # ✅ Force TensorFlow to use GPUs if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)  # Prevent memory overflow
            tf.config.set_visible_devices(gpus, 'GPU')  # Use first GPU
            print("✅ Using GPU:", gpus[0])
        except RuntimeError as e:
            print("⚠ GPU setup failed:", e)
    else:
        print("⚠ No GPU detected. Running on CPU.")

    # ✅ Enable multi-GPU training
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # Load config
        params = config.Params()
        random_seed = params['general']['random_seed']
        epochs = params['train']['epochs']
        learning_rate = params['train']['learning_rate']

        # Set a random seed for reproducibility across all devices. Add more devices if needed
        config.set_random_seeds(random_seed)

        # Load and preprocess data
        spectrogram_tensor, _, _, _, _, labels_raw = load_spectrogram_data(os.path.join("data", "processed", "*.h5"))
        print(labels_raw)

        labels_tensor, label_classes = encode_labels(labels_raw)

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

    # ✅ Enable mixed precision (float16 training for memory efficiency)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Tensorboard logs
    def get_run_logdir():
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(os.curdir, '_logs', run_id)

    # TensorBoard callback
    tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir())

    # Early stopping callback
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                          min_delta=0, 
                                                          patience=5, 
                                                          verbose=1, 
                                                          mode='auto', 
                                                          baseline=None, 
                                                          restore_best_weights=True)

    print("🚀 Starting training...")
    train_model(model, train_dataset, eval_dataset, epochs, [tensorboard_cb, early_stopping_cb])
    print("✅ Training completed successfully")

    # Save trained model
    modelname = time.strftime("model_%Y_%m_%d-%H_%M_%S") + ".keras"
    modelpath = "models"
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
    model.save(os.path.join("models", modelname))
    print("💾 Model saved as", modelname)

if __name__ == "__main__":
    main()
