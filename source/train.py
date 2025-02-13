import tensorflow as tf
from preprocess import load_spectrogram_data, encode_labels, prepare_datasets
from model import build_model, compile_model
from utils import config
import os
import time

def train_model(model, train_dataset, eval_dataset, epochs=10, callbacks=None):
    history = model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    return history

def main():
    gpus = tf.config.list_physical_devices('GPU')
    print("Checking available GPUs:", gpus)
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for: {gpus}")
        except RuntimeError as e:
            print(e)

    params = config.Params()
    random_seed = params['general']['random_seed']
    epochs = params['train']['epochs']
    learning_rate = params['train']['learning_rate']
    batch_size = params['train']['batch_size']

    config.set_random_seeds(random_seed)

    print("Loading dataset. This may take a while...")
    spectrogram_dataset, book_start, book_end, chapter_start, chapter_end, labels_raw = load_spectrogram_data("data/processed/*.h5")
    print("Dataset loaded.")

    print(f"Books: {list(book_start.keys())}")
    print(f"Total Samples: {len(labels_raw)}")

    labels_tensor, label_classes = encode_labels(labels_raw)

    print("Preparing datasets...")
    train_dataset, eval_dataset = prepare_datasets(spectrogram_dataset, labels_tensor, batch_size=batch_size)
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
    model = build_model(input_shape, output_length)
    model, lr_scheduler = compile_model(model, learning_rate)
    print("Model built.")

    print("Starting training...")
    train_model(model, train_dataset, eval_dataset, epochs, [lr_scheduler])
    print("Training completed successfully.")

    modelname = time.strftime("model_%Y_%m_%d-%H_%M_%S") + ".keras"
    modelpath = "models"
    if not os.path.exists(modelpath):
        os.makedirs(modelpath)
    model.save(os.path.join("models", modelname))
    print("Model saved as", modelname)

if __name__ == "__main__":
    main()
