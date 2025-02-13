import tensorflow as tf

def build_model(input_shape, output_length):
    """Builds a CNN model for audio classification using mel spectrograms.

    Args:
        input_shape (tuple): Shape of the input spectrogram (height, width, channels).
        output_length (int): Number of output classes.

    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """
    model = tf.keras.Sequential()

    # ✅ Ensure correct input shape
    model.add(tf.keras.Input(shape=(input_shape[0], input_shape[1], 1)))  # Ensure grayscale (1 channel)
    model.add(tf.keras.layers.BatchNormalization())

    # 🔹 First Conv Block
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))

    # 🔹 Second Conv Block
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))

    # 🔹 Third Conv Block
    model.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))

    # 🔹 Fourth Conv Block (New)
    model.add(tf.keras.layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.GlobalAveragePooling2D())  # ✅ Changed from GlobalMaxPool2D

    # 🔹 Fully Connected Layer
    model.add(tf.keras.layers.Dense(128, activation="relu", activity_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Dropout(0.5))

    # 🔹 Output Layer
    model.add(tf.keras.layers.Dense(output_length, activation="softmax"))

    print("✅ Model architecture successfully built")
    return model


def compile_model(model, learning_rate=0.001):
    """Compiles the CNN model with Adam optimizer and learning rate decay.

    Args:
        model (tf.keras.Model): Keras model.
        learning_rate (float, optional): Initial learning rate. Defaults to 0.001.

    Returns:
        tf.keras.Model: Compiled model.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ✅ Learning Rate Reduction Callback
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6
    )

    print("✅ Model successfully compiled with learning rate reduction")
    return model, lr_scheduler
