import os
import pandas as pd
import tensorflow as tf
import random
import sacred

from sacred import Experiment
from sacred.observers import FileStorageObserver


class_ex = Experiment('URL_classification')


@class_ex.config
def base_config():
    # Set random seed
    RAND_SEED = 1320
    random.seed(RAND_SEED)
    # Random seed for TensorFlow
    tf.random.set_seed(RAND_SEED)

    ##### Data #####
    # CATEGORIES_TO_USE = ['Adult', 'Computers', 'Recreation', 'Science']
    categories_to_use = ['Left', 'Center', 'Right']
    n_classes = len(categories_to_use)  # Number of classes to predict
    # Size of training dataset
    train_size = 0.8
    data_dir = None

    ##### Text Processing #####
    # Vocabulary size for tokenization
    vocab_size = 10000
    # Maximum length of token sequence
    max_len = 20

    ##### Classification Model #####
    # Size of data batches
    batch_size = 40000
    # Dimension of Word Embedding layer
    embedding_dim = 512
    # Learning rate of optimizer
    lr = 1e-3
    # Number of epochs to train model
    n_epoch = 100


def load_data(data_dir, train_size):
    # Path to data file
    data_path = os.path.join(data_dir, "URL Classification.csv")

    # Load data into pandas DataFrame
    data_df = pd.read_csv(data_path, header=None)

    # First, shuffle the data, by shuffling indices.
    idx_all = list(range(len(data_df)))
    random.shuffle(idx_all)

    # Find out where to split training/test set
    m = int(len(data_df) * train_size)

    # Split indices
    idx_train, idx_test = idx_all[:m], idx_all[m:]
    # Split data
    X_train, y_train = data_df.iloc[idx_train, 1].values, data_df.iloc[idx_train, 2].values
    X_test, y_test = data_df.iloc[idx_test, 1].values, data_df.iloc[idx_test, 2].values

    return X_train, X_test, y_train, y_test


def construct_model(vocab_size, embedding_dim, max_len, n_classes, lr):
    # Construct a neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])

    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr),
                  metrics=['accuracy'])

    return model


@class_ex.main
def run(categories_to_use, train_size, vocab_size, max_len, batch_size, embedding_dim,
        lr, n_epoch, data_path):
    pass


if __name__ == '__main__':
    sacred.SETTINGS['CAPTURE_MODE'] = 'sys'
    class_ex.observers.append(FileStorageObserver('runs'))
    class_ex.run_commandline()
