import os
import random
import numpy as np
import pandas as pd
import wordninja

# One-Hot Encoder
from sklearn.preprocessing import OneHotEncoder

# Text processing methods with TF
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# The code below are adapted from COMP3359 AI Applications course material.

# Default Settings
VOCAB_SIZE = 10000  # Vocab size of tokenizer
MAX_LEN = 20        # Max length for padding

def preprocess_data(
        X, y=None, 
        tokenizer=None, vocab_size=VOCAB_SIZE, max_len=MAX_LEN,
        encoder=None, categories=None,
        return_processors=False,
        
    ) :

    # Recognize each word and split them.
    # e.g. "infococoonbreaker" -> ["info", "cocoon", "breaker"]
    X = [ wordninja.split(str(x)) for x in X ]

    # Convert list of tokens into one long string with tokens separated by spaces " ".
    # e.g. ["info", "cocoon", "breaker"] ==> "info cocoon breaker".
    X = [ " ".join(tokens) for tokens in X ]

    # Convert word tokens to integer tokens
    # If tokenizer is not provided, construct a new tokenizer and fit data to it
    if tokenizer is None :
        tokenizer = Tokenizer(oov_token=True, num_words=VOCAB_SIZE)
        tokenizer.fit_on_texts(X)

    # Transform texts to index sequences
    # E.g. "info cocoon breaker" -> [156, 243, 33]
    X = tokenizer.texts_to_sequences(X)

    # Zero-padding
    # E.g. [156, 243, 33] -> [156, 243, 33, 0, 0, 0, 0, 0]
    X = pad_sequences(X, maxlen=max_len, padding='post')

    # Return if no label to process
    if y is None :
        if not return_processors :
            # Return processed data only
            return X
        else :
            # Return processed data and the tokenizer & encoder used
            return X, tokenizer, encoder

    ##### Process Labels #####

    # After preprocessing,
    # central: [1, 0, 0]
    # left: [0, 1, 0]
    # right: [0, 0, 1]
    
    # Use one-hot encoding to convert string classes to one-hot encoding vectors.

    # If encoder not provided, construct a new encoder
    if encoder is None:
        # Construct new encoder
        encoder = OneHotEncoder(handle_unknown='ignore')

        if categories is not None :
            # If categories to use is specified, use them to fit the encoder
            encoder.fit(np.asarray(categories).reshape(-1,1))
        else :
            # else, fit encoder using provided labels
            encoder.fit(np.asarray(y).reshape(-1,1))

    # Convert categories into one-hot vectors
    y = encoder.transform(y.reshape(-1,1)).toarray()

    
    if not return_processors :
        # Return processed data only
        return X, y
    else :
        # Return processed data and the tokenizer & encoder used
        return X, y, tokenizer, encoder
    

def predict_url(url, model, tokenizer, encoder, class_names=None, return_logits=False) :
    """
    Preprocess and classify URL string(s).
    Steps:
    1. Preprocess URL string(s) and convert URL(s) to model inputs
    2. Feed input(s) to model
    3. Convert model outputs to integer/string class(es)

    Arguments:
    - url (str or list of str): URL string(s) to use for prediction
    - model (tf.keras.Sequential): URL classification model
    - tokenizer (keras_preprocessing.text.Tokenizer): Tokenizer object for 
         tokenization of URLs
    - encoder (sklearn.preprocessing.OneHotEncoder): Encoder object 
         for converting labels to one-hot encoding
    - class_names (list of str): class names of prediction targets
    - return_logits (bool): If true, the model outputs are returned immediately

    """
    # If only one url (i.e. string) is provided, convert variable url 
    # to a list of just one string (required by preprocess_data(...))
    if isinstance(url, str) :
        url = [url]

    # Process URL 
    x = preprocess_data(url, tokenizer=tokenizer, encoder=encoder)

    # Feed input to model
    outs = model.predict(x)

    # Return model outputs if it is required
    if return_logits :
        return outs

    # Get the predicted class
    y = np.argmax(outs, axis=1)

    # If class names are provided, convert the integer class to the 
    # category name.
    if class_names is not None:
        y = np.asarray(class_names)[y]

    return y

    

