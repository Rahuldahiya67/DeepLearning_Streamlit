import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class PerceptronSentiment:
    def __init__(self):
        # Load the dataset
        df = pd.read_csv(IMDB_Dataset.csv)

        # Prepare data
        sentences = df['review']
        labels = df['sentiment'].map({'negative': 0, 'positive': 1})
        X_train, X_test, y_train, y_test = train_test_split(np.array(sentences), np.array(labels), train_size=0.8, random_state=42)

        # Tokenization and padding
        total_words = 1000
        max_length = 120
        self.tokenizer = Tokenizer(num_words=total_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X_train)
        train_seq = self.tokenizer.texts_to_sequences(X_train)
        train_padd = pad_sequences(train_seq, maxlen=max_length, truncating='post')
        test_seq = self.tokenizer.texts_to_sequences(X_test)
        test_padd = pad_sequences(test_seq, maxlen=max_length, truncating='post')

        # Build Perceptron model with embedding layer
        self.model = Sequential([
            Embedding(input_dim=total_words, output_dim=16, input_length=max_length),
            Flatten(),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Train the model
        self.history = self.model.fit(
            train_padd, 
            y_train, 
            epochs=20, 
            validation_data=(test_padd, y_test), 
            callbacks=[early_stopping]
        )

    def predict(self, text):
        # Tokenize and pad the input text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=120, truncating='post')

        # Predict sentiment
        prediction = (self.model.predict(padded_sequence) > 0.5).astype("int32")

        return "positive" if prediction[0][0] == 1 else "negative"

class BackpropagationSentiment:
    def __init__(self):
        # Load the dataset
        df = pd.read_csv(IMDB_Dataset.csv)

        # Prepare data
        sentences = df['review']
        labels = df['sentiment'].map({'negative': 0, 'positive': 1})
        X_train, X_test, y_train, y_test = train_test_split(np.array(sentences), np.array(labels), train_size=0.8, random_state=42)

        # Tokenization and padding
        total_words = 1000
        max_length = 120
        self.tokenizer = Tokenizer(num_words=total_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X_train)
        train_seq = self.tokenizer.texts_to_sequences(X_train)
        train_padd = pad_sequences(train_seq, maxlen=max_length, truncating='post')
        test_seq = self.tokenizer.texts_to_sequences(X_test)
        test_padd = pad_sequences(test_seq, maxlen=max_length, truncating='post')

        # Build a simple neural network
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(max_length,)),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Train the model
        self.history = self.model.fit(
            train_padd,
            y_train,
            epochs=20,
            validation_data=(test_padd, y_test),
            callbacks=[early_stopping]
        )

    def predict(self, text):
        # Tokenize and pad the input text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=120, truncating='post')

        # Predict sentiment
        prediction = (self.model.predict(padded_sequence) > 0.5).astype("int32")

        return "positive" if prediction[0][0] == 1 else "negative"



class DNNSentiment:
    def __init__(self):
        # Load the dataset
        df = pd.read_csv(IMDB_Dataset.csv)

        # Prepare data
        sentences = df['review']
        labels = df['sentiment'].map({'negative': 0, 'positive': 1})
        X_train, X_test, y_train, y_test = train_test_split(np.array(sentences), np.array(labels), train_size=0.8, random_state=42)

        # Tokenization and padding
        total_words = 1000
        max_length = 120
        self.tokenizer = Tokenizer(num_words=total_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X_train)
        train_seq = self.tokenizer.texts_to_sequences(X_train)
        train_padd = pad_sequences(train_seq, maxlen=max_length, truncating='post')
        test_seq = self.tokenizer.texts_to_sequences(X_test)
        test_padd = pad_sequences(test_seq, maxlen=max_length, truncating='post')

        # Build a Deep Neural Network (DNN)
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(max_length,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Train the model
        self.history = self.model.fit(
            train_padd,
            y_train,
            epochs=20,
            validation_data=(test_padd, y_test),
            callbacks=[early_stopping]
        )

    def predict(self, text):
        # Tokenize and pad the input text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=120, truncating='post')

        # Predict sentiment
        prediction = (self.model.predict(padded_sequence) > 0.5).astype("int32")

        return "positive" if prediction[0][0] == 1 else "negative"

class RNNSentiment:
    def __init__(self):
        # Load the dataset
        df = pd.read_csv(IMDB_Dataset.csv)

        # Prepare data
        sentences = df['review']
        labels = df['sentiment'].map({'negative': 0, 'positive': 1})
        X_train, X_test, y_train, y_test = train_test_split(np.array(sentences), np.array(labels), train_size=0.8, random_state=42)

        # Tokenization and padding
        total_words = 1000
        max_length = 120
        self.tokenizer = Tokenizer(num_words=total_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X_train)
        train_seq = self.tokenizer.texts_to_sequences(X_train)
        train_padd = pad_sequences(train_seq, maxlen=max_length, truncating='post')
        test_seq = self.tokenizer.texts_to_sequences(X_test)
        test_padd = pad_sequences(test_seq, maxlen=max_length, truncating='post')

        # Build an RNN model
        self.model = Sequential([
            Embedding(input_dim=total_words, output_dim=16, input_length=max_length),
            SimpleRNN(64),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Train the model
        self.history = self.model.fit(
            train_padd,
            y_train,
            epochs=20,
            validation_data=(test_padd, y_test),
            callbacks=[early_stopping]
        )

    def predict(self, text):
        # Tokenize and pad the input text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=120, truncating='post')

        # Predict sentiment
        prediction = (self.model.predict(padded_sequence) > 0.5).astype("int32")

        return "positive" if prediction[0][0] == 1 else "negative"


class LSTMSentiment:
    def __init__(self):
        # Load the dataset
        df = pd.read_csv(IMDB_Dataset.csv)

        # Prepare data
        sentences = df['review']
        labels = df['sentiment'].map({'negative': 0, 'positive': 1})
        X_train, X_test, y_train, y_test = train_test_split(np.array(sentences), np.array(labels), train_size=0.8, random_state=42)

        # Tokenization and padding
        total_words = 1000
        max_length = 120
        self.tokenizer = Tokenizer(num_words=total_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(X_train)
        train_seq = self.tokenizer.texts_to_sequences(X_train)
        train_padd = pad_sequences(train_seq, maxlen=max_length, truncating='post')
        test_seq = self.tokenizer.texts_to_sequences(X_test)
        test_padd = pad_sequences(test_seq, maxlen=max_length, truncating='post')

        # Build an LSTM model
        self.model = Sequential([
            Embedding(input_dim=total_words, output_dim=16, input_length=max_length),
            LSTM(64),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Train the model
        self.history = self.model.fit(
            train_padd,
            y_train,
            epochs=20,
            validation_data=(test_padd, y_test),
            callbacks=[early_stopping]
        )

    def predict(self, text):
        # Tokenize and pad the input text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=120, truncating='post')

        # Predict sentiment
        prediction = (self.model.predict(padded_sequence) > 0.5).astype("int32")

        return "positive" if prediction[0][0] == 1 else "negative"
