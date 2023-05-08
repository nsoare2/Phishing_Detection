from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse
from keras.models import Sequential
from keras.layers import Dense

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Load the dataset
    dataset = pd.read_csv('fraud_email_.csv')

    # Replace NaN values with empty strings
    dataset = dataset.fillna('')
    X = dataset['Text'].values
    y = dataset['Class'].values

    # Convert the text to numerical features using bag-of-words representation
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)

    # Total number of samples
    n_samples = X.shape[0]
    n_train = int(n_samples * 0.8)

    # Create the sparse matrix for the training and testing data
    X_train = scipy.sparse.csr_matrix(X[:n_train, :])
    X_test = scipy.sparse.csr_matrix(X[n_train:, :])

    # Sort the indices of the sparse matrix
    X_train.sort_indices()
    X_test.sort_indices()

    # Split the dataset into training and testing sets, with 80% for training and 20% for testing
    y_train = y[:n_train]
    y_test = y[n_train:]

    # Create the neural network model
    model = Sequential()
    model.add(Dense(units=16, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))
   
    # Predict the target variable for the input data
    text = request.json['text']
    input_data = vectorizer.transform([text])
    input_data = scipy.sparse.csr_matrix(input_data)
    input_data.sort_indices()
    y_pred = model.predict(input_data)[0][0]
    y_pred = int(round(y_pred))

    # Return the predicted class
    return jsonify({'class': y_pred})

if __name__ == '__main__':
    app.run(debug=True)
