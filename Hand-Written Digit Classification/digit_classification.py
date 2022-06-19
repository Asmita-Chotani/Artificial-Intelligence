import math
import time
import sys
import numpy as np
import pandas as pd

batch_size = 32
learning_rate = 0.01
input_data_size = 784
output_size = 10
hidden_layer_1_size = 128
hidden_layer_2_size = 64
predicted_data = []


def normalization(test_images):
    test_data = test_images / 255.0
    test_data = np.asarray(test_data)
    return test_data


class NNModel:
    def __init__(self):
        self.epochs = 50 #80 #1000

    def initialization(self):
        limit1 = math.sqrt(6/(input_data_size+hidden_layer_1_size))
        self.W1 = np.random.uniform(low=-limit1, high=limit1, size=(hidden_layer_1_size, input_data_size))

        limit2 = math.sqrt(6/(hidden_layer_1_size+hidden_layer_2_size))
        self.W2 = np.random.uniform(low=-limit2, high=limit2, size=(hidden_layer_2_size, hidden_layer_1_size))

        limit3 = math.sqrt(6/(hidden_layer_2_size+output_size))
        self.W3 = np.random.uniform(low=-limit3, high=limit3, size=(output_size, hidden_layer_2_size))

    def one_hot_encoding(self):
        for i in range(1, output_size+1):
            self.full_data[input_data_size+i] = 0
        total_no_of_rows = self.full_data.shape[0]
        for i in range(total_no_of_rows):
            column = input_data_size+int(self.full_data.loc[i, input_data_size])
            self.full_data.loc[i, column+1] = 1

    def randomizing_data(self):
        val = []
        no_of_rows_X = self.matrix_X.shape[0]
        for i in range(no_of_rows_X):
            val.append(i)
        np.random.shuffle(val)
        self.matrix_X = self.matrix_X[val]
        self.encoded_matrix_Y = self.encoded_matrix_Y[val]

    def splitting_data(self, train_image_data, train_image_label):
        encoded_column = []
        columns_not_included = []
        self.full_data = train_image_data/255

        self.full_data[input_data_size] = train_image_label[0]

        self.one_hot_encoding()

        # split into training and validation data, 8:2 ratio
        # Assigning 80% of the dataset into the training set
        self.training_data = self.full_data.sample(frac=0.8, random_state=42)

        # Assigning 20% of the dataset into the validation set
        validation_set_index = ~self.full_data.index.isin(self.training_data.index)
        self.validation_data = self.full_data.loc[validation_set_index].reset_index(drop=True)
        self.training_data = self.training_data.reset_index(drop=True)

        for col1 in range(1, output_size + 1):
            encoded_column.append(input_data_size + col1)

        for col2 in range(0, output_size + 1):
            columns_not_included.append(input_data_size + col2)

        new_columns = {}

        for t in range(0, output_size):
            new_columns[input_data_size+1+t] = t

        # Not including the encoded columns endpoints
        matrix_X_not_part = ~self.training_data.columns.isin(columns_not_included)
        matrix_validate_X_not_part = ~self.training_data.columns.isin(columns_not_included)

        # Not including the encoded columns
        self.matrix_X = np.asarray(self.training_data.loc[:, matrix_X_not_part])  # (48000, 784)
        self.validated_matrix_X = np.asarray(self.validation_data.loc[:, matrix_validate_X_not_part]) # (12000, 784)

        # Encoded columns to be included
        encoded_matrix_Y_columns = self.training_data.columns.isin(encoded_column)
        validated_encoded_matrix_Y_columns = self.validation_data.columns.isin(encoded_column)


        # Only including the encoded columns
        self.encoded_matrix_Y = np.asarray(self.training_data.loc[:,  encoded_matrix_Y_columns].rename(columns=new_columns)) # (48000, 10)
        self.validated_encoded_matrix_Y = np.asarray(self.validation_data.loc[:, validated_encoded_matrix_Y_columns].rename(columns=new_columns)) # (12000, 10)

    def forward_pass_call(self,x):
        predicted_value = self.feed_forward(x)
        prediction = np.argmax(predicted_value)
        return prediction

    def feed_forward(self, x):
        # INPUT LAYER
        self.Z1 = self.W1.dot(x)
        self.A1 = 1 / (1 + np.exp(-(self.Z1))) # activation function - SIGMOID

        # HIDDEN LAYER 1
        self.Z2 = self.W2.dot(self.A1)
        self.A2 = 1 / (1 + np.exp(-(self.Z2)))  # activation function - SIGMOID

        # HIDDEN LAYER 2
        self.Z3 = self.W3.dot(self.A2)
        value = np.exp(self.Z3 - self.Z3.max())
        self.A3 = value / np.sum(value, axis=0) # activation function - SOFTMAX

        return self.A3

    def accuracy_value(self, predicted, actual):
        no_of_hits = []
        for i, j in zip(predicted, actual):
            prediction = self.forward_pass_call(i)
            no_of_hits.append(prediction == np.argmax(j))

        return np.mean(no_of_hits)

    #Function for the back propagation
    def back_pass(self):

        #Computing the error
        cost_value = (self.A3 - self.y_encoded_batch)
        self.W3 = self.W3 - learning_rate * np.dot(cost_value, self.A2.T)

        cost_value = ((np.exp(-self.Z2)) / ((np.exp(-self.Z2) + 1) ** 2)) * np.dot(self.W3.T, cost_value)  # sigmoid derivative
        self.W2 = self.W2 - learning_rate * np.dot(cost_value, self.A1.T)

        cost_value = ((np.exp(-self.Z1)) / ((np.exp(-self.Z1) + 1) ** 2)) * np.dot(self.W2.T, cost_value)  # sigmoid derivative
        self.W1 = self.W1 - learning_rate * np.dot(cost_value, self.x_batch.T)

    #Train the model for the given number of epochs
    def train(self, train_image_data, train_image_label):
        self.initialization()
        self.splitting_data(train_image_data, train_image_label)
        for epoch in range(self.epochs):
            loss=0
            self.randomizing_data()
            starting = time.time()
            for batch_no in range(self.matrix_X.shape[0]//batch_size - 1):
                starting_point = batch_no*batch_size
                last_point = (batch_no+1)*batch_size
                new_X = self.matrix_X[starting_point:last_point]
                self.x_batch = new_X.T
                new_Y = self.encoded_matrix_Y[starting_point:last_point]
                self.y_encoded_batch = new_Y.T
                self.y_predicted_batch = self.feed_forward(self.x_batch)
                self.back_pass()
                cross_entropy = (np.sum(self.y_encoded_batch*np.log(self.y_predicted_batch))/self.y_predicted_batch.shape[0])
                loss = loss - cross_entropy
            accuracy = self.accuracy_value(self.validated_matrix_X, self.validated_encoded_matrix_Y)
            if abs(accuracy) < 0.05:
                break
            print("epoch number:", epoch, "time taken:", time.time() - starting, "loss occurred:", loss, "accuracy received:", accuracy * 100)


start = time.time()
if (len(sys.argv)>1):
    print("-commandline input-")
    filename1 = sys.argv[1]
    training_images = pd.read_csv(filename1, header=None)
    filename2 = sys.argv[2]
    training_labels = pd.read_csv(filename2, header=None)
    filename3 = sys.argv[3]
    test_images = pd.read_csv(filename3, header=None)
else:
    print("-work directory input-")
    test_images = pd.read_csv("test_image.csv", header=None)
    training_images = pd.read_csv("train_image.csv", header=None)
    training_labels = pd.read_csv("train_label.csv", header=None)

# normalization of test images
test_data = normalization(test_images)

model = NNModel()
model.train(training_images, training_labels)

for data in test_data:
    prediction = model.forward_pass_call(data)
    predicted_data.append(prediction)

np.savetxt('test_predictions.csv', predicted_data, delimiter=',', fmt='%d')

end = time.time()
print("time", end-start)
