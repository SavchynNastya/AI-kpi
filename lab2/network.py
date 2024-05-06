import random
import math

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = self.xavier_init(input_size, hidden_size)
        self.weights_hidden_output = self.xavier_init(hidden_size, output_size)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def softmax(x):
        exp_x = [math.exp(i) for i in x]
        sum_exp_x = sum(exp_x)

        if sum_exp_x == 0:
            return [1.0 / len(x) for _ in x]

        return [i / sum_exp_x for i in exp_x]

    @staticmethod
    def xavier_init(input_size, output_size):
        return [[random.uniform(-1, 1) * math.sqrt(2 / (input_size + output_size)) for _ in range(output_size)] for _ in range(input_size)]

    def initialize_weights(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = self.xavier_init(input_size, hidden_size)
        self.weights_hidden_output = self.xavier_init(hidden_size, output_size)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                # Forward pass
                input_data = X[i]

                hidden_layer_input = [sum(input_data[j] * self.weights_input_hidden[j][k] for j in range(len(input_data))) for k in range(len(self.weights_input_hidden[0]))]
                hidden_layer_output = [self.sigmoid(x) for x in hidden_layer_input]

                output_layer_input = [sum(hidden_layer_output[j] * self.weights_hidden_output[j][k] for j in range(len(hidden_layer_output))) for k in range(len(self.weights_hidden_output[0]))]
                predicted_output = self.softmax(output_layer_input)

                # Calculate error
                error = -sum(y[i][j] * math.log(predicted_output[j] + 1e-15) for j in range(len(predicted_output)))
                total_error += error

                # Backpropagation
                output_error_term = [predicted_output[j] - y[i][j] for j in range(len(predicted_output))]
                hidden_error = [sum(output_error_term[j] * self.weights_hidden_output[k][j] for j in range(len(output_error_term))) for k in range(len(self.weights_hidden_output))]
                hidden_error_term = [hidden_error[j] * self.sigmoid_derivative(hidden_layer_output[j]) for j in range(len(hidden_error))]

                # Update weights
                for j in range(len(self.weights_hidden_output)):
                    for k in range(len(self.weights_hidden_output[0])):
                        self.weights_hidden_output[j][k] -= learning_rate * hidden_layer_output[j] * output_error_term[k]

                for j in range(len(self.weights_input_hidden)):
                    for k in range(len(self.weights_input_hidden[0])):
                        self.weights_input_hidden[j][k] -= learning_rate * input_data[j] * hidden_error_term[k]

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Error: {total_error}")

    def predict(self, input_data):
        hidden_layer_input = [sum(input_data[j] * self.weights_input_hidden[j][k] for j in range(len(input_data))) for k in range(len(self.weights_input_hidden[0]))]
        hidden_layer_output = [self.sigmoid(x) for x in hidden_layer_input]

        output_layer_input = [sum(hidden_layer_output[j] * self.weights_hidden_output[j][k] for j in range(len(hidden_layer_output))) for k in range(len(self.weights_hidden_output[0]))]
        predicted_output = self.softmax(output_layer_input)

        predicted_number = predicted_output.index(max(predicted_output))
        return predicted_number

def read_data_from_file(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
        data = []
        labels = []

        for line in lines:
            values = list(map(int, line.split()))
            input_figure = values[:-1]
            label = values[-1]

            data.append(input_figure)
            labels.append(label)

        return data, labels

def main():
    train_data, train_labels = read_data_from_file("lab2\\train.txt")

    num_classes = len(set(train_labels))
    y_train = [[1 if j == int(label) else 0 for j in range(num_classes)] for label in train_labels]

    input_size = len(train_data[0])
    hidden_size = 64
    output_size = num_classes

    neural_network = NeuralNetwork(input_size, hidden_size, output_size)
    neural_network.train(train_data, y_train, epochs=6000, learning_rate=0.9)

    test_data, _ = read_data_from_file("lab2\\test.txt")

    predictions = [neural_network.predict(test_data[i]) for i in range(len(test_data))]

    print("\nPredicted Numbers for the new figures:", predictions)

if __name__ == '__main__':
    main()
