import random
import math

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers_count, hidden_layers_size, output_size, activation_function='sigmoid'):
        self.hidden_layers_count = hidden_layers_count
        self.hidden_layers_size = hidden_layers_size
        self.weights_input_hidden = self.initialize_weights(input_size, hidden_layers_size[0] if hidden_layers_count > 0 else 0)
        self.weights_hidden_output = self.initialize_weights(hidden_layers_size[-1] if hidden_layers_count > 0 else input_size, output_size)
        self.activation_function = activation_function

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def tanh(x):
        return math.tanh(x)

    @staticmethod
    def relu(x):
        return max(0, x)

    def activate(self, x):
        if self.activation_function == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_function == 'tanh':
            return self.tanh(x)
        elif self.activation_function == 'relu':
            return self.relu(x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation_function}")

    def activate_derivative(self, x):
        if self.activation_function == 'sigmoid':
            # Derivative of sigmoid function
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        elif self.activation_function == 'tanh':
            # Derivative of tanh function
            return 1 - math.tanh(x)**2
        elif self.activation_function == 'relu':
            # Derivative of ReLU function
            return 1 if x > 0 else 0
        else:
            raise ValueError(f"Unknown activation function: {self.activation_function}")

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

    def initialize_weights(self, input_size, output_size):
        weights = []
        
        if self.hidden_layers_count == 0:
            return self.xavier_init(input_size, output_size)
        
        # Input to the first hidden layer
        weights.append(self.xavier_init(input_size, self.hidden_layers_size[0]))
        
        # Hidden layers
        for i in range(self.hidden_layers_count - 1):
            weights.append(self.xavier_init(self.hidden_layers_size[i], self.hidden_layers_size[i + 1]))
        
        # Output layer
        weights.append(self.xavier_init(self.hidden_layers_size[-1], output_size))
        
        return weights

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                # Forward pass
                input_data = X[i]
                hidden_layer_outputs = [input_data]

                # Hidden layers
                for layer in range(self.hidden_layers_count):
                    hidden_layer_input = [sum(hidden_layer_outputs[layer][j] * self.weights_input_hidden[layer][j][k] for j in range(len(input_data))) for k in range(len(self.weights_input_hidden[layer][0]))]
                    hidden_layer_output = [self.activate(x) for x in hidden_layer_input]
                    hidden_layer_outputs.append(hidden_layer_output)

                # Output layer
                if self.hidden_layers_count == 0:
                    output_layer_input = [sum(input_data[j] * self.weights_hidden_output[j][k] for j in range(len(input_data))) for k in range(len(self.weights_hidden_output[0]))]
                else:
                    output_layer_input = [sum(hidden_layer_outputs[-1][j] * self.weights_hidden_output[-1][j][k] for j in range(len(hidden_layer_outputs[-1]))) for k in range(len(self.weights_hidden_output[-1][0]))]
                predicted_output = self.softmax(output_layer_input)

                # Calculate error
                error = -sum(y[i][j] * math.log(predicted_output[j] + 1e-15) for j in range(len(predicted_output)))
                total_error += error

                # Backpropagation
                output_error_term = [predicted_output[j] - y[i][j] for j in range(len(predicted_output))]

                # Update weights for the output layer
                if self.hidden_layers_count == 0:
                    for j in range(len(self.weights_hidden_output)):
                        for k in range(len(self.weights_hidden_output[0])):
                            self.weights_hidden_output[j][k] -= learning_rate * input_data[j] * output_error_term[k]
                else:
                    for j in range(len(self.weights_hidden_output[-1])):
                        for k in range(len(self.weights_hidden_output[-1][0])):
                            self.weights_hidden_output[-1][j][k] -= learning_rate * hidden_layer_outputs[-1][j] * output_error_term[k]

                # Update weights for hidden layers
                for layer in reversed(range(self.hidden_layers_count)):
                    if layer == self.hidden_layers_count - 1:
                        # For the last hidden layer, use output_error_term directly
                        hidden_error_term = [output_error_term[k] * self.weights_hidden_output[layer][k][j] * self.activate_derivative(hidden_layer_outputs[layer + 1][k]) for j in range(len(hidden_layer_outputs[layer]))]
                    else:
                        # For previous hidden layers, sum over the next layer's errors
                        hidden_error_term = [sum(self.weights_hidden_output[layer + 1][k][j] * self.activate_derivative(hidden_layer_outputs[layer + 1][k]) * hidden_error_term[k] for k in range(len(hidden_layer_outputs[layer + 1]))) for j in range(len(hidden_layer_outputs[layer]))]

                    # Update weights for the current hidden layer
                    for j in range(len(self.weights_input_hidden[layer])):
                        for k in range(len(self.weights_input_hidden[layer][j])):
                            if len(hidden_error_term) > k:
                                self.weights_input_hidden[layer][j][k] -= learning_rate * hidden_layer_outputs[layer][j] * hidden_error_term[k]

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Error: {total_error}")

    def predict(self, input_data):
        hidden_layer_output = input_data

        if self.hidden_layers_count > 0:
            for layer in range(self.hidden_layers_count):
                hidden_layer_input = [sum(hidden_layer_output[j] * self.weights_input_hidden[layer][j][k] for j in range(len(hidden_layer_output))) for k in range(len(self.weights_input_hidden[layer][0]))]
                hidden_layer_output = [self.activate(x) for x in hidden_layer_input]
        
        if self.hidden_layers_count == 0:
            output_layer_input = [sum(hidden_layer_output[j] * self.weights_hidden_output[j][k] for j in range(len(hidden_layer_output))) for k in range(len(self.weights_hidden_output[0]))]
        else:
            output_layer_input = [sum(hidden_layer_output[j] * self.weights_hidden_output[-1][j][k] for j in range(len(hidden_layer_output))) for k in range(len(self.weights_hidden_output[-1][0]))]
        
        predicted_output = self.softmax(output_layer_input)
        print(f"Softmax predicted output: {predicted_output}")

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
    test_data, _ = read_data_from_file("lab2\\test.txt")
    num_classes = len(set(train_labels))
    y_train = [[1 if j == int(label) else 0 for j in range(num_classes)] for label in train_labels]

    input_size = len(train_data[0])
    output_size = num_classes

    # Scenario 1: Original Configuration
    print("Scenario 1: Original Configuration")
    neural_network_1 = NeuralNetwork(input_size, hidden_layers_count=1, hidden_layers_size=[64,], output_size=output_size, activation_function='sigmoid')
    neural_network_1.train(train_data, y_train, epochs=6000, learning_rate=0.9)
    predictions_1 = [neural_network_1.predict(test_data[i]) for i in range(len(test_data))]
    print("Predicted Numbers:", predictions_1)

    # # Scenario 2: More Hidden Neurons
    # print("\nScenario 2: More Hidden Neurons")
    # neural_network_2 = NeuralNetwork(input_size, hidden_layers_count=1, hidden_layers_size=[128,], output_size=output_size, activation_function='sigmoid')
    # neural_network_2.train(train_data, y_train, epochs=6000, learning_rate=0.9)
    # predictions_2 = [neural_network_2.predict(test_data[i]) for i in range(len(test_data))]
    # print("Predicted Numbers:", predictions_2)

    # # Scenario 3: Different Activation Function (Tanh)
    # print("\nScenario 3: Different Activation Function (Tanh)")
    # neural_network_3 = NeuralNetwork(input_size, hidden_layers_count=1, hidden_layers_size=[64,], output_size=output_size, activation_function='tanh')
    # neural_network_3.train(train_data, y_train, epochs=6000, learning_rate=0.9)
    # predictions_3 = [neural_network_3.predict(test_data[i]) for i in range(len(test_data))]
    # print("Predicted Numbers:", predictions_3)

    # # Scenario 4: Different Activation Function (ReLU)
    # print("\nScenario 4: Different Activation Function (ReLU)")
    # neural_network_4 = NeuralNetwork(input_size, hidden_layers_count=1, hidden_layers_size=[64,], output_size=output_size, activation_function='relu')
    # neural_network_4.train(train_data, y_train, epochs=6000, learning_rate=0.9)
    # predictions_4 = [neural_network_4.predict(test_data[i]) for i in range(len(test_data))]
    # print("Predicted Numbers:", predictions_4)

    # # Scenario 5: No Hidden Layer
    # print("\nScenario 5: No Hidden Layer")
    # neural_network_5 = NeuralNetwork(input_size, hidden_layers_count=0, hidden_layers_size=[], output_size=output_size, activation_function='sigmoid')
    # neural_network_5.train(train_data, y_train, epochs=8000, learning_rate=0.0001)
    # predictions_5 = [neural_network_5.predict(test_data[i]) for i in range(len(test_data))]
    # print("Predicted Numbers:", predictions_5)

    # # Scenario 6: Multiple Hidden Layers
    # print("\nScenario 6: Multiple Hidden Layers")
    # neural_network_6 = NeuralNetwork(input_size, hidden_layers_count=2, hidden_layers_size=[36, 36], output_size=output_size, activation_function='sigmoid')
    # neural_network_6.train(train_data, y_train, epochs=8000, learning_rate=0.6)
    # predictions_6 = [neural_network_6.predict(test_data[i]) for i in range(len(test_data))]
    # print("Predicted Numbers:", predictions_6)

if __name__ == '__main__':
    main()
