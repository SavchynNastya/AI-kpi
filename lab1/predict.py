# # import numpy as np
# # import matplotlib.pyplot as plt

# # class ArtificialNeuron:
# #     def __init__(self, num_inputs, learning_rate):
# #         self.weights = np.random.rand(num_inputs)
# #         self.bias = 0
# #         self.learning_rate = learning_rate

# #     def sigmoid(self, x):
# #         return 1 / (1 + np.exp(-x))

# #     def activate(self, inputs):
# #         # Лінійна комбінація вхідних значень та ваг
# #         weighted_sum = np.dot(inputs, self.weights) + self.bias
# #         return self.sigmoid(weighted_sum)

# #     def train(self, inputs, targets, num_epochs):
# #         for epoch in range(num_epochs):
# #             for i in range(len(inputs)):
# #                 x = inputs[i]
# #                 y_true = targets[i]

# #                 # Активація нейрона
# #                 y_pred = self.activate(x)

# #                 # Градієнт для ваг та зсуву
# #                 gradient_weights = 2 * (y_pred - y_true) * self.sigmoid(y_pred) * (1 - self.sigmoid(y_pred)) * x
# #                 gradient_bias = 2 * (y_pred - y_true) * self.sigmoid(y_pred) * (1 - self.sigmoid(y_pred))

# #                 # Оновлення ваг та зсуву
# #                 self.weights -= self.learning_rate * gradient_weights
# #                 self.bias -= self.learning_rate * gradient_bias

# # def predict_time_series(inputs, num_epochs, learning_rate):
# #     # Ініціалізація нейрона для прогнозування часового ряду
# #     neuron = ArtificialNeuron(num_inputs=1, learning_rate=learning_rate)

# #     # Навчання нейрона
# #     neuron.train(inputs[:-1], inputs[1:], num_epochs)

# #     # Предикція за допомогою нейрона
# #     predictions = [neuron.activate(x) for x in inputs]

# #     return predictions

# # # Вхідні дані - часовий ряд
# # time_series = np.array([2.65, 5.60, 1.21, 5.48, 0.73, 4.08, 1.88, 5.31, 0.78, 4.36, 1.71, 5.62, 0.43, 4.21, 1.21])

# # # Налаштування гіперпараметрів
# # num_epochs = 1000
# # learning_rate = 0.01

# # # Прогнозування часового ряду
# # predictions = predict_time_series(time_series, num_epochs, learning_rate)
# # print(predictions)

# # Графік результатів
# # plt.plot(time_series, label='Actual')
# # plt.plot(np.arange(1, len(predictions) + 1), predictions, label='Predicted')
# # plt.legend()
# # plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# class ArtificialNeuron:
#     def __init__(self, num_inputs, learning_rate):
#         self.weights = np.random.rand(num_inputs)
#         self.bias = 0
#         self.learning_rate = learning_rate

#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))

#     def activate(self, inputs):
#         # Linear combination of input values and weights
#         weighted_sum = np.dot(inputs, self.weights) + self.bias
#         return self.sigmoid(weighted_sum)

#     def train(self, inputs, targets, num_epochs):
#         for epoch in range(num_epochs):
#             for i in range(len(inputs)):
#                 x = inputs[i]
#                 y_true = targets[i]

#                 # Activation of the neuron
#                 y_pred = self.activate(x)

#                 # Gradient for weights and bias
#                 gradient_weights = 2 * (y_pred - y_true) * self.sigmoid(y_pred) * (1 - self.sigmoid(y_pred)) * x
#                 gradient_bias = 2 * (y_pred - y_true) * self.sigmoid(y_pred) * (1 - self.sigmoid(y_pred))

#                 # Update weights and bias
#                 self.weights -= self.learning_rate * gradient_weights
#                 self.bias -= self.learning_rate * gradient_bias

# def normalize_data(data):
#     min_val = np.min(data)
#     max_val = np.max(data)
#     return (data - min_val) / (max_val - min_val)

# def predict_time_series(inputs, num_epochs, learning_rate):
#     inputs = normalize_data(inputs)
#     # Initialization of the neuron for time series prediction
#     neuron = ArtificialNeuron(num_inputs=1, learning_rate=learning_rate)

#     # Training the neuron
#     neuron.train(inputs[:-1], inputs[1:], num_epochs)

#     # Prediction using the neuron
#     predictions = [neuron.activate(x) for x in inputs]

#     return predictions

# # Input data - time series
# time_series = np.array([2.65, 5.60, 1.21, 5.48, 0.73, 4.08, 1.88, 5.31, 0.78, 4.36, 1.71, 5.62, 0.43, 4.21, 1.21])

# # Hyperparameter settings
# num_epochs = 1000
# learning_rate = 0.01

# # Time series prediction
# predictions = predict_time_series(time_series, num_epochs, learning_rate)

# print(predictions)

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Ваги між вхідним та прихованим шаром
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        # Ваги між прихованим та вихідним шаром
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, targets, num_epochs):
        for epoch in range(num_epochs):
            # Forward pass
            hidden_layer_input = np.dot(inputs, self.weights_input_hidden)
            hidden_layer_output = self.sigmoid(hidden_layer_input)

            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
            predicted_output = self.sigmoid(output_layer_input)

            # Backward pass
            output_error = targets - predicted_output
            output_delta = output_error * self.sigmoid_derivative(predicted_output)

            hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
            hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(hidden_layer_output)

            # Update weights
            self.weights_hidden_output += hidden_layer_output.T.dot(output_delta) * self.learning_rate
            self.weights_input_hidden += inputs.T.dot(hidden_layer_delta) * self.learning_rate

    def predict(self, inputs):
        hidden_layer_input = np.dot(inputs, self.weights_input_hidden)
        hidden_layer_output = self.sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        predicted_output = self.sigmoid(output_layer_input)

        return predicted_output

# Вхідні дані - перші 13 чисел для навчання
input_data = np.array([2.65, 5.60, 1.21, 5.48, 0.73, 4.08, 1.88, 5.31, 0.78, 4.36, 1.71, 5.62, 0.43])

# Останні два числа для тестування
test_data = np.array([4.21, 1.21])

# Нормалізація даних
input_data_normalized = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))

# Створення та навчання нейронної мережі
neural_network = NeuralNetwork(input_size=1, hidden_size=5, output_size=1, learning_rate=0.0098)
neural_network.train(input_data_normalized[:-2].reshape(-1, 1), input_data_normalized[2:].reshape(-1, 1), num_epochs=15000)

# Прогнозування на тестових даних
test_data_normalized = (test_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
predicted_output_normalized = neural_network.predict(test_data_normalized.reshape(-1, 1))

# Денормалізація прогнозованих даних
predicted_output = predicted_output_normalized * (np.max(input_data) - np.min(input_data)) + np.min(input_data)

# Виведення результатів
print("Тестові дані:", test_data)
print("Прогнозовані дані:", predicted_output)

# # Графік прогнозованих та реальних значень
# plt.plot(input_data, label='Дані для навчання')
# plt.scatter(len(input_data), test_data[0], color='red', label='Тестові дані (x1)')
# plt.scatter(len(input_data) + 1, test_data[1], color='blue', label='Тестові дані (x2)')
# plt.plot(len(input_data) + 2, predicted_output, marker='o', markersize=8, color='green', label='Прогнозовані дані')
# plt.legend()
# plt.show()


