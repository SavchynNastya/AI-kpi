# import random


# class KohonenNetwork:
#     def __init__(self, input_size, output_size):
#         self.weights = self.initialize_weights(input_size, output_size)

#     @staticmethod
#     def initialize_weights(input_size, output_size):
#         return [[random.uniform(0, 1) for _ in range(input_size)] for _ in range(output_size)]

#     def find_winner(self, input_data):
#         distances = [sum((input_data[j] - self.weights[i][j]) ** 2 for j in range(len(input_data))) for i in range(len(self.weights))]
#         winner_index = distances.index(min(distances))
#         return winner_index

#     def update_weights(self, input_data, winner_index, learning_rate):
#         for j in range(len(input_data)):
#             self.weights[winner_index][j] += learning_rate * (input_data[j] - self.weights[winner_index][j])

#     def train_kohonen(self, X, epochs, learning_rate):
#         for epoch in range(epochs):
#             total_error = 0
#             for i in range(len(X)):
#                 input_data = X[i]
#                 winner_index = self.find_winner(input_data)
#                 self.update_weights(input_data, winner_index, learning_rate)
#                 total_error += sum((input_data[j] - self.weights[winner_index][j]) ** 2 for j in range(len(input_data)))

#             if epoch % 1000 == 0:
#                 print(f"Epoch {epoch}, Error: {total_error}")