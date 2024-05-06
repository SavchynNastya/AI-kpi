# class ArtificialNeuron:
#     def __init__(self, num_inputs, weights, bias):
#         self.num_inputs = num_inputs
#         self.weights = weights
#         self.bias = bias

#     def activate(self, inputs):
#         # Кастомна функція активації для вказаної таблиці істинності
#         weighted_sum = sum([inputs[i] * self.weights[i] for i in range(self.num_inputs)]) + self.bias
#         return int(weighted_sum >= 0.5)  # Змінено поріг активації на 0.5

# def logical_function_example():
#     # Ініціалізація нейрону для моделювання вказаної логічної функції
#     neuron = ArtificialNeuron(3, [1, 1, 1], -2)

#     # Вхідні дані для тестування
#     test_inputs = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]]

#     # Вивід результатів тестування
#     print("x1\tx2\tx3\ty")
#     for input_data in test_inputs:
#         result = neuron.activate(input_data)
#         print(f"{input_data[0]}\t{input_data[1]}\t{input_data[2]}\t{result}")

# # Виклик функції для моделювання вказаної логічної функції
# logical_function_example()
