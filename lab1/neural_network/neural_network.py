import numpy as np
from constants import STUDY_SPEED, ITERATION_NUMBER, ITERATION_INFO


class Sigmoid:
    def __call__(self, number: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-number))

    def derivative(self, number):
        sgm = self(number)
        return sgm * (1 - sgm)


class NeuralNetwork:
    def __init__(self, input_number: int = 3, neurons_number: int = 4, sigmoid: Sigmoid = Sigmoid()):
        self.input_number = input_number
        self.wages = neurons_number
        self.activate = sigmoid

    @property
    def activate(self):
        return self._activate

    @activate.setter
    def activate(self, sigmoid):
        self._activate = sigmoid
        
    @property
    def input_number(self):
        return self._input_number

    @input_number.setter
    def input_number(self, input_number):
        if input_number > 0:
            self._input_number = input_number
        else:
            raise ValueError("Input number can`t be less then 1")

    @property
    def wages(self):
        return self._wages

    @wages.setter
    def wages(self, neurons_number):
        self._wages = {}
        self._wages["hidden_w"] = np.random.random((self.input_number, neurons_number))
        self._wages["hidden_b"] = np.random.random((1, neurons_number))
        self._wages["out_w"] = np.random.random((neurons_number, 1))
        self._wages["out_b"] = np.random.random((1, 1))

        self._wages["hidden_w"] = (self._wages["hidden_w"] - 0.5) * 2 * np.sqrt(1 / self.input_number)
        self._wages["hidden_b"] = (self._wages["hidden_b"] - 0.5) * 2 * np.sqrt(1 / self.input_number)
        self._wages["out_w"] = (self._wages["out_w"] - 0.5) * 2 * np.sqrt(1 / neurons_number)
        self._wages["out_b"] = (self._wages["out_b"] - 0.5) * 2 * np.sqrt(1 / neurons_number)

    def forward_propagation(self, input_data: np.ndarray):
        result = {}
        result["hidden_s"] = np.dot(input_data, self._wages["hidden_w"]) + self.wages["hidden_b"]
        result["hidden_y"] = self.activate(result["hidden_s"])
        result["out"] = np.dot(result["hidden_y"], self.wages["out_w"]) + self.wages["out_b"]
        return result

    def back_propagation(self, input: np.ndarray, output: np.ndarray, result):
        delta_wages = {}
        delta_2 = result["out"] - output
        delta_wages["delta_out_w"] = np.dot(result["hidden_y"].T, delta_2)
        delta_wages["delta_out_b"] = np.sum(delta_2, axis=0, keepdims=True)
        delta_h1 = np.dot(delta_2, self.wages["out_w"].T)
        delta_1 = delta_h1 * self.activate.derivative(result["hidden_s"])
        delta_wages["delta_hidden_w"] = np.dot(input.T, delta_1)
        delta_wages["delta_hidden_b"] = np.sum(delta_1, axis=0, keepdims=True)

        return delta_wages

    def update_wages(self, delta_wages):
        self._wages["hidden_w"] -= STUDY_SPEED*delta_wages["delta_hidden_w"]
        self._wages["hidden_b"] -= STUDY_SPEED*delta_wages["delta_hidden_b"]
        self._wages["out_w"] -= STUDY_SPEED*delta_wages["delta_out_w"]
        self._wages["out_b"] -= STUDY_SPEED*delta_wages["delta_out_b"]

    def study(self, input_data: np.ndarray, output_data: np.ndarray):
        for i in range(ITERATION_NUMBER):
            result = self.forward_propagation(input_data)

            self.update_wages(self.back_propagation(input_data, output_data, result))

            if i % ITERATION_INFO == 0:
                total_error = np.sum((result["out"] - output_data) ** 2)
                print(str(i) + " iterations MISTAKE:" + str(total_error))
        
    def predict(self, input_data: np.ndarray):
        result = self.forward_propagation(input_data)
        return result["out"]

    def mean_squared_error(actual, predicted):
        n = len(actual)
        mse = np.sum((actual - predicted) ** 2) / n
        return mse
    