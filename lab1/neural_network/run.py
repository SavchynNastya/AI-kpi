from neural_network import NeuralNetwork
import numpy as np


def main():
    np.random.seed(121)

    neural_net = NeuralNetwork()

    inpt_data = np.array([[2.65, 5.60, 1.21], [5.60, 1.21, 5.48], [1.21, 5.48, 0.73], [5.48, 0.73, 4.08],
                          [0.73, 4.08, 1.88], [4.08, 1.88, 5.31], [1.88, 5.31, 0.78], [5.31, 0.78, 4.36],
                          [0.78, 4.36, 1.71], [4.36, 1.71, 5.62]])

    reslt_data = np.array([[5.48], [0.73], [4.08], [1.88], [5.31], [0.78], [4.36], [1.71], [5.62], [0.43]])

    inpt_data_check = np.array([[1.71, 5.62, 0.43], [5.62, 0.43, 4.21]])
    reslt_data_check = np.array([[4.21], [1.21]])

    neural_net.study(input_data=inpt_data, output_data=reslt_data)
    print("\nExpect: " + str(np.concatenate(reslt_data_check, axis=None)))
    
    predictions = neural_net.predict(inpt_data_check)
    print("\nPredictions: " + str(np.concatenate(predictions, axis=None)))

    mse = NeuralNetwork.mean_squared_error(reslt_data_check, predictions)
    print("\nMean Squared Error (MSE): " +  str(mse))

if __name__ == '__main__':
    main()