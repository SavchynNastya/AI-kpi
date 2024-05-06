import numpy as np


def logical_predict(input_data: np.ndarray, wages: np.ndarray, threshold: float):
    return (np.dot(input_data, wages) >= threshold).astype(int)


def main():
    # And
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    wages = np.array([[1], [1]])
    t = 1.5

    result = logical_predict(input_data, wages, t)

    print('And')
    print(' x1 x2 y ')
    print(np.concatenate((input_data, result), axis=1))

    # Or
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    wages = np.array([[1], [1]])
    t = 0.5

    result = logical_predict(input_data, wages, t)

    print('Or')
    print(' x1 x2 y ')
    print(np.concatenate((input_data, result), axis=1))

    # Not
    input_data = np.array([[0], [1]])
    wages = np.array([[-1.5]])
    t = -1

    result = logical_predict(input_data, wages, t)

    print('Not')
    print('  x y ')
    print(np.concatenate((input_data, result), axis=1))

    # Xor
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    wages1 = np.array([[1, -1], [-1, 1]]) 
    wages2 = np.array([[1], [1]])
    t = 0.5

    h = logical_predict(input_data, wages1, t) # [0,0],[-1,1],[1,-1],[0,0]
    print('hidden')
    print(h)
    result = logical_predict(h, wages2, t)

    print('Xor')
    print(' x1 x2 y ')
    print(np.concatenate((input_data, result), axis=1))


main()