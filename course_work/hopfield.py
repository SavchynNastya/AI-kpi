import numpy as np


# Функція для генерації випадкового шуму
def generate_noise(shape):
    return np.random.choice([-1, 1], size=shape)

# Функція для додавання шуму до вхідних даних
def add_noise_to_data(data, noise_level):
    noise = generate_noise(data.shape) * noise_level
    return np.clip(data + noise, -1, 1)

# Функція для відновлення даних за допомогою мережі Хопфілда
def hopfield_network(data):
    weights = np.dot(data.T, data)
    np.fill_diagonal(weights, 0)  # Нульові ваги на діагоналі
    output = np.sign(np.dot(data, weights))
    return output

# Функція для оцінки точності відновлення
def calculate_accuracy(original, reconstructed):
    return np.mean(original == reconstructed)

# Задаємо параметри
num_samples = 100
input_size = 1000
noise_levels = [0, 0.1, 0.3, 0.5]

# Генеруємо випадкові зразки даних
data = np.random.choice([-1, 1], size=(num_samples, input_size))

# Проводимо дослідження
for noise_level in noise_levels:
    noisy_data = add_noise_to_data(data, noise_level)
    reconstructed_data = hopfield_network(noisy_data)
    accuracy = calculate_accuracy(data, reconstructed_data)
    print(f"Noise Level: {noise_level}, Accuracy: {accuracy}")
