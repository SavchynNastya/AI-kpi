import numpy as np
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, x_min, x_max, num_generations, population_size, num_parents, mutation_rate):
        self.x_min = x_min
        self.x_max = x_max
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_parents = num_parents
        self.mutation_rate = mutation_rate

        self.optimal_solution_max = 0
        self.optimal_value_max = 0

        self.optimal_solution_min = 0
        self.optimal_value_min = 0
    
    def func(self, x):
        return 5 * np.sin(10 * x) * np.sin(3 * x) / (x ** x)
    
    def generate_population(self):
        return np.random.uniform(low=self.x_min, high=self.x_max, size=(self.population_size,))
    
    def calculate_fitness(self, population):
        return self.func(population)
    
    def select_best_individuals(self, population, fitness, reverse=False):
        fitness = -fitness if reverse else fitness
        sorted_indices = np.argsort(fitness)
        return np.expand_dims(population[sorted_indices[-self.num_parents:]], axis=1)
    
    def crossover(self, parents):
        offspring_size = (self.population_size - self.num_parents, 1)
        offspring = np.zeros(offspring_size)
        crossover_point = offspring_size[0] // 2
        parents = parents.reshape((-1, 1))  
        for i in range(offspring_size[0]):
            parent1_idx = i % parents.shape[0]
            parent2_idx = (i + 1) % parents.shape[0]
            offspring[i, :] = np.concatenate((parents[parent1_idx, :crossover_point], parents[parent2_idx, crossover_point:]))
        return offspring
    
    def mutate(self, offspring_crossover):
        for idx in range(offspring_crossover.shape[0]):
            if np.random.random() < self.mutation_rate:
                random_value = np.random.uniform(low=self.x_min, high=self.x_max)
                offspring_crossover[idx] = random_value
        return offspring_crossover
    
    def form_population(self, population, reverse=False):
        for generation in range(self.num_generations):
            fitness = self.calculate_fitness(population)
            parents = self.select_best_individuals(population, fitness, reverse=reverse)
            offspring_crossover = self.crossover(parents)
            offspring_mutation = self.mutate(offspring_crossover)
            parent_index = 0
            for i in range(parents.shape[0]):
                population[i] = parents[parent_index]
                parent_index += 1
            for i in range(parents.shape[0], self.population_size):
                population[i] = offspring_mutation[i - parents.shape[0]]
    
            print("Generation \tMin \t\tMax")
            print(f"{generation}\t\t{np.min(fitness):.5f}\t{np.max(fitness):.5f}")
        return population
        
    
    def optimize(self):
        population_max = self.generate_population()
        population_max = self.form_population(population=population_max, reverse=False)
        
        self.optimal_solution_max = population_max[np.argmax(self.calculate_fitness(population_max))]
        self.optimal_value_max = self.func(self.optimal_solution_max)
        print("Y(x)max", self.optimal_value_max)
        print("Xmax:", self.optimal_solution_max)

        population_min = self.generate_population()
        population_min = self.form_population(population=population_min, reverse=True)

        self.optimal_solution_min = population_min[np.argmin(self.calculate_fitness(population_min))]
        self.optimal_value_min = self.func(self.optimal_solution_min)
        print("Y(x)min", self.optimal_value_min)
        print("Xmin:", self.optimal_solution_min)

        return self.optimal_solution_max, self.optimal_value_max, self.optimal_solution_min, self.optimal_value_min
    
    def plot_function(self):
        x = np.linspace(self.x_min, self.x_max, 1000)
        y = self.func(x)
        plt.plot(x, y, label='Y(x)=5*sin(10*x)*sin(3*x)/(x^x)')
        plt.scatter(self.optimal_solution_max, self.optimal_value_max, color='red', label='Максимум')
        plt.scatter(self.optimal_solution_min, self.optimal_value_min, color='green', label='Мінімум')
        plt.title('Графік функції')
        plt.xlabel('x')
        plt.ylabel('Y(x)')
        plt.legend()
        plt.grid(True)
        plt.show()


x_min = 0
x_max = 8
num_generations = 200
population_size = 100
num_parents = 20
mutation_rate = 0.1

ga = GeneticAlgorithm(x_min, x_max, num_generations, population_size, num_parents, mutation_rate)
optimal_solution_max, optimal_value_max, optimal_solution_min, optimal_value_min = ga.optimize()
ga.plot_function()
