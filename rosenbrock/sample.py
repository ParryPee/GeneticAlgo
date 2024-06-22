from deap import base, creator, tools, algorithms
import numpy as np

# Define the Rosenbrock function


def rosenbrock(x):
    x = np.asarray(x)
    return np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


# Create the DEAP optimization toolbox
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Define the search space
BOUND_LOW, BOUND_UP = -5.0, 5.0
NDIM = 2

# Create the individual and population
toolbox.register("attr_float", np.random.uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate,
                 creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the evaluation function
toolbox.register("evaluate", rosenbrock)

# Register the genetic operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Set the seed for reproducibility
np.random.seed(42)

# Define the population size and number of generations
POPULATION_SIZE = 50
NGEN = 50

# Create the initial population
population = toolbox.population(n=POPULATION_SIZE)

# Define the statistics to track the progress of the optimization
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)

# Perform the optimization
population, logbook = algorithms.eaSimple(
    population, toolbox, cxpb=0.5, mutpb=0.2, ngen=NGEN, stats=stats, verbose=True)

# Print the final result
best_individual = tools.selBest(population, k=1)[0]
print("Best individual:", best_individual)
print("Fitness:", rosenbrock(best_individual))
