import numpy as np
import random
import matplotlib.pyplot as plt
import time
MAXGENERATIONS = 2000

# Rastrigin function

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Initialize population


def init_population(pop_size, dim):
    population = np.random.uniform(-5.12, 5.12, (pop_size, dim))
    return population

# Evaluation


def evaluate(population):
    fitness = np.apply_along_axis(rastrigin, 1, population)
    return fitness

# Selection


def selection(population, fitness):
    idx = np.argsort(fitness)
    parents = population[idx[:2]]
    return parents

# Linear crossover


def linear_crossover(parents, alpha=0.5,beta = 1.5):
    child = np.zeros(parents.shape[1])
    child2 = np.zeros(parents.shape[1])
    for i in range(parents.shape[1]):
        child[i] = alpha*parents[0][i] + (1-alpha)*parents[1][i]
    # for i in range(parents.shape[1]):
    #     child2 = beta*parents[0][i] + (1-beta)*parents[1][i]
    return child,child2

# Mutation


def mutation(child, mutation_rate=0.01):
    for i in range(len(child)):
        if random.random() < mutation_rate:
            child[i] += np.random.normal()
    return child

# Main loop


def genetic_algorithm(pop_size=600, dim=3, generations=MAXGENERATIONS):
    population = init_population(pop_size, dim)
    bestRecord = []
    bestFitness = []
    start_time = time.time()
    for i in range(generations):
        fitness = evaluate(population)
        parents = selection(population, fitness)
        prev_bests = parents[0]
        child,child2 = linear_crossover(parents)
        population[np.argmin(fitness)] = child
        # population[np.argmax(fitness)] = child2
        bestRecord.append(population[np.argmin(evaluate(population))])
        bestFitness.append(np.min(fitness))
        #elitism
        # fitness = evaluate(population)
        # parents = selection(population,fitness)
        # curr_best = parents[0]
        # if rastrigin(prev_bests) < rastrigin(curr_best):
        #     population[np.argmin(fitness)] = prev_bests
    best_solution = population[np.argmin(evaluate(population))]
    print(f"Best ever: {np.min(bestFitness)}")
    print("Time taken for program: ",time.time() - start_time)
    return best_solution,bestRecord,bestFitness

i = [i for i in range(MAXGENERATIONS)]
best_solution,best_record,best_fitness = genetic_algorithm()
# print(best_fitness[0])
# print(len(i))

fig, ax = plt.subplots()
maximumIndex = best_fitness.index(min(best_fitness))
ax.plot(i, best_fitness)

def annot_max(x, y, ax=None):
    xmax = i[maximumIndex]
    ymax = min(best_fitness)
    text = "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(
        arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)

annot_max(i, best_fitness)
# # plt.plot(PLOTCOUNTER, PLOTBEST)
# plt.ylim(0, 500)
# # plt.show()
# plt.plot(i,best_fitness)
print("Best Ever solution: ",best_record[np.argmin(best_fitness)])
print("Generation number: ", np.argmin(best_fitness))
# print("Best Final solution: ", best_solution)
plt.show()
