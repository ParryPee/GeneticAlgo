import random
import string
import math
from random import randint
import matplotlib.pyplot as plt
import numpy as np

bestEver = float("inf")
bestGeneration = 0
bestIndiv = 0
counter = 0
plotBest = []
plotCounter = []
bestGene = []

# Generating population


def randomPop(p):
    pop = [[0]*3 for x in range(p)]
    for i in range(p):
        for j in range(3):
            pop[i][j] = random.uniform(-10, 10)
    return pop


# Rosenbrock function
def func(x):
    total = 0
    for i in range(len(x)-1):
        total += 100*math.pow((x[i+1] - math.pow(x[i], 2)),
                              2) + math.pow((1- x[i]), 2)
    return total


def fAndPrint(pop):
    global counter
    global bestEver
    global bestIndiv
    global bestGeneration
    global bestGene
    ind = 0
    best = float("inf")
    total = 0
    fpop = []
    for i in range(len(pop)):
        x = pop[i]
        fxy = func(x)
        fpop.insert(i, fxy)
        total += fxy
        # print("Ind%i : %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f," %
        #       (i, x[0], x[1], x[2], x[3], x[4], x[5], x[6]), end='')
        # print(" %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f," %
        #       (x[7], x[8], x[9], x[10], x[11], x[12], x[13]), end='')
        # print(" %.2f, %.2f, %.2f, %.2f, %.2f, %.2f" %
        #       (x[14], x[15], x[16], x[17], x[18], x[19]), end='')
        # print("  ---> f(Ind): %f" % (fxy))
        if fxy < best:
            best = fxy
            ind = i
        if fxy < bestEver:
            bestEver = fxy
            bestIndiv = i
            bestGene = x
            bestGeneration = counter

    print("\n Best Individual: %i = %f || Average: %f" %
          (ind, best, (total/len(pop))))
    plotBest.append(best)
    plotCounter.append(counter)
    return fpop, total


# Selection


def choose(fpop, total):
    ptotal = 0
    pb = []
    for x in range(len(fpop)):
        pb.insert(x, math.exp(12*(1-(fpop[x]/total)))/100)
#		pb.insert(x,(fpop[x]/total))
    c = []
    m = min(pb)

    for x in range(len(pb)):
        pb[x] = pb[x] - (m/1.1)
    for x in range(len(pb)):
        ptotal += pb[x]

    for x in range(3):
        z = 0
        ran = random.uniform(0.0000, ptotal)
        for i in range(len(pb)):
            z += pb[i]
            if z > ran:
                break
        c.insert(x, i)

    return c

# LINEAR CROSSOVER


def crossover(p1, p2):
    x0 = 0
    x1 = 0
    x2 = 0
    ind = 0
    best = float("inf")
    absBest = 1
    f_ind = []
    y = [[0]*3 for i in range(3)]

    for i in range(3):
        y[0][i] = (0.5*p1[i] + 0.5*p2[i])
        y[1][i] = (1.5*p1[i] - 0.5*p2[i])
        y[2][i] = (-0.5*p1[i] + 1.5*p2[i])
    for i in range(3):
        for j in range(3):
            if y[i][j] > 10:
                y[i][j] = 10
            if y[i][j] < -10:
                y[i][j] = -10

    for i in range(3):
        f_ind.insert(i, func(y[i]))
        if f_ind[i]< best:
            best = f_ind[i]
            ind = i

    return y[ind]

# Linear mutation


def mutation(pop):
    new = 0
    pb = 0
    for i in range(len(pop)):
        pb = random.uniform(0.0000, 1.0000)
        for j in range(2):
            if pb <= 0.20:
                pop[i][j] = random.uniform(-10, 10)
    return pop


def main():
    global counter
    global plotBest
    global plotCounter
    global bestGeneration
    global bestGene
    print("\n------ROSENBROCK GA : 3 DIMENSIONAL------")
    p = int(input("\nEnter the population size:"))
    g = int(input("\nEnter maximum number of generations:"))
    pop = randomPop(p)
    teste = [0.1]*20
    for i in range(g):
        counter += 1
        fpop, total = fAndPrint(pop)
        popnew = [[0]*3 for x in range(p)]
        idx = np.argsort(fpop)
        prev_best = pop[idx[0]]
        for j in range(p):
            c = choose(fpop, total)
            popnew[j] = crossover(pop[c[0]], pop[c[1]])
        pop = popnew
        pop = mutation(pop)
        fpop = []
        for x in range(len(pop)):
            fpop.append(func(pop[x]))
        idx = np.argsort(fpop)
        curr_best = pop[idx[0]]
        if func(prev_best) < func(curr_best):
            pop[idx[1]] = prev_best
        print("\Generation (%i): " % i)
    print(f"Best Record: {bestEver}\n")
    print(f"Best Individual: {bestIndiv}\n")
    print(f"Best Generation: {bestGeneration}\n")
    print(f"Best Gene: {bestGene}")
    # print(plotBest)
    # print(plotCounter)
    fig, ax = plt.subplots()
    maximumIndex = plotBest.index(min(plotBest))
    ax.plot(plotCounter, plotBest)

    def annot_max(x, y, ax=None):
        xmax = plotCounter[maximumIndex]
        ymax = min(plotBest)
        text = "x={:.3f}, y={:.3f}".format(xmax, ymax)
        if not ax:
            ax = plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops = dict(
            arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords='data', textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
        ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)

    annot_max(plotCounter, plotBest)
    # plt.plot(PLOTCOUNTER, PLOTBEST)
    plt.ylim(0, 5)
    plt.show()


if __name__ == "__main__":
    main()
