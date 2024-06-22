import random
import numpy as npy
import math
import matplotlib.pyplot as plt
import time
import pandas as pd
import csv

bestEver = float("inf")
counter = 0
plotBest = []
plotCounter = []
bestGene = []
bestIndiv = 0

def randomPop(p):
    pop = [[0]*3 for x in range(p)]
    for i in range(p):
        for j in range(3):
            pop[i][j] = random.uniform(-5.12, 5.12)
    return pop


def func(x):
    total = 0
    temp = 10 * len(x)

    for i in range(len(x)):
        total += (pow(x[i], 2) - (10 * math.cos(2 * math.pi * x[i])))
    total += temp
    return total


def fitness(pop):
    global bestEver
    global plotBest
    global plotCounter
    global counter
    global bestIndiv
    global bestGene
    best = float("inf")
    fpop = []
    total = 0
    for i in range(len(pop)):
        x = pop[i]
        fxy = func(x)
        # print(fxy)
        fpop.insert(i, fxy)
        total += fxy
        if fxy < best:
            best = fxy
        if fxy < bestEver:
            bestGene = x
            bestEver = fxy
            bestIndiv = i

    plotBest.append(best)
    plotCounter.append(counter)
    return fpop, total

# Linear crossover


def crossover(p1, p2):
    x0 = 0
    x1 = 0
    x2 = 0
    ind = 0
    best = float("inf")
    f_ind = []
    y = [[0]*3 for i in range(3)]

    for i in range(3):
        y[0][i] = (0.5*p1[i] + 0.5*p2[i])
        y[1][i] = (1.5*p1[i] - 0.5*p2[i])
        y[2][i] = (-0.5*p1[i] + 1.5*p2[i])
    for i in range(3):
        for j in range(3):
            if y[i][j] > 5.12:
                y[i][j] = 5.12
            if y[i][j] < -5.12:
                y[i][j] = -5.12

    for i in range(3):
        f_ind.insert(i, func(y[i]))
        if f_ind[i] < best:
            best = f_ind[i]
            ind = i

    return y[ind]


def linCrossover(p1, p2):
    temp = p1
    best = float("inf")
    mutated_k_gene = []
    k = random.randrange(0, 2)
    rand_k_gene1, rand_k_gene2 = p1[k], p2[k]
    mutated_k_gene.append(0.5 * rand_k_gene1 + 0.5 * rand_k_gene2)
    mutated_k_gene.append(1.5 * rand_k_gene1 - 0.5 * rand_k_gene2)
    mutated_k_gene.append(-0.5 * rand_k_gene1 + 1.5 * rand_k_gene2)
    holder = []

    for i in range(3):
        if mutated_k_gene[i] > 5.12:
            mutated_k_gene[i] = 5.12
        if mutated_k_gene[i] < -5.12:
            mutated_k_gene[i] = -5.12
        temp[k] = mutated_k_gene[i]
        value = func(temp)
        if value < best:
            holder = temp
            best = value
    return holder


def choose(fpop, total):
    ptotal = 0
    pb = []
    for x in range(len(fpop)):
        pb.insert(x, math.exp(12*(1-(fpop[x]/total)))/100)
# pb.insert(x,(fpop[x]/total))
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


def mutation(pop):
    new = 0
    pb = 0
    for i in range(len(pop)):
        pb = random.uniform(0.0000, 1.0000)
        for j in range(3):
            if pb <= 0.20:
                pop[i][j] = random.uniform(-5.12, 5.12)
    return pop


def main():
    global counter
    global bestEver
    global plotCounter
    global plotBest
    global bestIndiv
    global bestGene

    # p = int(input("\nEnter the population size:"))
    p = 400
    # g = int(input("Enter the Maximum number of generations: "))
    g = 1000
    start_time = time.time()
    pop = randomPop(p)
    for i in range(g):
        fpop, total = fitness(pop)
        popNew = [[0]*3 for x in range(p)]
        for j in range(p):
            c = choose(fpop, total)
            popNew[j] = linCrossover(pop[c[0]], pop[c[1]])
        # print(popNew)
        popNew = mutation(popNew)
        pop = popNew
        print(f"Generation : {i}")
        counter += 1
    print(f"Best Record: {bestEver}")
    print(f"Best Gene: {bestGene}")
    print("--- %s seconds ---" % (time.time() - start_time))
    data = [bestEver, plotBest.index(min(plotBest)), bestIndiv,bestGene,plotBest,plotCounter]
    with open("./results.csv", "a",newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(data)
    bestEver = float("inf")
    counter = 0
    plotBest = []
    plotCounter = []
    bestGene = []
    bestIndiv = 0
    # fig, ax = plt.subplots()
    # maximumIndex = plotBest.index(min(plotBest))
    # ax.plot(plotCounter, plotBest)

    # def annot_max(x, y, ax=None):
    #     xmax = plotCounter[maximumIndex]
    #     ymax = min(plotBest)
    #     text = "x={:.3f}, y={:.3f}".format(xmax, ymax)
    #     if not ax:
    #         ax = plt.gca()
    #     bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    #     arrowprops = dict(
    #         arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    #     kw = dict(xycoords='data', textcoords="axes fraction",
    #               arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    #     ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)

    # annot_max(plotCounter, plotBest)
    # # plt.plot(PLOTCOUNTER, PLOTBEST)
    # plt.ylim(0, 500)
    # # plt.show()


if (__name__ == "__main__"):
    header = ["score", "Best generation", "Best Individual","Best gene","graph_bests","graph_generations"]
    with open(r"./results.csv", "w",newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    for i in range(100):
        main()
    data = pd.read_csv("./results.csv",header=0)
