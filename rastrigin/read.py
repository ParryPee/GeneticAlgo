import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

def read_file(f_name):
    df = pd.read_csv(f_name)
    return df

def plot_graph(plotBest,plotCounter):
    plotBest =  [float(i) for i in plotBest]
    fig, ax = plt.subplots()
    maximumIndex = plotBest.index(min(plotBest))
    ax.plot(plotCounter, plotBest)

    def annot_max(x, y, ax=None):
        xmax = plotCounter[maximumIndex]
        ymax = min(plotBest)
        xmax = float(xmax)
        ymax = float(ymax)

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
    plt.ylim(0, 500)
    plt.show()

def main():
    df = read_file("./results.csv")
    # print(df.loc[df["score"].idxmin()])
    plotBest = df.loc[df["score"].idxmin()]["graph_bests"]
    plotBest = plotBest.strip('][').split(', ')
    plotCounter = df.loc[df["score"].idxmin()]["graph_generations"]
    plotCounter = plotCounter.strip('][').split(', ')
    print(df.loc[df["score"].idxmin()]["Best gene"])
    plot_graph(plotBest,plotCounter)


if __name__ == "__main__":
    main()