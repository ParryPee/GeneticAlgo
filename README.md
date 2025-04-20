# Genetic Algorithms for Optimization Problems

This repository contains implementations of genetic algorithms (GA) to solve complex optimization problems, specifically the Rastrigin and Rosenbrock functions. These functions are common benchmarks for optimization algorithms due to their challenging landscapes.

## Overview

Genetic algorithms are a type of evolutionary algorithm inspired by natural selection. They work by:
1. Creating a population of potential solutions (individuals)
2. Evaluating each solution's fitness
3. Selecting the best individuals to reproduce
4. Creating new solutions through crossover and mutation
5. Repeating the process for multiple generations

This project demonstrates GA implementations for two classic optimization test functions:

### Rastrigin Function
- A non-convex function with many local minima
- Global minimum at `f(x) = 0` when all `x = 0`
- Implementation uses 3-dimensional space with value range between -5.12 and 5.12

![Rastrigin function](https://github.com/SunnySideUpSun/GeneticAlgo/assets/78585950/1fd6e333-c7e5-42e1-8e9e-88b610887b8f)

### Rosenbrock Function
- A non-convex function with a narrow curved valley
- Global minimum at `f(x) = 0` when all `x = 1`
- Implementation uses 3-dimensional space with value range between -10 and 10

![Rosenbrock function](https://github.com/SunnySideUpSun/GeneticAlgo/assets/78585950/a92cfee5-64d1-4755-a211-e8c28ee945f6)

## Features

- **Multiple Implementations**: Standard GA and GA with elitism
- **Linear Crossover**: Different crossover strategies implemented
- **Adaptive Mutation**: 20% mutation rate with boundary constraints
- **Visualization**: Results plotted using matplotlib
- **Benchmarking**: Save and analyze results across multiple runs

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/GeneticAlgo.git
cd GeneticAlgo
pip install -r requirements.txt
```

## Required Dependencies

- numpy
- matplotlib
- pandas
- random
- time

## Usage

### Running the Rastrigin Function Optimization

```bash
cd rastrigin
python app.py  # Standard implementation
python "app + elitism.py"  # Implementation with elitism
```

### Running the Rosenbrock Function Optimization

```bash
cd rosenbrock
python "app + elitism.py"
```

### Analyzing Results

For the Rastrigin implementation that saves results to CSV:

```bash
cd rastrigin
python read.py
```

## Implementation Details

### Rastrigin Function

- **Population size**: Configurable (default 400 for standard, 10000 for elitism version)
- **Dimensions**: 3D optimization
- **Selection**: Fitness-proportionate selection
- **Crossover**: Linear crossover with three possible child points
- **Mutation**: 20% mutation rate
- **Boundary constraint**: Values constrained to [-5.12, 5.12]
- **Termination**: Fixed number of generations (default 1000)

### Rosenbrock Function

- **Population size**: User-configurable
- **Dimensions**: 3D optimization
- **Selection**: Exponential fitness-proportionate selection
- **Crossover**: Linear crossover with three possible child points
- **Mutation**: 20% mutation rate
- **Boundary constraint**: Values constrained to [-10, 10]
- **Elitism**: Best solutions preserved between generations
- **Termination**: User-defined maximum generations

## Results

The algorithms visualize optimization progress by plotting the best fitness value in each generation. Annotations highlight the generation number and fitness value when the best solution was found.

For the Rastrigin implementation with data collection (app.py), results from 100 runs are stored in a CSV file and can be analyzed using read.py.

## Contributing

Contributions are welcome! Some potential improvements:
- Implement other optimization test functions
- Add different selection methods
- Experiment with adaptive mutation rates
- Add parallel processing for faster execution
- Improve visualization options
