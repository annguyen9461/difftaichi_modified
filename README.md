# Snowflake Evolution

An evolutionary algorithm implementation for evolving virtual snowflake structures with locomotion capabilities using the Taichi differentiable physics engine.

## Overview

This repository contains a genetic algorithm implementation that evolves branching, snowflake-like structures capable of movement through a simulated physical environment. The algorithm optimizes the structures for horizontal movement across a surface using a Material Point Method (MPM) physics simulation by modifying diffmpm.py from https://github.com/taichi-dev/difftaichi/tree/master.

The project consists of two main components:
- `generate.py`: The main evolutionary algorithm that generates and evolves structures
- `display.py`: A visualization and simulation subprocess that tests the validity of generated configurations

## Features

- Genetic algorithm with configurable population size and generation count
- Branching tree-like structures with customizable parameters
- Fitness evaluation based on horizontal movement
- Spring connections between particles
- Random parameter generation
- Crossover and mutation operations
- Automatic saving of generation data and best configurations

## Setup

1. Clone the repository:
```bash
git clone git@github.com:annguyen9461/difftaichi_modified.git
cd difftaichi_modified
```

2. Set up a virtual environment with the required dependencies:
```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

The requirements file includes:
- taichi>=1.1.0
- matplotlib
- numpy
- opencv-python
- scipy
- imageio
- torch
- torchvision

## Usage

Run the main evolutionary algorithm with default parameters:
```bash
python generate.py
```

### Command-line Options

The algorithm supports the following command-line arguments:

- `--population`: Population size for each generation (default: 20)
- `--generations`: Number of generations to evolve (default: 100)
- `--max-particles`: Maximum number of particles allowed (default: 10000)

Example with custom parameters:
```bash
python generate.py --population 30 --generations 50 --max-particles 15000
```

## Output

The algorithm creates a folder named `run_YYYYMMDD_HHMMSS` for each run with the following structure:

```
run_20250318_123456/
├── fitness.csv                  # All configurations with fitness and generation number
├── best_config.csv              # The best configuration found
├── run_params.txt               # Parameters used for this run
├── gen_1/
│   ├── structure_1.csv
│   ├── structure_2.csv
│   └── fitness_scores.txt       # Fitness scores for this generation
├── gen_2/
└── ...
```

Each generation folder contains the structure parameters and fitness scores for that generation.

## How It Works

1. The algorithm generates a population of random structures
2. Each structure is simulated using the physics engine to evaluate its movement capability
3. The best performers are selected to produce the next generation through:
   - Copying the top performers (70%)
   - Applying crossover and mutation to generate new variants
   - Adding random structures as needed
4. This process continues for the specified number of generations
5. The best configuration found is saved for future use

This will run the physics simulation and display the structure's movement in real-time.

## Demo

https://github.com/user-attachments/assets/82d271bc-63bf-4bde-9e91-77852746568b


Music from #InAudio: https://inaudio.org/Nostalgia
