import taichi as ti
import argparse
import numpy as np
import os
import csv
import math
import random

# Initialize taichi
real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

# Global variables
dim = 2
n_particles = 5000
n_solid_particles = 0
n_actuators = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 10
mu = E
la = E
max_steps = 2048
steps = 1024
gravity = 3.8

# Function to create field objects
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

# Define fields
actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()
loss = scalar()
n_sin_waves = 4
weights = scalar()
bias = scalar()
x_avg = vec()
actuation = scalar()
actuation_omega = 20
act_strength = 4

# Import core functions from snow.py (ensure snow.py is in the same directory)
from snow import (
    allocate_fields, clear_grid, p2g, grid_op, g2p, compute_actuation,
    advance, forward, Scene, create_complex_robot, visualize, load_params_from_csv
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration CSV file')
    parser.add_argument('--steps', type=int, default=1500, help='Number of simulation steps to run')
    parser.add_argument('--output', type=str, default='visualization', help='Output folder for visualization frames')
    parser.add_argument('--interval', type=int, default=16, help='Frame interval for visualization')
    options = parser.parse_args()
    
    # Load configuration
    snowflake_params = load_params_from_csv(options.config)
    
    print("Loaded Snowflake Parameters:")
    for key, value in snowflake_params.items():
        print(f"{key}: {value}")
    
    # Initialize scene with complex robot
    scene = Scene()
    create_complex_robot(scene, snowflake_params)
    
    # Important: Allocate fields AFTER updating the global variables
    allocate_fields()
    
    # Initialize particle positions, deformation gradient, and velocity
    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]
    
    # Randomize the weights for interesting movement
    for i in range(n_actuators):
        for j in range(n_sin_waves):
            weights[i, j] = random.uniform(-1, 1)
        bias[i] = random.uniform(-0.5, 0.5)
    
    # Run forward simulation
    forward(options.steps)
    
    # Visualize at intervals
    os.makedirs(options.output, exist_ok=True)
    for s in range(15, options.steps, options.interval):
        visualize(s, options.output)
    
    print(f"Visualization frames saved to {options.output}/ folder")

if __name__ == '__main__':
    main()