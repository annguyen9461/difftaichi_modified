import taichi as ti
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import gc
from net import (
    real, dx, inv_dx, dt, p_vol, E, mu, la,
    steps, gravity, target, n_sin_waves, act_strength, actuation_omega,
    Scene, visualize, load_params_from_csv, create_random_net, 
    clear_grid, compute_actuation, p2g, grid_op, g2p, forward,
    reset_fields
)

# Import needed global variables
from net import (
    n_grid, n_particles, n_solid_particles, n_actuators,
    x, v, F, C, actuator_id, particle_type, x_avg
)

def view_robot_from_csv(csv_filepath, output_folder="robot_visualization"):
    """
    Load a robot configuration from a CSV file and visualize its movement.
    """
    print(f"Loading configuration from: {csv_filepath}")
    
    # Load parameters from CSV
    net_params = load_params_from_csv(csv_filepath)
    print("Loaded parameters:", net_params)
    
    # Initialize scene
    scene = Scene()
    
    # Create the robot using the loaded parameters
    create_random_net(scene, net_params)
    
    # Finalize the scene setup (this also updates global n_actuators)
    scene.finalize()
    
    # Print values to verify
    print(f"Number of particles: {n_particles}")
    print(f"Number of solid particles: {n_solid_particles}")
    print(f"Number of actuators: {n_actuators}")
    
    # Initialize particle positions
    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]
    
    # Calculate initial center of mass position
    x_initial = 0.0
    initial_count = 0
    for i in range(scene.n_particles):
        if particle_type[i] == 1:
            x_initial += x[0, i][0]
            initial_count += 1
    if initial_count > 0:
        x_initial /= initial_count
    
    print(f"Initial center of mass x-position: {x_initial}")
    
    # Run the simulation
    forward()
    
    # Calculate final center of mass position
    x_final = x_avg[None][0]
    print(f"Final center of mass x-position: {x_final}")
    print(f"Distance traveled: {x_final - x_initial}")
    
    # Visualize the movement
    os.makedirs(output_folder, exist_ok=True)
    
    # Reset fields for visualization
    reset_fields()
    
    # Initialize particles again
    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]
    
    # Visualize at regular intervals
    for s in range(0, steps, steps // 20):
        if s < steps - 1:
            clear_grid()
            compute_actuation(s)
            p2g(s)
            grid_op()
            g2p(s)
            if s % (steps // 20) == 0:
                visualize(s + 1, output_folder)
    
    print(f"Visualization frames saved to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View a robot from a CSV configuration file")
    parser.add_argument("--csv", type=str, required=True, help="Path to the CSV configuration file")
    parser.add_argument("--output", type=str, default="robot_visualization", help="Output folder for visualization frames")
    
    args = parser.parse_args()
    
    # Set default values for global variables before allocate_fields is called
    # This is crucial to avoid the "dimension 0" error
    global n_actuators, n_particles, n_solid_particles
    n_actuators = 5  # Start with a reasonable default
    n_particles = 1000  # Default value
    n_solid_particles = 1000  # Default value
    
    # Initialize Taichi
    ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)
    
    # Import allocate_fields after setting defaults for global variables
    from net import allocate_fields
    
    # Now allocate fields with non-zero values
    allocate_fields()
    
    # View the robot
    view_robot_from_csv(args.csv, args.output)