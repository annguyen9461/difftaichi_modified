import taichi as ti
import argparse
import numpy as np
import matplotlib.pyplot as plt

import os
import csv
from datetime import datetime
import random
import math
import subprocess
import gc  # For garbage collection
import time  # For time.sleep()

real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

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
# TODO: update
mu = E
la = E
max_steps = 2048
steps = 1024
gravity = 3.8
target = [0.8, 0.2]

MAX_PARTICLES = 10000  # Set a reasonable limit

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

actuator_id = ti.field(ti.i32)  # Remove shape parameter
particle_type = ti.field(ti.i32)  # Remove shape parameter
x = ti.Vector.field(dim, dtype=real)  # Remove shape parameter
v = ti.Vector.field(dim, dtype=real)  # Remove shape parameter
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

# Field declarations (without allocation)
actuator_id = None
particle_type = None
x, v = None, None
grid_v_in, grid_m_in = None, None
grid_v_out = None
C, F = None, None
loss = None
x_avg = None
weights = None
bias = None
actuation = None

def allocate_fields():
    global actuator_id, particle_type, x, v, C, F, grid_v_in, grid_m_in, grid_v_out, actuation, weights, bias, loss, x_avg
    
    print(f"Allocating fields with n_particles={n_particles}, n_actuators={n_actuators}")
    
    # First create the fields
    actuator_id = ti.field(dtype=ti.i32)
    particle_type = ti.field(dtype=ti.i32)
    x = ti.Vector.field(dim, dtype=real)
    v = ti.Vector.field(dim, dtype=real)
    C = ti.Matrix.field(dim, dim, dtype=real)
    F = ti.Matrix.field(dim, dim, dtype=real)
    grid_v_in = ti.Vector.field(dim, dtype=real)
    grid_m_in = ti.field(dtype=real)
    grid_v_out = ti.Vector.field(dim, dtype=real)
    actuation = ti.field(dtype=real)
    weights = ti.field(dtype=real)
    bias = ti.field(dtype=real)
    loss = ti.field(dtype=real)
    x_avg = ti.Vector.field(dim, dtype=real)
    
    # Then place them
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.ij, (max_steps, n_particles)).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)
    ti.root.place(loss, x_avg)
    
    # Set up automatic differentiation
    ti.root.lazy_grad()
    
def reset_fields():
    x.fill(0)
    v.fill(0)
    C.fill(0)
    F.fill(0)
    grid_v_in.fill(0)
    grid_m_in.fill(0)
    grid_v_out.fill(0)
    actuation.fill(0)
    weights.fill(0)
    bias.fill(0)
    loss[None] = 0
    x_avg[None] = [0, 0]
    ti.sync()

import os
import csv
from datetime import datetime
import re
import glob

def evaluate_fitness(params):
    """
    Evaluate the fitness of a configuration by running the simulation and extracting the fitness score
    from the latest view_<datetime> folder.
    """
    # Save the parameters to a temporary CSV file
    save_params_to_csv(params, "param-test.csv")
    print(f"Testing configuration: {params}")

    # Define the path to view_snow2.py
    view_snow2_path = "/home/annguyen/classes/artificial-life/difftaichi_modified/final/view_snow2.py"

    try:
        # Run view_snow2.py as a subprocess
        result = subprocess.run(
            ["python", view_snow2_path],
            check=True,  # Raise CalledProcessError if the subprocess returns a non-zero exit code
            capture_output=True,
            text=True
        )
        print("Configuration test passed.")

        # Find the latest view_<datetime> folder
        latest_view_folder = find_latest_view_folder()
        if latest_view_folder:
            # Extract fitness score from the run_data.csv file in the latest view folder
            fitness_score = extract_fitness_from_csv(os.path.join(latest_view_folder, "run_data.csv"))
            return fitness_score
        else:
            print("No view_<datetime> folder found.")
            return -1000  # Return a low fitness score if no folder is found

    except subprocess.CalledProcessError as e:
        # Log the error and continue
        print(f"Configuration test failed with error (exit code {e.returncode}):")
        print(e.stderr)
        return -1000  # Return a low fitness score for invalid configurations
    except Exception as e:
        # Catch any other exceptions (e.g., file not found, permission errors)
        print(f"Unexpected error while testing configuration: {e}")
        return -1000  # Return a low fitness score for any other error
    
def find_latest_view_folder():
    """
    Find the latest view_<datetime> folder in the current directory.
    """
    # Get all folders matching the view_<datetime> pattern
    view_folders = glob.glob("view_*")
    
    # Filter folders that match the exact pattern
    view_folders = [folder for folder in view_folders if re.match(r"view_\d{8}_\d{6}", folder)]
    
    if not view_folders:
        return None  # No matching folders found
    
    # Sort folders by creation time (most recent first)
    view_folders.sort(key=os.path.getmtime, reverse=True)
    
    # Return the latest folder
    return view_folders[0]

def extract_fitness_from_csv(csv_file):
    """
    Extract the final fitness score from the run_data.csv file.
    """
    try:
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            last_row = None
            for row in reader:
                last_row = row  # Get the last row
            if last_row:
                return float(last_row[1])  # Return the fitness score from the last row
            else:
                print("No data found in run_data.csv")
                return -1000  # Return a low fitness score if no data is found
    except FileNotFoundError:
        print(f"File {csv_file} not found.")
        return -1000  # Return a low fitness score if the file is not found
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return -1000  # Return a low fitness score for any other error


def save_fitness_to_csv(params, fitness_score, fitness_csv_file, generation=None):
    """
    Save the configuration parameters and fitness score to a CSV file.
    If generation is provided, it will be included in the CSV.
    """
    file_exists = os.path.isfile(fitness_csv_file)
    with open(fitness_csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header if the file doesn't exist
            header = list(params.keys()) + ["Fitness"]
            if generation is not None:
                header.append("Generation")
            writer.writerow(header)
        
        # Write the parameters and fitness score
        row = list(params.values()) + [fitness_score]
        if generation is not None:
            row.append(generation)
        writer.writerow(row)

def load_fitness_from_csv(fitness_csv_file):
    """
    Load previously evaluated configurations and their fitness scores from a CSV file.
    """
    evaluated_configs = {}
    if os.path.isfile(fitness_csv_file):
        with open(fitness_csv_file, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip the header row
            for row in reader:
                params = {key: float(value) if "." in value else int(value) for key, value in zip(header[:-1], row[:-1])}
                fitness_score = float(row[-1])
                evaluated_configs[tuple(params.values())] = fitness_score
    return evaluated_configs


@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]


@ti.kernel
def clear_particle_grad():
    # for all time steps and all particles
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]


@ti.kernel
def clear_actuation_grad():
    for t, i in actuation:
        actuation[t, i] = 0.0


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        J = (new_F).determinant()
        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)

        act_id = actuator_id[p]

        act = actuation[f, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act = 0.0
        # ti.print(act)

        A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                     ti.Matrix.diag(2, la * (J - 1) * J)
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base +
                          offset] += weight * (mass * v[f, p] + affine @ dpos)
                grid_m_in[base + offset] += weight * mass


bound = 3
coeff = 0.5


@ti.kernel
def grid_op():
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        # This will allow easier movement in the x-direction
        if j < bound and v_out[1] < 0:
            # Reduce x-direction friction to facilitate horizontal movement
            v_out[0] *= 0.9  # Less friction in x-direction
            v_out[1] = 0
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                else:
                    lin = v_out.dot(normal)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + coeff * lin <= 0:
                            v_out[0] = 0
                            v_out[1] = 0
                        else:
                            v_out = (1 + coeff * lin / lit) * vit
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0

        grid_v_out[i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C


@ti.kernel
def compute_actuation(t: ti.i32):
    for i in range(n_actuators):
        act = 0.0
        for j in ti.static(range(n_sin_waves)):
            # Create a phase difference between actuators to encourage wave-like motion
            phase_offset = 2 * math.pi * i / n_actuators
            act += weights[i, j] * ti.sin(actuation_omega * t * dt + 
                                        2 * math.pi / n_sin_waves * j + 
                                        phase_offset)
        act += bias[i]
        actuation[t, i] = ti.tanh(act)

@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg[None], contrib * x[steps - 1, i])


# @ti.kernel
# def compute_loss():
#     dist = x_avg[None][0]
#     loss[None] = -dist

# 1. Modify the compute_loss function to prioritize horizontal movement
@ti.kernel
def compute_loss():
    x_avg_final = x_avg[None]
    x_avg_initial = ti.Vector([0.0, 0.0])
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg_initial, contrib * x[0, i])
    
    distance_traveled_x = x_avg_final[0] - x_avg_initial[0]
    loss[None] = -distance_traveled_x

@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    compute_actuation(s)
    p2g(s)
    grid_op()
    g2p(s)


@ti.ad.grad_for(advance)
def advance_grad(s):
    clear_grid()
    p2g(s)
    grid_op()

    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)
    compute_actuation.grad(s)


def forward(total_steps=steps):
    # simulation
    for s in range(total_steps - 1):
        advance(s)
    x_avg[None] = [0, 0]
    compute_x_avg()
    compute_loss()


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0
        self.connections = []  # Store spring/joint connections
        self.n_actuators = 1  # Initialize with at least 1 actuator
        self.springs = []  # Store spring connections

    def add_branching_structure(self, start_x, start_y, depth, branch_length, angle, actuation_start, ptype, params):
        """Create a recursive branching structure (e.g., snowflake-like) with thicker branches"""
        if depth <= 0:
            return

        # Add main branch
        end_x = start_x + branch_length * math.cos(angle)
        end_y = start_y + branch_length * math.sin(angle)

        # Add particles along the branch
        n_points = max(5, int(branch_length / dx * 4))  # Increase particle density
        prev_particle_idx = None  # Track the previous particle index for spring connections

        for i in range(n_points):
            t = i / (n_points - 1)
            x_pos = start_x + t * (end_x - start_x)
            y_pos = start_y + t * (end_y - start_y)

            # Add particles in a perpendicular direction to create thickness
            for j in range(-1, 2):  # Add 3 particles in a line perpendicular to the branch
                offset_x = -j * params["thickness"] * math.sin(angle)
                offset_y = j * params["thickness"] * math.cos(angle)

                # Add particle
                self.x.append([x_pos + offset_x + self.offset_x, y_pos + offset_y + self.offset_y])
                self.actuator_id.append(actuation_start)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)
                if actuation_start != -1:
                    self.n_actuators = max(self.n_actuators, actuation_start + 1)

                # Add spring connection to previous particle in the same line
                if prev_particle_idx is not None and j == 0:
                    self.add_spring(prev_particle_idx, self.n_particles - 1, params["stiffness"], params["damping"])

                # Add spring connections between particles in the perpendicular direction
                if j != -1:
                    self.add_spring(self.n_particles - 1, self.n_particles - 2, params["stiffness"], params["damping"])

            prev_particle_idx = self.n_particles - 1

        # Create sub-branches
        for i in range(params["num_sub_branches"]):
            sub_angle = angle + i * params["sub_branch_angle"]
            new_length = branch_length * params["sub_branch_length_ratio"]
            self.add_branching_structure(end_x, end_y, depth - 1, new_length, sub_angle, actuation_start + 1, ptype, params)

        # Add asymmetry to encourage directional movement (more branches on one side)
        for i in range(params["num_sub_branches"]):
            # Bias angles toward the right side to encourage rightward movement
            if i < params["num_sub_branches"] // 2:
                sub_angle = angle + i * params["sub_branch_angle"]
            else:
                # Make right-side branches slightly longer to create asymmetric rolling
                sub_angle = angle + i * params["sub_branch_angle"]
                new_length = branch_length * params["sub_branch_length_ratio"] * 1.2
                
            self.add_branching_structure(end_x, end_y, depth - 1, new_length, sub_angle, 
                                        actuation_start + 1, ptype, params)
    def finalize(self):
        global n_particles, n_solid_particles, n_actuators
        
        # Cap particles if needed
        if self.n_particles > MAX_PARTICLES:
            print(f"Warning: Capping particles from {self.n_particles} to {MAX_PARTICLES}")
            self.n_particles = MAX_PARTICLES
            
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        n_actuators = max(1, self.n_actuators)  # Ensure at least one actuator
        
        print('n_particles:', n_particles)
        print('n_solid_particles:', n_solid_particles)
        print('n_actuators:', n_actuators)
    
    def set_offset(self, x, y):
        """Set position offset for the scene"""
        self.offset_x = x
        self.offset_y = y

    def add_spring(self, p1, p2, stiffness, damping):
        """Add a spring connection between two particles"""
        self.springs.append({
            'p1': p1,
            'p2': p2,
            'stiffness': stiffness,
            'damping': damping,
            'rest_length': np.linalg.norm(np.array(self.x[p1]) - np.array(self.x[p2]))
        })

gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)


def randomize_snowflake_params():
    snowflake_params = {
        "start_x": random.uniform(0.1, 0.3),
        "start_y": random.uniform(0.4, 0.6),
        "depth": random.randint(2, 5),  # Control recursion depth
        "branch_length": random.uniform(0.05, 0.3),  # Shorter branches
        "angle": random.uniform(0, 2 * math.pi),
        "thickness": random.uniform(0.005, 0.01),
        "stiffness": random.uniform(400.0, 600.0),
        "damping": random.uniform(0.03, 0.07),
        "num_sub_branches": random.randint(2, 5),  # Fewer branches
        "sub_branch_angle": random.uniform(math.pi / 4, math.pi / 2),
        "sub_branch_length_ratio": random.uniform(0.5, 0.7),
        "actuation_start": 0,
        "ptype": 1,
    }
    return snowflake_params

# Function to save parameters to a CSV file
def save_params_to_csv(params, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter", "Value"])  # Write header
        for key, value in params.items():
            writer.writerow([key, value])

# Function to test a configuration by running view_snow2.py
def test_configuration(params):
    """
    Test a configuration by running view_snow2.py as a subprocess.
    If the subprocess errors out, log the error and return False.
    """
    # Save the parameters to a temporary CSV file
    save_params_to_csv(params, "param-test.csv")
    print(f"Testing configuration: {params}")

    # Define the path to view_snow2.py in the main project directory
    view_snow2_path = "/home/annguyen/classes/artificial-life/difftaichi_modified/final/view_snow2.py"

    try:
        # Run view_snow2.py from the main project directory
        result = subprocess.run(
            ["python", view_snow2_path],
            check=True,  # Raise CalledProcessError if the subprocess returns a non-zero exit code
            capture_output=True,
            text=True
        )
        print("Configuration test passed.")
        return True
    except subprocess.CalledProcessError as e:
        # Log the error and continue
        print(f"Configuration test failed with error (exit code {e.returncode}):")
        print(e.stderr)
        return False
    except Exception as e:
        # Catch any other exceptions (e.g., file not found, permission errors)
        print(f"Unexpected error while testing configuration: {e}")
        return False
    
def load_params_from_csv(filename):
    """Load snowflake parameters from a CSV file."""
    params = {}
    with open(filename, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            key, value = row
            # Convert value to the appropriate type
            if key in ["depth", "num_sub_branches", "actuation_start", "ptype"]:
                params[key] = int(value)
            elif key in ["start_x", "start_y", "branch_length", "angle", "thickness", "stiffness", "damping", "sub_branch_angle", "sub_branch_length_ratio", "right_bias"]:
                params[key] = float(value)
            else:
                params[key] = value
    
    # Add right_bias if missing
    if "right_bias" not in params:
        params["right_bias"] = 1.2
        
    return params

def crossover(parent1, parent2):
    child = {}
    for key in parent1.keys():
        if random.random() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child

def mutate(params):
    for key in params.keys():
        if random.random() < 0.1:  # 10% mutation rate
            if isinstance(params[key], int):
                params[key] += random.randint(-1, 1)
            elif isinstance(params[key], float):
                params[key] += random.uniform(-0.1, 0.1)
    return params

def reset_fields():
    if not hasattr(reset_fields, 'initialized'):
        allocate_fields()  # Ensure fields are allocated before resetting
        reset_fields.initialized = True
    """Reset Taichi fields to their initial state"""
    print("Resetting Taichi fields...")
    global x, v, C, F, grid_v_in, grid_m_in, grid_v_out, actuation, weights, bias, loss, x_avg
    
    # Clear matrices first to release memory
    x.fill(0)
    v.fill(0)
    C.fill(0)
    F.fill(0)
    
    # Reset matrices to proper values
    for f in range(max_steps):
        for i in range(n_particles):
            F[f, i] = [[1, 0], [0, 1]]
    
    # Reset grid fields
    grid_v_in.fill(0)
    grid_m_in.fill(0)
    grid_v_out.fill(0)
    
    # Reset actuation fields
    actuation.fill(0)
    weights.fill(0)
    bias.fill(0)
    
    # Reset scalar fields
    loss[None] = 0
    x_avg[None] = [0, 0]
    
    # Force synchronization to make sure everything is cleared
    ti.sync()

import os
from datetime import datetime
import shutil

def create_snowflake_structure(scene, params):
    """Create a snowflake-like structure using the provided parameters"""
    scene.set_offset(0.5, 0.5)  # Center the structure
    start_x = params["start_x"]
    start_y = params["start_y"]
    depth = params["depth"]
    branch_length = params["branch_length"]
    angle = params["angle"]
    actuation_start = params["actuation_start"]
    ptype = params["ptype"]

    # Introduce some randomness in the initial angles
    initial_angles = [angle + random.uniform(-math.pi/6, math.pi/6) for _ in range(params["num_sub_branches"])]

    # Add the snowflake-like branching structure
    for initial_angle in initial_angles:
        scene.add_branching_structure(start_x, start_y, depth, branch_length, initial_angle, actuation_start, ptype, params)

    scene.finalize()

def add_branching_structure(self, start_x, start_y, depth, branch_length, angle, actuation_start, ptype, params):
    """Create a recursive branching structure with more control over parameters"""
    if depth <= 0:
        return

    # Add main branch
    end_x = start_x + branch_length * math.cos(angle)
    end_y = start_y + branch_length * math.sin(angle)

    # Add particles along the branch
    n_points = max(5, int(branch_length / dx * 4))  # Increase particle density
    prev_particle_idx = None  # Track the previous particle index for spring connections

    for i in range(n_points):
        t = i / (n_points - 1)
        x_pos = start_x + t * (end_x - start_x)
        y_pos = start_y + t * (end_y - start_y)

        # Add particles in a perpendicular direction to create thickness
        for j in range(-1, 2):  # Add 3 particles in a line perpendicular to the branch
            offset_x = -j * params["thickness"] * math.sin(angle)
            offset_y = j * params["thickness"] * math.cos(angle)

            # Add particle
            self.x.append([x_pos + offset_x + self.offset_x, y_pos + offset_y + self.offset_y])
            self.actuator_id.append(actuation_start)
            self.particle_type.append(ptype)
            self.n_particles += 1
            self.n_solid_particles += int(ptype == 1)
            if actuation_start != -1:
                self.n_actuators = max(self.n_actuators, actuation_start + 1)

            # Add spring connection to previous particle in the same line
            if prev_particle_idx is not None and j == 0:
                self.add_spring(prev_particle_idx, self.n_particles - 1, params["stiffness"], params["damping"])

            # Add spring connections between particles in the perpendicular direction
            if j != -1:
                self.add_spring(self.n_particles - 1, self.n_particles - 2, params["stiffness"], params["damping"])

        prev_particle_idx = self.n_particles - 1

    # Create sub-branches with more control over angles and lengths
    for i in range(params["num_sub_branches"]):
        sub_angle = angle + i * params["sub_branch_angle"]
        new_length = branch_length * params["sub_branch_length_ratio"]
        self.add_branching_structure(end_x, end_y, depth - 1, new_length, sub_angle, actuation_start + 1, ptype, params)

def load_parameters_from_csv(filename):
    params = {}
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if row:  # Skip empty rows
                key, value = row
                # Convert value to appropriate type
                if key in ["depth", "num_sub_branches", "actuation_start", "ptype"]:
                    params[key] = int(value)
                else:
                    params[key] = float(value)
    return params

# def main():
#     # Load parameters from CSV
#     params = load_parameters_from_csv("generated_parameters.csv")
    
#     # Create scene and add structure
#     scene = Scene()
#     create_snowflake_structure(scene, params)
    
#     # Rest of your simulation code...

# Main function to generate and test configurations

def main():
    # Initialize the scene
    scene = Scene()
    params = randomize_snowflake_params()
    create_snowflake_structure(scene, params)
    scene.finalize()  # Ensure n_actuators is updated
    
    # Allocate fields after n_actuators is set
    allocate_fields()
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"run_{current_time}"
    os.makedirs(run_folder, exist_ok=True)
    print(f"Created run folder: {run_folder}")

    fitness_csv_file = os.path.join(run_folder, "fitness.csv")
    evaluated_configs = load_fitness_from_csv(fitness_csv_file)

    population_size = 3
    num_generations = 10
    max_particles = 10000

    best_params = evolutionary_optimization(population_size, num_generations, run_folder, max_particles, fitness_csv_file, evaluated_configs)

def evolutionary_optimization(population_size, num_generations, run_folder, max_particles, fitness_csv_file, evaluated_configs):
     # Generate initial valid configuration to set n_particles and n_actuators
    scene = Scene()
    params = randomize_snowflake_params()
    create_snowflake_structure(scene, params)
    scene.finalize()
    
    # Allocate fields for the first generation
    allocate_fields()

    # Initialize population with valid configurations
    population = []
    gen_folder = os.path.join(run_folder, "gen_1")
    os.makedirs(gen_folder, exist_ok=True)
    print(f"Created generation folder: {gen_folder}")

    # Generate the first generation with valid configurations
    print("Generating first generation with valid configurations...")
    while len(population) < population_size:
        params = randomize_snowflake_params()
        params_tuple = tuple(params.values())

        # Skip evaluation if the configuration has already been evaluated
        if params_tuple in evaluated_configs:
            print(f"Configuration already evaluated. Fitness: {evaluated_configs[params_tuple]}")
            population.append(params)
            continue

        if test_configuration(params):  # Only add if the configuration is valid
            fitness_score = evaluate_fitness(params)
            evaluated_configs[params_tuple] = fitness_score
            save_fitness_to_csv(params, fitness_score, fitness_csv_file, generation=1)
            population.append(params)
            config_file = os.path.join(gen_folder, f"structure_{len(population)}.csv")
            print(f"Saving configuration to: {config_file}")
            try:
                save_params_to_csv(params, config_file)
                print(f"Successfully saved configuration to {config_file}")
            except Exception as e:
                print(f"Failed to save configuration to {config_file}: {e}")
        else:
            print("Configuration is invalid. Skipping...")

    all_fitness_scores = []

    for generation in range(1, num_generations):  # Start from generation 1
        print(f"Generation {generation + 1}/{num_generations}")
        gen_folder = os.path.join(run_folder, f"gen_{generation + 1}")
        os.makedirs(gen_folder, exist_ok=True)
        print(f"Created generation folder: {gen_folder}")

        # Generate the next generation
        population = generate_next_generation(run_folder, population_size)

        # Evaluate fitness for each individual in the population
        fitness_scores = []
        for idx in range(population_size):
            print(f"Evaluating structure {idx + 1}/{population_size}")
            params = population[idx]
            params_tuple = tuple(params.values())

            # Skip evaluation if the configuration has already been evaluated
            if params_tuple in evaluated_configs:
                print(f"Configuration already evaluated. Fitness: {evaluated_configs[params_tuple]}")
                fitness_scores.append((evaluated_configs[params_tuple], params))
                all_fitness_scores.append((evaluated_configs[params_tuple], params))
                continue

            if test_configuration(params):  # Only evaluate if the configuration is valid
                fitness_score = evaluate_fitness(params)
                evaluated_configs[params_tuple] = fitness_score
                save_fitness_to_csv(params, fitness_score, fitness_csv_file, generation=generation + 1)
                fitness_scores.append((fitness_score, params))
                all_fitness_scores.append((fitness_score, params))

                config_file = os.path.join(gen_folder, f"structure_{idx + 1}.csv")
                print(f"Saving configuration to: {config_file}")
                try:
                    save_params_to_csv(params, config_file)
                    print(f"Successfully saved configuration to {config_file}")
                except Exception as e:
                    print(f"Failed to save configuration to {config_file}: {e}")

                with open(os.path.join(gen_folder, "fitness_scores.txt"), "a") as f:
                    f.write(f"Structure {idx + 1}: {fitness_score}\n")
            else:
                print(f"Configuration {idx + 1} is invalid. Skipping evaluation.")

            gc.collect()
            time.sleep(0.2)

        # Force CUDA synchronization and garbage collection between generations
        ti.sync()
        gc.collect()

    # Return the best structure found across all generations
    valid_scores = [(score, params) for score, params in all_fitness_scores if score > -1000]
    if not valid_scores:
        return randomize_snowflake_params()  # Return a random structure if all failed

    # Sort by score only (not by the dictionary)
    valid_scores.sort(key=lambda x: x[0], reverse=True)
    return valid_scores[0][1]  # Return the params of the best structure

def generate_next_generation(run_folder, population_size):
    """
    Generate the next generation of configurations based on the fitness scores in fitness.csv.
    """
    # Load fitness data
    fitness_csv_file = os.path.join(run_folder, "fitness.csv")
    evaluated_configs = load_fitness_from_csv(fitness_csv_file)

    # Convert evaluated_configs into a list of (fitness, params) tuples
    fitness_scores = []
    for params_tuple, fitness in evaluated_configs.items():
        # Convert the tuple back into a dictionary
        params = {
            "start_x": params_tuple[0],
            "start_y": params_tuple[1],
            "depth": params_tuple[2],
            "branch_length": params_tuple[3],
            "angle": params_tuple[4],
            "thickness": params_tuple[5],
            "stiffness": params_tuple[6],
            "damping": params_tuple[7],
            "num_sub_branches": params_tuple[8],
            "sub_branch_angle": params_tuple[9],
            "sub_branch_length_ratio": params_tuple[10],
            "actuation_start": params_tuple[11],
            "ptype": params_tuple[12],
        }
        fitness_scores.append((fitness, params))

    # Sort the population by fitness (descending order)
    fitness_scores.sort(key=lambda x: x[0], reverse=True)

    # Drop the bottom 30% performers (rounding down)
    num_to_drop = int(0.3 * population_size)
    if num_to_drop > 0:
        fitness_scores = fitness_scores[:-num_to_drop]

    # Select the top performers for crossover and mutation
    top_performers = [params for (fitness, params) in fitness_scores]

    # Create new population through crossover and mutation
    new_population = top_performers.copy()

    # Add one new random configuration
    max_attempts = 100  # Maximum attempts to find a valid configuration
    attempts = 0
    while attempts < max_attempts:
        new_random = randomize_snowflake_params()
        if test_configuration(new_random):
            new_population.append(new_random)
            break
        attempts += 1
    else:
        print("Failed to generate a valid random configuration after maximum attempts.")

    # Fill the remaining slots with mutated and crossover configurations
    while len(new_population) < population_size:
        max_attempts = 100  # Maximum attempts to find a valid configuration
        attempts = 0
        while attempts < max_attempts:
            parent1, parent2 = random.sample(top_performers, 2)
            child = crossover(parent1, parent2)
            if test_configuration(child):
                new_population.append(child)
                break
            attempts += 1
        else:
            print("Failed to generate a valid crossover configuration after maximum attempts.")
            new_random = randomize_snowflake_params()
            if test_configuration(new_random):
                new_population.append(new_random)

        attempts = 0
        while attempts < max_attempts:
            parent1, parent2 = random.sample(top_performers, 2)
            child = mutate(parent1, parent2)
            if test_configuration(child):
                new_population.append(child)
                break
            attempts += 1
        else:
            print("Failed to generate a valid mutated configuration after maximum attempts.")
            new_random = randomize_snowflake_params()
            if test_configuration(new_random):
                new_population.append(new_random)

    return new_population

def generate_parameters():
    params = {
        "start_x": random.uniform(0.0, 0.2),  # Start within a reasonable x-range
        "start_y": random.uniform(0.2, 0.4),  # Adjusted to start closer to the ground
        "depth": random.randint(2, 4),  # Control recursion depth
        "branch_length": random.uniform(0.05, 0.2),  # Shorter branches for stability
        "angle": random.uniform(0.0, 2 * math.pi),  # Random initial angle
        "thickness": random.uniform(0.005, 0.02),  # Thickness of branches
        "stiffness": random.uniform(400.0, 600.0),  # Stiffness of the structure
        "damping": random.uniform(0.03, 0.07),  # Damping factor
        "num_sub_branches": random.randint(3, 7),  # Number of sub-branches
        "sub_branch_angle": random.uniform(0.5, 1.5),  # Angle of sub-branches
        "sub_branch_length_ratio": random.uniform(0.5, 0.7),  # Length ratio of sub-branches
        "actuation_start": 0,  # Actuation start index
        "ptype": 1  # Particle type (1 for solid)
    }
    return params

def generate_multiple_parameter_sets(num_sets):
    parameter_sets = []
    for _ in range(num_sets):
        params = generate_parameters()
        parameter_sets.append(params)
    return parameter_sets

def write_parameters_to_csv(parameter_sets, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter", "Value"])  # Write header
        for params in parameter_sets:
            for key, value in params.items():
                writer.writerow([key, value])
            writer.writerow([])  # Add an empty row between sets

if __name__ == '__main__':
    main()