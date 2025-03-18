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
    if not hasattr(reset_fields, 'initialized'):
        allocate_fields()  # Ensure fields are allocated before resetting
        reset_fields.initialized = True
    
    # Reset field values
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
    view_snow2_path = "display.py"

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
        
    print(f"Saved fitness {fitness_score} for generation {generation} to {fitness_csv_file}")

def load_fitness_from_csv(fitness_csv_file):
    """
    Load previously evaluated configurations and their fitness scores from a CSV file.
    Returns a dictionary mapping parameter tuples to fitness scores.
    """
    evaluated_configs = {}
    
    if not os.path.isfile(fitness_csv_file):
        print(f"Fitness file {fitness_csv_file} does not exist yet")
        return evaluated_configs
        
    try:
        with open(fitness_csv_file, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Get the header
            
            # Find indices of important columns
            param_indices = []
            fitness_idx = None
            generation_idx = None
            
            for i, col in enumerate(header):
                if col == "Fitness":
                    fitness_idx = i
                elif col == "Generation":
                    generation_idx = i
                else:
                    param_indices.append(i)
            
            if fitness_idx is None:
                print(f"Error: 'Fitness' column not found in {fitness_csv_file}")
                return evaluated_configs
            
            # Process rows
            for row in reader:
                if len(row) <= fitness_idx:
                    continue  # Skip incomplete rows
                
                # Extract parameter values
                param_values = []
                for idx in param_indices:
                    if idx < len(row):
                        param_values.append(float(row[idx]) if '.' in row[idx] else int(row[idx]))
                
                # Create a tuple for the dictionary key
                param_tuple = tuple(param_values)
                
                # Extract fitness
                fitness_score = float(row[fitness_idx])
                
                # Store in dictionary
                evaluated_configs[param_tuple] = fitness_score
                
        print(f"Loaded {len(evaluated_configs)} configurations from {fitness_csv_file}")
    except Exception as e:
        print(f"Error loading fitness data from {fitness_csv_file}: {e}")
    
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

def reallocate_fields_if_needed():
    global n_particles, n_actuators
    if not hasattr(reallocate_fields_if_needed, 'last_n_particles'):
        reallocate_fields_if_needed.last_n_particles = n_particles
        reallocate_fields_if_needed.last_n_actuators = n_actuators
        allocate_fields()
        return

    if (n_particles != reallocate_fields_if_needed.last_n_particles or
        n_actuators != reallocate_fields_if_needed.last_n_actuators):
        print(f"Reallocating fields: n_particles={n_particles}, n_actuators={n_actuators}")
        allocate_fields()
        reallocate_fields_if_needed.last_n_particles = n_particles
        reallocate_fields_if_needed.last_n_actuators = n_actuators

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
            config_file = os.path.join(gen_folder, f"structure_{len(population)}.csv")
            save_params_to_csv(params, config_file)
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

    # Loop through the remaining generations
    for generation in range(1, num_generations):
        current_gen = generation + 1  # Generation starts from 1, loop starts from 0
        print(f"Generation {current_gen}/{num_generations}")
        
        # Reset global state
        global n_particles, n_solid_particles, n_actuators
        n_particles = 0
        n_solid_particles = 0
        n_actuators = 0

        # Clear Taichi fields and reset the runtime
        ti.reset()
        ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

        # Generate a valid configuration to initialize fields
        scene = Scene()
        params = randomize_snowflake_params()
        create_snowflake_structure(scene, params)
        scene.finalize()
        
        # Reallocate fields if n_particles or n_actuators has changed
        reallocate_fields_if_needed()
        
        # Create folder for the current generation
        gen_folder = os.path.join(run_folder, f"gen_{current_gen}")
        os.makedirs(gen_folder, exist_ok=True)
        print(f"Created generation folder: {gen_folder}")

        # Generate the next generation
        population = generate_next_generation(run_folder, population_size)

        # Evaluate fitness for each individual in the population
        fitness_scores = []
        for idx, params in enumerate(population):
            idx_1_based = idx + 1  # 1-based index for file naming
            print(f"Evaluating structure {idx_1_based}/{population_size}")
            params_tuple = tuple(params.values())
            
            # Always save the configuration file, regardless of evaluation status
            config_file = os.path.join(gen_folder, f"structure_{idx_1_based}.csv")
            print(f"Saving configuration to: {config_file}")
            try:
                save_params_to_csv(params, config_file)
                print(f"Successfully saved configuration to {config_file}")
            except Exception as e:
                print(f"Failed to save configuration to {config_file}: {e}")

            # Skip evaluation if the configuration has already been evaluated
            if params_tuple in evaluated_configs:
                print(f"Configuration already evaluated. Fitness: {evaluated_configs[params_tuple]}")
                fitness_score = evaluated_configs[params_tuple]
                fitness_scores.append((fitness_score, params))
                all_fitness_scores.append((fitness_score, params))
                
                # Make sure to save this to fitness CSV with the current generation
                save_fitness_to_csv(params, fitness_score, fitness_csv_file, generation=current_gen)
                
                # Record fitness in the gen folder's fitness_scores.txt
                with open(os.path.join(gen_folder, "fitness_scores.txt"), "a") as f:
                    f.write(f"Structure {idx_1_based}: {fitness_score} (previously evaluated)\n")
                continue

            if test_configuration(params):  # Only evaluate if the configuration is valid
                fitness_score = evaluate_fitness(params)
                evaluated_configs[params_tuple] = fitness_score
                save_fitness_to_csv(params, fitness_score, fitness_csv_file, generation=current_gen)
                fitness_scores.append((fitness_score, params))
                all_fitness_scores.append((fitness_score, params))

                with open(os.path.join(gen_folder, "fitness_scores.txt"), "a") as f:
                    f.write(f"Structure {idx_1_based}: {fitness_score}\n")
            else:
                print(f"Configuration {idx_1_based} is invalid. Skipping evaluation.")
                with open(os.path.join(gen_folder, "fitness_scores.txt"), "a") as f:
                    f.write(f"Structure {idx_1_based}: INVALID CONFIGURATION\n")

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
    Generate the next generation of configurations based ONLY on the most recent generation's top performers.
    The top 70% performers from the latest generation are kept and the bottom 30% are replaced with random configurations.
    """
    # Load fitness data
    fitness_csv_file = os.path.join(run_folder, "fitness.csv")
    
    # Check if the fitness CSV file exists
    if not os.path.exists(fitness_csv_file):
        print(f"Warning: {fitness_csv_file} not found. Generating random configurations.")
        return [randomize_snowflake_params() for _ in range(population_size)]
    
    # Load evaluated configurations and their fitness scores
    fitness_scores = []
    latest_generation = None
    
    try:
        with open(fitness_csv_file, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Get the header
            
            # Find the indices of key columns
            param_names = [col for col in header if col != "Fitness" and col != "Generation"]
            fitness_idx = header.index("Fitness") if "Fitness" in header else None
            generation_idx = header.index("Generation") if "Generation" in header else None
            
            if fitness_idx is None:
                print("Error: 'Fitness' column not found in CSV.")
                return [randomize_snowflake_params() for _ in range(population_size)]
            
            if generation_idx is None:
                print("Error: 'Generation' column not found in CSV.")
                return [randomize_snowflake_params() for _ in range(population_size)]
            
            # First find the latest generation number
            all_configs = []
            for row in reader:
                if len(row) <= max(fitness_idx, generation_idx):
                    continue  # Skip incomplete rows
                
                # Get generation number
                gen_num = int(float(row[generation_idx]))
                
                # Update latest generation
                if latest_generation is None or gen_num > latest_generation:
                    latest_generation = gen_num
                
                all_configs.append(row)
            
            # Now filter configs by latest generation
            if latest_generation is not None:
                print(f"Focusing on generation {latest_generation} only")
                for row in all_configs:
                    if len(row) <= max(fitness_idx, generation_idx):
                        continue
                    
                    gen_num = int(float(row[generation_idx]))
                    
                    # Only process configs from the latest generation
                    if gen_num == latest_generation:
                        # Extract parameters
                        params = {}
                        for i, name in enumerate(param_names):
                            if i < len(row):
                                if name in ["depth", "num_sub_branches", "actuation_start", "ptype"]:
                                    params[name] = int(float(row[i]))
                                else:
                                    params[name] = float(row[i])
                        
                        # Extract fitness
                        fitness = float(row[fitness_idx])
                        
                        # Add to fitness scores
                        fitness_scores.append((fitness, params))
    except Exception as e:
        print(f"Error loading fitness data: {e}")
        return [randomize_snowflake_params() for _ in range(population_size)]
    
    if not fitness_scores:
        print("No valid fitness scores found for the latest generation. Generating random configurations.")
        return [randomize_snowflake_params() for _ in range(population_size)]

    # Sort the population by fitness (descending order)
    fitness_scores.sort(key=lambda x: x[0], reverse=True)
    print(f"Found {len(fitness_scores)} configurations from generation {latest_generation}")

    # Calculate how many configurations to keep (70%) and how many to generate randomly (30%)
    num_to_keep = max(1, int(population_size * 0.7))
    num_random = population_size - num_to_keep
    
    print(f"Keeping top {num_to_keep} configurations and generating {num_random} random ones")

    # Select the top performers for the next generation (only from the latest generation)
    top_performers = [params for (fitness, params) in fitness_scores[:num_to_keep]]
    
    # Start with the top performers
    new_population = top_performers.copy()
    
    # Fill remaining slots (30%) with crossover/mutation first, 
    # then use random configurations as the fallback
    max_attempts = 50  # Maximum attempts to find a valid configuration
    slots_remaining = population_size - len(new_population)
    
    # Try crossover and mutation first
    crossover_attempts = 0
    while len(new_population) < population_size and crossover_attempts < max_attempts:
        if len(top_performers) >= 2:
            parent1, parent2 = random.sample(top_performers, 2)
            child = generate_valid_configuration(lambda: mutate(crossover(parent1, parent2)), 10)
            if child:
                new_population.append(child)
                continue
        
        crossover_attempts += 1
        
    # If we still have slots to fill, use random configurations as fallback
    while len(new_population) < population_size:
        # Random will always work eventually, so we add it without checking
        # Just directly generate random parameters without testing
        new_random = randomize_snowflake_params()
        new_population.append(new_random)
        print("Added random configuration as fallback")

    # Ensure we return exactly population_size configurations
    return new_population[:population_size]

def generate_valid_configuration(config_generator, max_attempts):
    """
    Helper function to generate a valid configuration using the provided generator function.
    """
    attempts = 0
    while attempts < max_attempts:
        config = config_generator()
        if test_configuration(config):
            return config
        attempts += 1
    return None

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

def main(population_size=20, num_generations=100, max_particles=10000):
    """
    Main function to run the evolutionary optimization with configurable parameters.
    
    Args:
        population_size: The size of the population in each generation (default: 20)
        num_generations: The number of generations to evolve (default: 100)
        max_particles: Maximum number of particles allowed (default: 10000)
    """
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
    
    # Save run parameters to file for reference
    with open(os.path.join(run_folder, "run_params.txt"), "w") as f:
        f.write(f"Population Size: {population_size}\n")
        f.write(f"Number of Generations: {num_generations}\n")
        f.write(f"Max Particles: {max_particles}\n")

    fitness_csv_file = os.path.join(run_folder, "fitness.csv")
    
    # Make sure the file exists with headers before loading
    if not os.path.exists(fitness_csv_file):
        # Create an empty file with headers
        with open(fitness_csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Create header with all possible parameter names plus Fitness and Generation
            example_params = randomize_snowflake_params()
            header = list(example_params.keys()) + ["Fitness", "Generation"]
            writer.writerow(header)
        print(f"Created empty fitness file: {fitness_csv_file}")
    
    # Now load the evaluated configs
    evaluated_configs = load_fitness_from_csv(fitness_csv_file)

    best_params = evolutionary_optimization(population_size, num_generations, run_folder, max_particles, fitness_csv_file, evaluated_configs)
    
    # Save the best configuration
    best_config_file = os.path.join(run_folder, "best_config.csv")
    save_params_to_csv(best_params, best_config_file)
    print(f"Saved best configuration to: {best_config_file}")
    
    # Print summary of the run
    print("\nEvolutionary Optimization Complete")
    print(f"Run folder: {run_folder}")
    print(f"Best configuration: {best_params}")
    if tuple(best_params.values()) in evaluated_configs:
        print(f"Best fitness score: {evaluated_configs[tuple(best_params.values())]}")
    else:
        print("Best fitness score unknown. Configuration may not have been evaluated.")

if __name__ == '__main__':
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run evolutionary optimization for snowflake structures')
    parser.add_argument('--population', type=int, default=20, help='Population size for each generation (default: 20)')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations to evolve (default: 100)')
    parser.add_argument('--max-particles', type=int, default=10000, help='Maximum number of particles allowed (default: 10000)')
    
    args = parser.parse_args()
    
    # Run with parsed arguments
    main(
        population_size=args.population,
        num_generations=args.generations,
        max_particles=args.max_particles
    )