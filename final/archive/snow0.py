import taichi as ti
import argparse
import numpy as np
import matplotlib.pyplot as plt

import os
import csv
from datetime import datetime
import random
import math

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

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

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

n_actuators = 1     # Default value


def allocate_fields():
    print(f"Allocating fields with n_actuators={n_actuators}, n_particles={n_particles}")
    if n_actuators <= 0 or n_particles <= 0:
        raise ValueError(f"Invalid field dimensions: n_actuators={n_actuators}, n_particles={n_particles}")
    
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)
    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, x_avg)

    ti.root.lazy_grad()


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
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
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
            act += weights[i, j] * ti.sin(actuation_omega * t * dt +
                                          2 * math.pi / n_sin_waves * j)
        act += bias[i]
        actuation[t, i] = ti.tanh(act)

    # Apply actuation forces to particles
    # for p in range(n_particles):
    #     act_id = actuator_id[p]
    #     if act_id != -1:
    #         # Apply a horizontal force to make the structure roll
    #         v[0, p][0] += actuation[t, act_id] * act_strength * dt
    #         # Apply a vertical force to make the structure bounce
    #         v[0, p][1] += actuation[t, act_id] * act_strength * dt * 0.5


@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg[None], contrib * x[steps - 1, i])


@ti.kernel
def compute_loss():
    dist = x_avg[None][0]
    loss[None] = -dist


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

    
    
    def add_spring(self, p1, p2, stiffness, damping):
        """Add a spring connection between two particles"""
        self.springs.append({
            'p1': p1,
            'p2': p2,
            'stiffness': stiffness,
            'damping': damping,
            'rest_length': np.linalg.norm(np.array(self.x[p1]) - np.array(self.x[p2]))
        })
    

    def add_rect(self, x, y, w, h, actuation, ptype=1):
        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    x + (i + 0.5) * real_dx + self.offset_x,
                    y + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)
                if actuation != -1:
                    self.n_actuators = max(self.n_actuators, actuation + 1)

    def add_circle(self, center_x, center_y, radius, actuation, ptype=1):
        global n_particles
        spacing = dx / 2  # Adjust spacing for denser particles
        # Iterate over a grid within the bounding box of the circle
        for i in range(int((center_x - radius) / spacing), int((center_x + radius) / spacing) + 1):
            for j in range(int((center_y - radius) / spacing), int((center_y + radius) / spacing) + 1):
                x_pos = center_x + (i * spacing - center_x)
                y_pos = center_y + (j * spacing - center_y)
                # Check if the point is inside the circle
                if (x_pos - center_x) ** 2 + (y_pos - center_y) ** 2 <= radius ** 2:
                    self.x.append([x_pos + self.offset_x, y_pos + self.offset_y])
                    self.actuator_id.append(actuation)
                    self.particle_type.append(ptype)
                    self.n_particles += 1
                    self.n_solid_particles += int(ptype == 1)
                    if actuation != -1:
                        self.n_actuators = max(self.n_actuators, actuation + 1)
    
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

    def finalize(self):
        global n_particles, n_solid_particles, n_actuators
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        n_actuators = self.n_actuators  # Update global n_actuators
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)
        print('n_actuators', n_actuators)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act

def create_complex_robot(scene, params):
    """Create a snowflake-like structure using the provided parameters"""
    start_x = params["start_x"]
    start_y = params["start_y"]
    depth = params["depth"]
    branch_length = params["branch_length"]
    angle = params["angle"]
    actuation_start = params["actuation_start"]
    ptype = params["ptype"]

    # Add the snowflake-like branching structure
    scene.add_branching_structure(start_x, start_y, depth, branch_length, angle, actuation_start, ptype, params)

    scene.finalize()

gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)


def visualize(s, folder):
    aid = actuator_id.to_numpy()
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    particles = x.to_numpy()[s]
    actuation_ = actuation.to_numpy()
    for i in range(n_particles):
        color = 0x111111
        if aid[i] != -1:
            act = actuation_[s - 1, int(aid[i])]
            color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)

    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')


def randomize_snowflake_params():
    snowflake_params = {
        "start_x": random.uniform(0.0, 0.2),
        "start_y": random.uniform(0.4, 0.6),
        "depth": random.randint(2, 4),
        "branch_length": random.uniform(0.03, 0.2),
        "angle": random.uniform(0, 2 * math.pi),
        "thickness": random.uniform(0.005, 0.015),
        "stiffness": random.uniform(400.0, 600.0),
        "damping": random.uniform(0.03, 0.07),
        "num_sub_branches": random.randint(4, 8),
        "sub_branch_angle": random.uniform(math.pi / 4, math.pi / 2),
        "sub_branch_length_ratio": random.uniform(0.5, 0.7),
        "actuation_start": 0,
        "ptype": 1
    }
    return snowflake_params

def save_params_to_csv(params, folder="config"):
    # Ensure the config folder exists
    os.makedirs(folder, exist_ok=True)
    
    # Generate a unique filename with the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"snowflake_config_{timestamp}.csv")
    
    # Save the parameters to a CSV file
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Parameter", "Value"])  # Write header
        for key, value in params.items():
            writer.writerow([key, value])
    
    print(f"Configuration saved to {filename}")

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
            elif key in ["start_x", "start_y", "branch_length", "angle", "thickness", "stiffness", "damping", "sub_branch_angle", "sub_branch_length_ratio"]:
                params[key] = float(value)
            else:
                params[key] = value
    return params

@ti.kernel
def reset_fields():
    """Reset the values of Taichi fields for a new simulation."""
    for i in range(n_particles):
        x[0, i] = [0.0, 0.0]
        v[0, i] = [0.0, 0.0]
        C[0, i] = [[0.0, 0.0], [0.0, 0.0]]
        F[0, i] = [[1.0, 0.0], [0.0, 1.0]]
    for i in range(n_actuators):
        bias[i] = 0.0
        for j in range(n_sin_waves):
            weights[i, j] = 0.0

def evaluate_robot(scene, params):
    # Reset fields for a new simulation
    reset_fields()
    
    # Initialize scene with the given parameters
    scene = Scene()
    create_complex_robot(scene, params)
    scene.finalize()  # Ensure this is called before allocate_fields()
    
    # Initialize particle positions, deformation gradient, and velocity
    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]
    
    # Simulate the robot
    forward()
    
    # Compute the loss (how well it moves across the y-axis)
    compute_loss()
    return loss[None]

def generate_population(parent_params, population_size=3):
    population = []
    
    # Check if parent_params is a list of parameters or a single parameter set
    if isinstance(parent_params, list):
        # If it's a list, use each parent to generate children
        for _ in range(population_size):
            # Randomly select one of the parents
            parent = random.choice(parent_params)
            # Mutate the selected parent's parameters
            child_params = mutate_params(parent)
            population.append(child_params)
    else:
        # If it's a single parameter set, use it to generate all children
        for _ in range(population_size):
            child_params = mutate_params(parent_params)
            population.append(child_params)
            
    return population

def evolve_robots(num_generations=3, population_size=3):
    # Create a folder for the current run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"run_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)
    
    # Create an initial population
    population = [randomize_snowflake_params() for _ in range(population_size)]
    
    for generation in range(num_generations):
        print(f"Generation {generation + 1}")
        
        # Evaluate each robot in the population
        fitness_scores = []
        for i, params in enumerate(population):
            print(f"Evaluating robot {i + 1}")
            fitness = evaluate_robot(Scene(), params)
            fitness_scores.append((fitness, params))
        
        # Sort the population by fitness (lower loss is better)
        fitness_scores.sort(key=lambda x: x[0])
        
        # Save configurations, loss values, and plots for the current generation
        save_generation_configs(generation + 1, fitness_scores, run_folder)
        
        # Select the top two robots' parameters
        top_two_params = [params for (_, params) in fitness_scores[:2]]
        
        # Generate the next generation using the top two robots as parents
        population = generate_population(top_two_params, population_size)
    
    # Return the best robot from the last generation
    return fitness_scores[0][1], run_folder

def mutate_params(params):
    # Ensure params is a dictionary
    if not isinstance(params, dict):
        raise TypeError("params must be a dictionary")
    
    # Create a copy of the parameters
    new_params = params.copy()
    
    # Mutate some parameters randomly
    new_params["branch_length"] += random.uniform(-0.01, 0.01)
    new_params["thickness"] += random.uniform(-0.001, 0.001)
    new_params["stiffness"] += random.uniform(-50.0, 50.0)
    new_params["damping"] += random.uniform(-0.01, 0.01)
    
    return new_params

def save_loss_values(generation, fitness_scores, folder):
    """Save the loss values for each robot in the generation."""
    filename = os.path.join(folder, f"gen{generation}_loss.csv")
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Rank", "Loss", "Robot ID"])
        for rank, (loss, params) in enumerate(fitness_scores):
            robot_id = f"gen{generation}_robot_{rank + 1}"
            writer.writerow([rank + 1, loss, robot_id])
    print(f"Loss values saved to {filename}")

import matplotlib.pyplot as plt

def save_loss_plot(generation, fitness_scores, folder):
    """Generate and save a plot of the loss values for the generation."""
    losses = [loss for (loss, _) in fitness_scores]
    ranks = range(1, len(losses) + 1)
    
    plt.figure()
    plt.plot(ranks, losses, marker="o", linestyle="-", color="b")
    plt.title(f"Loss Values for Generation {generation}")
    plt.xlabel("Rank")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plot_filename = os.path.join(folder, f"gen{generation}_loss_plot.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Loss plot saved to {plot_filename}")

def save_generation_configs(generation, fitness_scores, run_folder):
    """Save configurations, loss values, and plots for the generation."""
    # Create a folder for the current generation
    gen_folder = os.path.join(run_folder, f"gen{generation}")
    os.makedirs(gen_folder, exist_ok=True)
    
    # Save each robot's configuration
    for rank, (loss, params) in enumerate(fitness_scores):
        robot_id = f"gen{generation}_robot_{rank + 1}"
        filename = os.path.join(gen_folder, f"{robot_id}.csv")
        save_params_to_csv(params, filename)
    
    # Save loss values
    save_loss_values(generation, fitness_scores, gen_folder)
    
    # Save loss plot
    save_loss_plot(generation, fitness_scores, gen_folder)
    
    # Print the top robot in the generation
    top_loss, top_params = fitness_scores[0]
    print(f"Top robot in generation {generation}:")
    for key, value in top_params.items():
        print(f"{key}: {value}")
    print(f"Fitness (loss): {top_loss}")

def main():
    # Create the scene first
    scene = Scene()
    params = randomize_snowflake_params()
    create_complex_robot(scene, params)
    scene.finalize()  # This updates global n_particles, n_solid_particles, n_actuators
    
    # Then allocate fields with the correct dimensions
    allocate_fields()
    
    # Then initialize particle positions, etc.
    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]
    
    # Now run the evolution
    evolve_robots(num_generations=3, population_size=3)
    
if __name__ == '__main__':
    main()
        
if __name__ == '__main__':
    main()