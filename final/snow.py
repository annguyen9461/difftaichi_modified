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



def allocate_fields():
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


# @ti.kernel
# def compute_loss():
#     dist = x_avg[None][0]
#     loss[None] = -dist

@ti.kernel
def compute_loss():
    # Calculate the center of mass at the final step
    x_avg_final = x_avg[None]
    
    # Calculate the center of mass at the initial step
    x_avg_initial = ti.Vector([0.0, 0.0])
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg_initial, contrib * x[0, i])
    
    # Calculate the distance traveled (Euclidean distance)
    distance_traveled = (x_avg_final - x_avg_initial).norm()
    
    # Define the loss as the negative distance traveled (since we want to maximize distance)
    loss[None] = -distance_traveled

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


def randomize_snowflake_params0():
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

def randomize_snowflake_params():
    snowflake_params = {
        "start_x": random.uniform(0.1, 0.3),
        "start_y": random.uniform(0.4, 0.6),
        "depth": random.randint(1, 3),  # Reduced max depth
        "branch_length": random.uniform(0.05, 0.15),
        "angle": random.uniform(0, 2 * math.pi),
        "thickness": random.uniform(0.005, 0.01),
        "stiffness": random.uniform(400.0, 600.0),
        "damping": random.uniform(0.03, 0.07),
        "num_sub_branches": random.randint(2, 5),  # Reduced max branches
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

def crossover(parent1, parent2):
    # Implement crossover logic (e.g., mix parameters from both parents)
    child = {}
    for key in parent1.keys():
        if random.random() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child

def mutate(params):
    # Implement mutation logic (e.g., randomly change some parameters)
    for key in params.keys():
        if random.random() < 0.1:  # 10% mutation rate
            if isinstance(params[key], int):
                params[key] += random.randint(-1, 1)
            elif isinstance(params[key], float):
                params[key] += random.uniform(-0.1, 0.1)
    return params

def reset_fields():
    """Reset Taichi fields to their initial state."""
    global x, v, C, F, grid_v_in, grid_m_in, grid_v_out, actuation, weights, bias, loss, x_avg
    x.fill(0)
    v.fill(0)
    C.fill([[0, 0], [0, 0]])
    F.fill([[1, 0], [0, 1]])
    grid_v_in.fill([0, 0])
    grid_m_in.fill(0)
    grid_v_out.fill([0, 0])
    actuation.fill(0)
    weights.fill(0)
    bias.fill(0)
    loss[None] = 0
    x_avg[None] = [0, 0]

import os
from datetime import datetime
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--population_size', type=int, default=10)
    parser.add_argument('--num_generations', type=int, default=5)
    parser.add_argument('--max_particles', type=int, default=5000)  # Add max particles parameter
    options = parser.parse_args()

    # Create a folder for this run with the current timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"run_{current_time}"
    os.makedirs(run_folder, exist_ok=True)

    # Initialize with reasonable limits
    global n_particles, n_solid_particles, n_actuators
    n_particles = options.max_particles
    n_solid_particles = options.max_particles
    n_actuators = 10  # Set a safe upper limit
    
    # Allocate fields just once
    allocate_fields()

    # Run evolutionary optimization
    best_structure = evolutionary_optimization(
        population_size=options.population_size,
        num_generations=options.num_generations,
        run_folder=run_folder,
        max_particles=options.max_particles
    )

    print("Best structure found:")
    for key, value in best_structure.items():
        print(f"{key}: {value}")

    # Save the best structure's configuration
    best_config_file = os.path.join(run_folder, "best_structure.csv")
    save_params_to_csv(best_structure, best_config_file)

    print(f"Best structure configuration saved to {best_config_file}")

def display_top_structure(params, gen_folder, max_particles):
    """Simulate and visualize the top structure in a generation with a pop-up window.
    Uses the exact same approach as the reference code's main visualization function."""
    
    # Create scene and robot
    scene = Scene()
    create_complex_robot(scene, params)
    
    # Skip if too many particles
    if scene.n_particles > max_particles:
        print(f"Top structure has too many particles ({scene.n_particles}). Skipping visualization.")
        return
    
    # Reset fields before simulation
    reset_fields()

    # Initialize particle positions
    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]

    # Create visualization directory
    vis_folder = os.path.join(gen_folder, "visualization")
    os.makedirs(vis_folder, exist_ok=True)

    # Set steps for simulation - shorter for quick feedback
    steps_to_run = 200
    
    print("\nRunning simulation for visualization...")
    # Run the simulation to generate all states
    for s in range(steps_to_run - 1):
        advance(s)
    
    print("Starting visualization... (Close window to continue)")
    
    # Display each frame, exactly like the reference code
    for s in range(5, steps_to_run, 5):  # Skip some frames for speed
        aid = actuator_id.to_numpy()
        
        # Get the particles for this step
        particles = x.to_numpy()[s]
        n_visible = min(n_particles, len(particles))  # Get actual number of particles to display
        
        # Create colors array with correct shape
        colors = np.empty(n_visible, dtype=np.uint32)
        
        # Get actuation values
        actuation_ = actuation.to_numpy()
        
        # Set colors for each particle
        for i in range(n_visible):
            color = 0x111111
            if i < len(aid) and aid[i] != -1:
                act_id = int(aid[i])
                if act_id >= 0 and s > 0 and act_id < len(actuation_[s-1]):
                    act = actuation_[s-1, act_id]
                    color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
            colors[i] = color
        
        # Draw exactly as in the reference but with n_visible particles
        gui.circles(pos=particles[:n_visible], color=colors, radius=1.5)
        gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)
        
        # Save and show the frame
        filename = f'{vis_folder}/{s:04d}.png'
        gui.show(filename)
        
        # Check if window was closed
        if not gui.running:
            break
            
        # Slow down the visualization slightly
        import time
        time.sleep(0.05)
    
    print(f"Visualization complete. Frames saved to {vis_folder}")

def evolutionary_optimization(population_size, num_generations, run_folder, max_particles):
    population = [randomize_snowflake_params() for _ in range(population_size)]
    all_fitness_scores = []
    
    for generation in range(num_generations):
        print(f"\n=== Generation {generation + 1}/{num_generations} ===")
        fitness_scores = []
        gen_folder = os.path.join(run_folder, f"gen_{generation + 1}")
        os.makedirs(gen_folder, exist_ok=True)

        for idx, params in enumerate(population):
            print(f"Evaluating structure {idx + 1}/{population_size}")
            
            # Create scene with safety check
            try:
                scene = Scene()
                create_complex_robot(scene, params)
                
                # Skip if too many particles
                if scene.n_particles > max_particles:
                    print(f"Structure {idx + 1} has too many particles ({scene.n_particles}). Skipping.")
                    fitness_scores.append(-1000)  # Penalize overly complex structures
                    continue
                
                # Reset fields before each simulation
                reset_fields()

                # Initialize particle positions, deformation gradient, and velocity
                for i in range(scene.n_particles):
                    x[0, i] = scene.x[i]
                    F[0, i] = [[1, 0], [0, 1]]
                    actuator_id[i] = scene.actuator_id[i]
                    particle_type[i] = scene.particle_type[i]
                
                # Run simulation and compute fitness
                try:
                    forward()
                    compute_loss()
                    fitness = -loss[None]
                except Exception as e:
                    print(f"Error during simulation: {e}")
                    fitness = -1000  # Penalize failed simulations
                
                fitness_scores.append(fitness)
                all_fitness_scores.append((fitness, params))  # Store as tuple

                # Save the configuration and fitness score
                config_file = os.path.join(gen_folder, f"structure_{idx + 1}.csv")
                save_params_to_csv(params, config_file)
                with open(os.path.join(gen_folder, "fitness_scores.txt"), "a") as f:
                    f.write(f"Structure {idx + 1}: {fitness}\n")
                    
            except Exception as e:
                print(f"Error creating structure {idx + 1}: {e}")
                fitness_scores.append(-1000)
                continue

        # Check if we have any valid structures
        if all(score == -1000 for score in fitness_scores):
            print("No valid structures in this generation. Generating new population.")
            population = [randomize_snowflake_params() for _ in range(population_size)]
            continue

        # Rank the structures in this generation
        ranked_indices = np.argsort(fitness_scores)[::-1]  # Sort in descending order
        for rank, idx in enumerate(ranked_indices):
            if fitness_scores[idx] > -1000:  # Only report valid structures
                print(f"Rank {rank + 1}: Structure {idx + 1} with fitness {fitness_scores[idx]}")
                # Save ranking information
                with open(os.path.join(gen_folder, "rankings.txt"), "a") as f:
                    f.write(f"Rank {rank + 1}: Structure {idx + 1} with fitness {fitness_scores[idx]}\n")

        # Visualize the top structure in this generation if valid
        top_structure_idx = ranked_indices[0]
        if fitness_scores[top_structure_idx] > -1000:
            top_params = population[top_structure_idx]
            print(f"\nVisualizing top structure in generation {generation + 1}")
            print(f"Close the visualization window to continue to the next generation.")
            try:
                # Use the simpler visualization approach
                display_top_structure(top_params, gen_folder, max_particles)
            except Exception as e:
                import traceback
                print(f"Error visualizing top structure: {traceback.format_exc()}")

        # Select top-performing structures for the next generation
        valid_structures = [(i, population[i]) for i in ranked_indices if fitness_scores[i] > -1000]
        if not valid_structures:
            print("No valid structures to select from. Generating new population.")
            population = [randomize_snowflake_params() for _ in range(population_size)]
            continue
            
        top_structures = [params for _, params in valid_structures[:max(1, population_size // 2)]]

        # Create new population through mutation and crossover
        new_population = top_structures.copy()
        while len(new_population) < population_size:
            if len(top_structures) >= 2:
                parent1, parent2 = random.sample(top_structures, 2)
                child = crossover(parent1, parent2)
            else:
                child = top_structures[0].copy()
            child = mutate(child)
            new_population.append(child)

        population = new_population

    # Return the best structure found across all generations
    # Fix the sorting problem by ensuring we're sorting tuples by their first element (fitness)
    valid_scores = [(score, params) for score, params in all_fitness_scores if score > -1000]
    if not valid_scores:
        return randomize_snowflake_params()  # Return a random structure if all failed
        
    valid_scores.sort(key=lambda x: x[0], reverse=True)  # Sort by fitness (first element of tuple)
    return valid_scores[0][1]  # Return the params of the best structure

def visualize_top_structure(params, gen_folder, max_particles):
    """Simulate and visualize the top structure in a generation."""
    scene = Scene()
    create_complex_robot(scene, params)
    
    # Skip if too many particles
    if scene.n_particles > max_particles:
        print(f"Top structure has too many particles ({scene.n_particles}). Skipping visualization.")
        return
    
    # Reset fields before simulation
    reset_fields()

    # Initialize particle positions
    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]

    # Run simulation and show visualization interactively
    steps_to_run = min(500, steps)  # Use fewer steps for visualization
    vis_folder = os.path.join(gen_folder, "visualization")
    os.makedirs(vis_folder, exist_ok=True)
    
    for s in range(steps_to_run - 1):
        advance(s)
        if s % 10 == 0:  # Show every 10 steps to make it more responsive
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
            
            # Save the frame
            filename = f'{vis_folder}/{s:04d}.png'
            gui.show(filename)
            
            # Add a small pause to make the visualization visible
            # This is a blocking operation so the simulation will pause briefly
            import time
            time.sleep(0.05)
    
    # Save a message about the visualization location
    print(f"Visualization frames saved to {vis_folder}")
    print("You can view these frames as a sequence or convert them to a video.")

if __name__ == '__main__':
    main()