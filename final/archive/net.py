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
    # Calculate the center of mass at the final step
    x_avg_final = x_avg[None]
    
    # Calculate the center of mass at the initial step
    x_avg_initial = ti.Vector([0.0, 0.0])
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg_initial, contrib * x[0, i])
    
    # Calculate the horizontal distance traveled (x-direction only)
    distance_traveled_x = x_avg_final[0] - x_avg_initial[0]
    
    # Define the loss as the negative horizontal distance traveled
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

    def set_offset(self, offset_x, offset_y):
        """Set the offset for the scene"""
        self.offset_x = offset_x
        self.offset_y = offset_y
    
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

def save_params_to_csv(params, folder="config"):
    # Ensure the config folder exists
    os.makedirs(folder, exist_ok=True)
    
    # Generate a unique filename with the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"net_config_{timestamp}.csv")
    
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

def create_random_net(scene, params):
    """Create a random net-like structure that starts from the origin (with an offset)"""
    scene.set_offset(0.0, params["y_offset"])  # Small offset from the ground
    
    # Starting point near the origin
    origin_x = params["origin_x"]
    origin_y = params["origin_y"]
    
    # Create the initial node at the origin
    nodes = [(origin_x, origin_y)]
    node_indices = {}  # Map (x,y) coordinates to particle indices
    
    # Create the first particle (origin)
    scene.x.append([origin_x + scene.offset_x, origin_y + scene.offset_y])
    scene.actuator_id.append(0)  # First actuator
    scene.particle_type.append(1)  # Solid particle
    scene.n_particles += 1
    scene.n_solid_particles += 1
    scene.n_actuators = max(scene.n_actuators, 1)
    
    # Store the index of this first particle
    first_idx = scene.n_particles - 1
    node_indices[(origin_x, origin_y)] = first_idx
    
    # Add connections to form a net-like structure
    for i in range(params["num_nodes"] - 1):
        # Pick a random existing node to branch from
        if not nodes:
            break
            
        parent_idx = random.randint(0, len(nodes) - 1)
        parent_x, parent_y = nodes[parent_idx]
        parent_particle_idx = node_indices[(parent_x, parent_y)]
        
        # Determine the new node position with some randomness
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(params["min_distance"], params["max_distance"])
        
        # With high probability, bias toward the right direction to encourage rightward movement
        if random.random() < params["rightward_bias"]:
            angle = random.uniform(-math.pi/2, math.pi/2)  # Favor right side
        
        new_x = parent_x + distance * math.cos(angle)
        new_y = parent_y + distance * math.sin(angle)
        
        # Make sure the new node stays within bounds
        new_x = max(0.05, min(0.95, new_x))
        new_y = max(params["y_offset"], min(0.95, new_y))
        
        # Create the new node
        scene.x.append([new_x + scene.offset_x, new_y + scene.offset_y])
        
        # Assign an actuator ID with some probability, or -1 for non-actuated
        if random.random() < params["actuator_probability"]:
            actuator_id = random.randint(0, params["max_actuators"] - 1)
        else:
            actuator_id = -1
            
        scene.actuator_id.append(actuator_id)
        scene.particle_type.append(1)  # Solid particle
        scene.n_particles += 1
        scene.n_solid_particles += 1
        if actuator_id != -1:
            scene.n_actuators = max(scene.n_actuators, actuator_id + 1)
        
        # Store the new node's index
        new_idx = scene.n_particles - 1
        node_indices[(new_x, new_y)] = new_idx
        
        # Add to nodes list
        nodes.append((new_x, new_y))
        
        # Create a spring connection between parent and child
        scene.add_spring(parent_particle_idx, new_idx, params["stiffness"], params["damping"])
        
        # Occasionally create additional connections to form a more net-like structure
        if random.random() < params["extra_connection_probability"] and len(nodes) > 2:
            # Find another random node that's not the parent to connect to
            other_nodes = [n for n in nodes if n != (parent_x, parent_y) and n != (new_x, new_y)]
            if other_nodes:
                other_node = random.choice(other_nodes)
                other_idx = node_indices[other_node]
                
                # Only connect if they're within a reasonable distance
                dist = math.sqrt((new_x - other_node[0])**2 + (new_y - other_node[1])**2)
                if dist < params["max_connection_distance"]:
                    scene.add_spring(new_idx, other_idx, params["stiffness"], params["damping"])
    
    # Create additional random connections to make the structure more interesting
    for _ in range(params["num_extra_connections"]):
        if len(nodes) < 2:
            break
            
        # Pick two random distinct nodes
        idx1, idx2 = random.sample(range(len(nodes)), 2)
        node1 = nodes[idx1]
        node2 = nodes[idx2]
        
        # Check they're not already connected and within reasonable distance
        dist = math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
        particle_idx1 = node_indices[node1]
        particle_idx2 = node_indices[node2]
        
        # Only connect if they're within a reasonable distance
        if dist < params["max_connection_distance"]:
            scene.add_spring(particle_idx1, particle_idx2, params["stiffness"], params["damping"])
    
    # Add mid-segment particles to ensure smoother connections
    original_nodes = list(node_indices.items())
    
    # For each spring connection, add intermediate particles
    for spring in scene.springs:
        p1_idx = spring['p1']
        p2_idx = spring['p2']
        
        # Get the positions of the endpoints
        p1_pos = scene.x[p1_idx]
        p2_pos = scene.x[p2_idx]
        
        # Number of intermediate particles (denser for longer springs)
        spring_length = np.linalg.norm(np.array(p1_pos) - np.array(p2_pos))
        n_intermediate = max(1, int(spring_length / params["particle_spacing"]))
        
        # Add intermediate particles
        for i in range(1, n_intermediate):
            t = i / (n_intermediate + 1)
            x_pos = p1_pos[0] * (1 - t) + p2_pos[0] * t
            y_pos = p1_pos[1] * (1 - t) + p2_pos[1] * t
            
            # Add the intermediate particle
            scene.x.append([x_pos, y_pos])
            
            # Inherit actuator from either endpoint with some randomness
            if random.random() < 0.5:
                actuator_id = scene.actuator_id[p1_idx]
            else:
                actuator_id = scene.actuator_id[p2_idx]
                
            scene.actuator_id.append(actuator_id)
            scene.particle_type.append(1)  # Solid particle
            scene.n_particles += 1
            scene.n_solid_particles += 1
            if actuator_id != -1:
                scene.n_actuators = max(scene.n_actuators, actuator_id + 1)
            
            # Create springs to connect with previous and next particles
            if i == 1:
                scene.add_spring(p1_idx, scene.n_particles - 1, params["stiffness"], params["damping"])
            else:
                scene.add_spring(scene.n_particles - 2, scene.n_particles - 1, params["stiffness"], params["damping"])
                
            if i == n_intermediate - 1:
                scene.add_spring(scene.n_particles - 1, p2_idx, params["stiffness"], params["damping"])
    
    # Finalize the scene
    scene.finalize()

def randomize_net_params(max_nodes=100):
    """Generate random parameters for the net-like structure"""
    net_params = {
        "origin_x": 0.1,  # Start near the left side
        "origin_y": 0.01,  # Start near the bottom with an offset
        "y_offset": 0.03,  # Small offset from the ground
        "num_nodes": random.randint(10, max_nodes),
        "min_distance": 0.02,
        "max_distance": 0.1,
        "stiffness": random.uniform(400.0, 800.0),
        "damping": random.uniform(0.03, 0.1),
        "rightward_bias": random.uniform(0.6, 0.9),  # Bias toward right direction
        "actuator_probability": random.uniform(0.3, 0.7),
        "max_actuators": random.randint(3, 10),
        "extra_connection_probability": random.uniform(0.1, 0.5),
        "max_connection_distance": random.uniform(0.1, 0.2),
        "num_extra_connections": random.randint(5, 20),
        "particle_spacing": random.uniform(0.01, 0.03)
    }
    return net_params

def estimate_net_particles(params):
    """Estimate the number of particles for the net structure"""
    # Base nodes
    node_count = params["num_nodes"]
    
    # Rough estimate of connections
    connection_count = node_count - 1  # Minimum tree connections
    connection_count += params["num_extra_connections"]  # Extra random connections
    
    # Intermediate particles along connections
    avg_connection_length = (params["min_distance"] + params["max_distance"]) / 2
    avg_intermediate_particles = avg_connection_length / params["particle_spacing"]
    
    # Total particles
    total_particles = node_count + connection_count * avg_intermediate_particles
    
    return int(total_particles)

import os
from datetime import datetime
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--population_size', type=int, default=10)
    parser.add_argument('--num_generations', type=int, default=5)
    parser.add_argument('--max_particles', type=int, default=2000)
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
    
    try:
        # Initialize Taichi and allocate fields
        ti.init(default_fp=real, arch=ti.gpu, flatten_if=True, device_memory_fraction=0.7)
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
        save_params_to_csv(best_structure, folder=run_folder)
        print(f"Best structure configuration saved to {best_config_file}")
        
    except Exception as e:
        print(f"Error in main process: {e}")
        
    finally:
        # Clean up Taichi resources to prevent CUDA errors
        print("Cleaning up resources...")
        import gc
        gc.collect()
        try:
            ti.reset()
        except:
            print("Warning: Error during ti.reset(). This is expected if CUDA resources are already freed.")
        print("Done!")
      
def evolutionary_optimization(population_size, num_generations, run_folder, max_particles):
    population = [randomize_net_params(max_nodes=min(50, max_particles // 20)) for _ in range(population_size)]
    all_fitness_scores = []
    
    for generation in range(num_generations):
        print(f"Generation {generation + 1}/{num_generations}")
        fitness_scores = []
        gen_folder = os.path.join(run_folder, f"gen_{generation + 1}")
        os.makedirs(gen_folder, exist_ok=True)

        for idx, params in enumerate(population):
            print(f"Evaluating structure {idx + 1}/{population_size}")
            
            # Pre-check estimated particle count
            est_particles = estimate_net_particles(params)
            if est_particles > max_particles * 0.9:  # 90% safety margin
                print(f"Estimated particles ({est_particles}) exceeds limit. Simplifying structure.")
                # Reduce complexity
                params["num_nodes"] = max(10, params["num_nodes"] // 2)
                params["num_extra_connections"] = max(2, params["num_extra_connections"] // 2)
                params["particle_spacing"] = min(0.05, params["particle_spacing"] * 1.5)
            
            # Create scene with safety check
            try:
                scene = Scene()
                create_random_net(scene, params)
                
                # Skip if too many particles
                if scene.n_particles > max_particles:
                    print(f"Structure {idx + 1} has too many particles ({scene.n_particles}). Skipping.")
                    fitness_scores.append(-1000)  # Penalize overly complex structures
                    continue
                
                # Reset fields before each simulation
                reset_fields()

                # Initialize particle positions
                for i in range(scene.n_particles):
                    x[0, i] = scene.x[i]
                    F[0, i] = [[1, 0], [0, 1]]
                    actuator_id[i] = scene.actuator_id[i]
                    particle_type[i] = scene.particle_type[i]
                
                # Run simulation and compute fitness
                try:
                    # Force CUDA synchronization before simulation
                    ti.sync()
                    
                    # Calculate initial x position
                    x_initial = 0.0
                    initial_count = 0
                    for i in range(scene.n_particles):
                        if particle_type[i] == 1:
                            x_initial += x[0, i][0]
                            initial_count += 1
                    if initial_count > 0:
                        x_initial /= initial_count
                    
                    forward()
                    
                    # Use the final x position directly rather than using compute_loss
                    x_final = x_avg[None][0]  # Get x-coordinate of final position
                    
                    # Calculate horizontal distance
                    horizontal_distance = x_final - x_initial
                    fitness = horizontal_distance  # Reward horizontal movement
                    
                    # Add visualization for interesting structures
                    if horizontal_distance > 0.2:  # Only visualize good performers
                        struct_viz_folder = os.path.join(gen_folder, f"structure_{idx + 1}_viz")
                        os.makedirs(struct_viz_folder, exist_ok=True)
                        
                        # Run simulation again for visualization
                        reset_fields()
                        for i in range(scene.n_particles):
                            x[0, i] = scene.x[i]
                            F[0, i] = [[1, 0], [0, 1]]
                            actuator_id[i] = scene.actuator_id[i]
                            particle_type[i] = scene.particle_type[i]
                        
                        # Visualize a few key frames
                        for s in range(0, steps, steps // 10):
                            if s < steps - 1:
                                clear_grid()
                                compute_actuation(s)
                                p2g(s)
                                grid_op()
                                g2p(s)
                                if s % (steps // 10) == 0:
                                    visualize(s + 1, struct_viz_folder)
                    
                    # Force synchronization after simulation
                    ti.sync()
                    
                    # Add a small delay to let GPU recover
                    import time
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error during simulation: {e}")
                    fitness = -1000  # Penalize failed simulations
                
                fitness_scores.append(fitness)
                all_fitness_scores.append((fitness, params))

                # Save the configuration and fitness score
                config_file = os.path.join(gen_folder, f"structure_{idx + 1}.csv")
                save_params_to_csv(params, folder=os.path.dirname(config_file))
                with open(os.path.join(gen_folder, "fitness_scores.txt"), "a") as f:
                    f.write(f"Structure {idx + 1}: {fitness}\n")
                    
                # Explicit garbage collection
                import gc
                gc.collect()
                    
            except Exception as e:
                print(f"Error creating structure {idx + 1}: {e}")
                fitness_scores.append(-1000)
                continue

        # Check if we have any valid structures
        if all(score == -1000 for score in fitness_scores):
            print("No valid structures in this generation. Generating new population.")
            population = [randomize_net_params(max_nodes=min(50, max_particles // 20)) for _ in range(population_size)]
            continue

        # Rank the structures in this generation
        ranked_indices = np.argsort(fitness_scores)[::-1]  # Sort in descending order
        for rank, idx in enumerate(ranked_indices):
            if fitness_scores[idx] > -1000:  # Only report valid structures
                print(f"Rank {rank + 1}: Structure {idx + 1} with fitness {fitness_scores[idx]}")
                # Save ranking information
                with open(os.path.join(gen_folder, "rankings.txt"), "a") as f:
                    f.write(f"Rank {rank + 1}: Structure {idx + 1} with fitness {fitness_scores[idx]}\n")

        # Select top-performing structures for the next generation
        valid_structures = [(i, population[i]) for i in ranked_indices if fitness_scores[i] > -1000]
        if not valid_structures:
            print("No valid structures to select from. Generating new population.")
            population = [randomize_net_params(max_nodes=min(50, max_particles // 20)) for _ in range(population_size)]
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

        # Force CUDA synchronization and garbage collection between generations
        ti.sync()
        gc.collect()

    # Return the best structure found across all generations
    valid_scores = [(score, params) for score, params in all_fitness_scores if score > -1000]
    if not valid_scores:
        return randomize_net_params(max_nodes=min(50, max_particles // 20))  # Return a random structure if all failed
        
    # Sort by score only (not by the dictionary)
    valid_scores.sort(key=lambda x: x[0], reverse=True)
    return valid_scores[0][1]  # Return the params of the best structure

def crossover(parent1, parent2):
    """Perform crossover between two parent structures"""
    child = {}
    for key in parent1.keys():
        # For numeric parameters, we can use crossover techniques
        if isinstance(parent1[key], (int, float)):
            if random.random() < 0.5:
                # Simple selection from either parent
                child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
            else:
                # Blending (for float values)
                if isinstance(parent1[key], float):
                    # Blend with some randomness
                    blend_ratio = random.random()
                    child[key] = parent1[key] * blend_ratio + parent2[key] * (1 - blend_ratio)
                else:
                    # For integers, just take one or the other
                    child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
        else:
            # For non-numeric parameters, just select from either parent
            child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
    return child

def mutate(params):
    """Mutate the parameters of a structure"""
    mutated_params = params.copy()
    for key in mutated_params.keys():
        # Different mutation rates for different parameters
        mutation_prob = 0.2  # 20% chance of mutation
        
        if random.random() < mutation_prob:
            if key == "num_nodes":
                mutated_params[key] = max(10, mutated_params[key] + random.randint(-5, 5))
            elif key == "max_actuators":
                mutated_params[key] = max(2, min(15, mutated_params[key] + random.randint(-2, 2)))
            elif key == "num_extra_connections":
                mutated_params[key] = max(0, mutated_params[key] + random.randint(-3, 3))
            elif key == "rightward_bias":
                # Keep bias in [0.5, 0.95] range to ensure tendency toward right
                mutated_params[key] = max(0.5, min(0.95, mutated_params[key] + random.uniform(-0.1, 0.1)))
            elif key == "actuator_probability":
                mutated_params[key] = max(0.1, min(0.9, mutated_params[key] + random.uniform(-0.1, 0.1)))
            elif key == "stiffness":
                mutated_params[key] = max(200, mutated_params[key] + random.uniform(-100, 100))
            elif key == "damping":
                mutated_params[key] = max(0.01, min(0.2, mutated_params[key] + random.uniform(-0.02, 0.02)))
            elif isinstance(mutated_params[key], float):
                # Generic float mutation
                mutated_params[key] += random.uniform(-0.05, 0.05) * mutated_params[key]
            elif isinstance(mutated_params[key], int):
                # Generic int mutation
                mutated_params[key] += random.randint(-2, 2)
                
    return mutated_params

if __name__ == '__main__':
    main()