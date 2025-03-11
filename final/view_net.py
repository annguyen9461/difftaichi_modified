import taichi as ti
import numpy as np
import os
import random
import math
from datetime import datetime

# Configure Taichi
real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, device_memory_fraction=0.7)

# Simulation parameters
dim = 2
n_particles = 2000  # Start with a smaller number for safety
n_solid_particles = 0
n_actuators = 10
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

# Create Taichi fields
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

# Scene class for building structures
class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0
        self.n_actuators = 1
        self.springs = []
    
    def set_offset(self, offset_x, offset_y):
        self.offset_x = offset_x
        self.offset_y = offset_y
    
    def add_spring(self, p1, p2, stiffness, damping):
        self.springs.append({
            'p1': p1,
            'p2': p2,
            'stiffness': stiffness,
            'damping': damping,
            'rest_length': np.linalg.norm(np.array(self.x[p1]) - np.array(self.x[p2]))
        })
    
    def finalize(self):
        global n_particles, n_solid_particles, n_actuators
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        n_actuators = self.n_actuators
        print('n_particles:', n_particles)
        print('n_solid:', n_solid_particles)
        print('n_actuators:', n_actuators)

# Allocate Taichi fields
def allocate_fields():
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)
    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, x_avg)
    ti.root.lazy_grad()

# Taichi kernels (copied from your original code)
@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]

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
                grid_v_in[base + offset] += weight * (mass * v[f, p] + affine @ dpos)
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

@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    compute_actuation(s)
    p2g(s)
    grid_op()
    g2p(s)

def forward(total_steps=steps):
    for s in range(total_steps - 1):
        advance(s)
    x_avg[None] = [0, 0]
    compute_x_avg()

# GUI and Visualization
gui = ti.GUI("Random Net Structure", (800, 800), background_color=0xFFFFFF)

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
    
    # Draw floor
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)
    
    # Draw spring connections first (so particles appear on top)
    if s <= 5:  # Visualize springs in the first few frames for clarity
        scene_x = np.array(scene.x)
        springs = scene.springs
        for spring in springs:
            p1_idx = spring['p1']
            p2_idx = spring['p2']
            p1_pos = particles[p1_idx]
            p2_pos = particles[p2_idx]
            gui.line(p1_pos, p2_pos, radius=1, color=0x777777)
    
    # Draw particles with larger radius
    gui.circles(pos=particles, color=colors, radius=3.5)
    
    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')

def create_random_net(scene, params):
    """Create a random net-like structure that starts from the origin (with an offset)"""
    scene.set_offset(0.0, params["y_offset"])  # Small offset from the ground
    
    # Starting points - create a small horizontal platform at the bottom as base
    base_width = 0.15
    base_height = 0.02
    origin_x = params["origin_x"]
    origin_y = params["origin_y"]
    
    # Create a small rectangular base of particles
    nodes = []
    node_indices = {}  # Map (x,y) coordinates to particle indices
    
    # Create a small rectangle of particles at the bottom
    rect_width = 5
    rect_height = 2
    
    for i in range(rect_width):
        for j in range(rect_height):
            x_pos = origin_x + (i * base_width / rect_width)
            y_pos = origin_y + (j * base_height / rect_height)
            
            # Add the particle
            scene.x.append([x_pos + scene.offset_x, y_pos + scene.offset_y])
            scene.actuator_id.append(i % params["max_actuators"])  # Alternate actuators
            scene.particle_type.append(1)  # Solid particle
            scene.n_particles += 1
            scene.n_solid_particles += 1
            scene.n_actuators = max(scene.n_actuators, (i % params["max_actuators"]) + 1)
            
            # Store in node list and indices
            node_pos = (x_pos, y_pos)
            nodes.append(node_pos)
            node_indices[node_pos] = scene.n_particles - 1
    
    # Connect adjacent particles in the base rectangle with springs
    for i in range(rect_width):
        for j in range(rect_height):
            x_pos = origin_x + (i * base_width / rect_width)
            y_pos = origin_y + (j * base_height / rect_height)
            current_idx = node_indices[(x_pos, y_pos)]
            
            # Connect to right neighbor
            if i < rect_width - 1:
                right_x = origin_x + ((i + 1) * base_width / rect_width)
                right_y = y_pos
                right_idx = node_indices[(right_x, right_y)]
                scene.add_spring(current_idx, right_idx, params["stiffness"], params["damping"])
            
            # Connect to bottom neighbor
            if j < rect_height - 1:
                bottom_x = x_pos
                bottom_y = origin_y + ((j + 1) * base_height / rect_height)
                bottom_idx = node_indices[(bottom_x, bottom_y)]
                scene.add_spring(current_idx, bottom_idx, params["stiffness"], params["damping"])
    
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
    for spring in scene.springs.copy():  # Use copy to avoid modifying while iterating
        p1_idx = spring['p1']
        p2_idx = spring['p2']
        
        # Get the positions of the endpoints
        p1_pos = scene.x[p1_idx]
        p2_pos = scene.x[p2_idx]
        
        # Number of intermediate particles (much denser for longer springs)
        spring_length = np.linalg.norm(np.array(p1_pos) - np.array(p2_pos))
        n_intermediate = max(3, int(spring_length / params["particle_spacing"]))  # At least 3 intermediate particles
        
        # Add intermediate particles
        prev_idx = p1_idx
        for i in range(1, n_intermediate + 1):
            t = i / (n_intermediate + 1)
            x_pos = p1_pos[0] * (1 - t) + p2_pos[0] * t
            y_pos = p1_pos[1] * (1 - t) + p2_pos[1] * t
            
            # Add a small random offset to prevent perfect straight lines
            if i > 1 and i < n_intermediate:
                x_pos += random.uniform(-0.005, 0.005)
                y_pos += random.uniform(-0.005, 0.005)
            
            # Add the intermediate particle
            scene.x.append([x_pos, y_pos])
            
            # Inherit actuator from either endpoint with some randomness
            if random.random() < 0.5:
                act_id = scene.actuator_id[p1_idx]
            else:
                act_id = scene.actuator_id[p2_idx]
                
            scene.actuator_id.append(act_id)
            scene.particle_type.append(1)  # Solid particle
            scene.n_particles += 1
            scene.n_solid_particles += 1
            if act_id != -1:
                scene.n_actuators = max(scene.n_actuators, act_id + 1)
            
            new_idx = scene.n_particles - 1
            
            # Create springs to connect with previous particle
            scene.add_spring(prev_idx, new_idx, params["stiffness"], params["damping"])
            prev_idx = new_idx
        
        # Connect the last intermediate particle to the second endpoint
        scene.add_spring(prev_idx, p2_idx, params["stiffness"], params["damping"])
        
    # Add some circular clusters for added density
    for _ in range(3):  # Add 3 circular clusters
        if len(nodes) > 0:
            # Pick a random existing node as center
            center_node = random.choice(nodes)
            center_idx = node_indices[center_node]
            center_x, center_y = center_node
            
            # Create a small cluster of particles around this center
            radius = random.uniform(0.02, 0.04)
            n_cluster_particles = random.randint(8, 12)
            
            # Create particles in a circular pattern
            cluster_indices = [center_idx]
            for i in range(n_cluster_particles):
                angle = 2 * math.pi * i / n_cluster_particles
                x_pos = center_x + radius * math.cos(angle)
                y_pos = center_y + radius * math.sin(angle)
                
                # Add the particle
                scene.x.append([x_pos + scene.offset_x, y_pos + scene.offset_y])
                
                # Use the same actuator as the center node
                act_id = scene.actuator_id[center_idx]
                scene.actuator_id.append(act_id)
                scene.particle_type.append(1)
                scene.n_particles += 1
                scene.n_solid_particles += 1
                
                # Store the particle index
                cluster_indices.append(scene.n_particles - 1)
            
            # Connect all particles in the cluster to the center
            for idx in cluster_indices[1:]:
                scene.add_spring(center_idx, idx, params["stiffness"] * 1.5, params["damping"])
                
            # Connect adjacent particles in the circle
            for i in range(1, len(cluster_indices)):
                scene.add_spring(cluster_indices[i], 
                                cluster_indices[1 if i == len(cluster_indices)-1 else i+1], 
                                params["stiffness"], params["damping"])
    
    # Finalize the scene
    scene.finalize()

def randomize_net_params(max_nodes=100):
    """Generate random parameters for the net-like structure"""
    net_params = {
        "origin_x": 0.1,  # Start near the left side
        "origin_y": 0.01,  # Start near the bottom with an offset
        "y_offset": 0.03,  # Small offset from the ground
        "num_nodes": random.randint(30, max_nodes),  # More nodes for denser structure
        "min_distance": 0.015,  # Smaller minimum distance
        "max_distance": 0.06,   # Smaller maximum distance for denser packing
        "stiffness": random.uniform(400.0, 800.0),
        "damping": random.uniform(0.03, 0.1),
        "rightward_bias": random.uniform(0.6, 0.9),  # Bias toward right direction
        "actuator_probability": random.uniform(0.3, 0.7),
        "max_actuators": random.randint(5, 15),  # More actuators
        "extra_connection_probability": random.uniform(0.5, 0.8),  # Higher probability for extra connections
        "max_connection_distance": random.uniform(0.1, 0.2),
        "num_extra_connections": random.randint(20, 50),  # More extra connections
        "particle_spacing": random.uniform(0.005, 0.01)  # Smaller spacing for denser packing
    }
    return net_params

def initialize_weights():
    """Initialize weights with values that will create oscillating motion"""
    for i in range(n_actuators):
        for j in range(n_sin_waves):
            # Random weights for oscillation
            weights[i, j] = random.uniform(-0.5, 0.5)
        # Small bias
        bias[i] = random.uniform(-0.1, 0.1)

# Main visualization function
def main():
    # Generate net structure parameters
    params = randomize_net_params(max_nodes=60)  # Increased to 60 nodes for denser structure
    
    print("Net Structure Parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"dense_net_viz_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save parameters
    with open(f"{output_dir}/params.txt", "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    
    # Build the scene
    global scene
    scene = Scene()
    create_random_net(scene, params)
    
    # Check if we have too many particles
    if scene.n_particles > n_particles:
        print(f"Warning: Generated structure has {scene.n_particles} particles, but limit is {n_particles}")
        print("Rebuilding with fewer nodes...")
        scene = Scene()
        params["num_nodes"] = min(40, params["num_nodes"] // 2)
        params["num_extra_connections"] = min(20, params["num_extra_connections"] // 2)
        create_random_net(scene, params)
        
    print(f"Created structure with {scene.n_particles} particles and {len(scene.springs)} springs")
    
    # Allocate fields
    allocate_fields()
    
    # Initialize particle positions and properties
    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]
    
    # Initialize weights for actuation
    initialize_weights()
    
    # Create a subfolder for the initial frame
    initial_dir = f"{output_dir}/initial"
    os.makedirs(initial_dir, exist_ok=True)
    visualize(0, initial_dir)
    
    # Run interactive mode for a few frames to let the user see the initial structure
    for s in range(5):
        visualize(s, initial_dir)
        print(f"Showing initial structure, frame {s+1}/5...")
    
    # Run simulation
    print("Running simulation...")
    num_frames = 300  # More frames for a longer simulation
    frame_interval = 4  # Save every 4th frame
    
    # Create simulation directory
    sim_dir = f"{output_dir}/simulation"
    os.makedirs(sim_dir, exist_ok=True)
    
    # Run simulation and capture frames
    for s in range(num_frames):
        advance(s)
        if s % frame_interval == 0 or s == num_frames - 1:
            print(f"Visualizing frame {s+1}/{num_frames}")
            visualize(s + 1, sim_dir)
    
    print(f"Visualization complete. Output saved to {output_dir}")

if __name__ == "__main__":
    main()