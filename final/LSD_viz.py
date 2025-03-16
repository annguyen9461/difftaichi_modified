import taichi as ti
import numpy as np
import os
import csv
import random
import math
from datetime import datetime
import argparse

# Configure Taichi
real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, device_memory_fraction=0.6)

# Simulation parameters
dim = 2
n_particles = 2000  # Start with a reasonable number
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
max_steps = 512  # Reduced to save memory
steps = 256
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
    
    def add_rect(self, x, y, w, h, actuation, ptype=1):
        """Add a rectangle of particles"""
        # Store start index for tracking
        start_idx = self.n_particles
        
        # Calculate grid based on dx
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
        
        # Return the indices range
        return (start_idx, self.n_particles - 1)

    def add_circle(self, center_x, center_y, radius, actuation, ptype=1):
        """Add a circle of particles"""
        # Store start index for tracking
        start_idx = self.n_particles
        
        # Use smaller spacing for denser circles
        spacing = dx / 2
        
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
        
        # Return the indices range
        return (start_idx, self.n_particles - 1)
    
    def finalize(self):
        """Finalize the scene and update global parameters"""
        global n_particles, n_solid_particles, n_actuators
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        n_actuators = self.n_actuators
        print('n_particles:', n_particles)
        print('n_solid:', n_solid_particles)
        print('n_actuators:', n_actuators)

# Allocate Taichi fields
def allocate_fields():
    """Allocate Taichi fields based on global parameters"""
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)
    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, x_avg)
    ti.root.lazy_grad()

# Core simulation kernels (essential functions only)
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

# GUI and Visualization
gui = ti.GUI("Structure Visualizer", (800, 800), background_color=0xFFFFFF)

def visualize(s, folder=None, save=True):
    """Visualize the structure at simulation step s"""
    aid = actuator_id.to_numpy()
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    particles = x.to_numpy()[s]
    
    # Use frame 0 for initial visualization
    actuation_frame = max(0, s-1)
    actuation_ = actuation.to_numpy()
    
    for i in range(n_particles):
        color = 0x111111
        if aid[i] != -1 and actuation_frame < actuation_.shape[0]:
            act = actuation_[actuation_frame, int(aid[i])]
            color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
    
    # Draw floor
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)
    
    # Draw spring connections first for clarity
    if s <= 5:
        for spring in scene.springs:
            p1_idx = spring['p1']
            p2_idx = spring['p2']
            if p1_idx < len(particles) and p2_idx < len(particles):
                p1_pos = particles[p1_idx]
                p2_pos = particles[p2_idx]
                gui.line(p1_pos, p2_pos, radius=1, color=0x777777)
    
    # Draw particles with larger radius
    gui.circles(pos=particles, color=colors, radius=5.0)
    
    if save and folder:
        os.makedirs(folder, exist_ok=True)
        gui.show(f'{folder}/{s:04d}.png')
    else:
        return gui.show()

def load_config_from_csv(filename):
    """Load configuration parameters from a CSV file"""
    params = {}
    try:
        with open(filename, mode="r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                if len(row) == 2:
                    key, value = row
                    # Convert value to the appropriate type
                    if key in ["num_nodes", "max_actuators", "num_extra_connections"]:
                        params[key] = int(value)
                    elif key in ["origin_x", "origin_y", "y_offset", "min_distance", "max_distance", 
                                "stiffness", "damping", "rightward_bias", "actuator_probability", 
                                "max_connection_distance", "particle_spacing"]:
                        params[key] = float(value)
                    else:
                        params[key] = value
        return params
    except Exception as e:
        print(f"Error loading config from {filename}: {e}")
        return None

def create_random_net(scene, params):
    """Create a random net-like structure starting from origin"""
    # Set offset
    scene.set_offset(0.0, params["y_offset"])
    
    # Generate initial points based on config
    if "use_rect_base" in params and params["use_rect_base"]:
        # Create a solid base rectangle at the bottom
        base_width = 0.2
        base_height = 0.04
        base_indices = scene.add_rect(params["origin_x"], params["origin_y"], 
                                     base_width, base_height, 0)
        
        # Create three circles above the base
        circle_spacing = base_width / 4
        for i in range(3):
            circle_x = params["origin_x"] + (i + 1) * circle_spacing
            circle_y = params["origin_y"] + base_height + 0.03
            circle_radius = 0.03
            circle_indices = scene.add_circle(circle_x, circle_y, circle_radius, i+1)
            
            # Connect circle to base with springs
            for _ in range(5):  # Add multiple springs for stability
                c_idx = random.randint(circle_indices[0], circle_indices[1])
                b_idx = random.randint(base_indices[0], base_indices[1])
                scene.add_spring(c_idx, b_idx, params["stiffness"], params["damping"])
    else:
        # Create net structure starting from one point
        origin_x = params["origin_x"]
        origin_y = params["origin_y"]
        
        # Create nodes and track their indices
        nodes = [(origin_x, origin_y)]
        node_indices = {}
        
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
            
            # With high probability, bias toward the right direction 
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
            
            # Add additional connections to make the structure more net-like
            if random.random() < params["extra_connection_probability"] and len(nodes) > 2:
                # Find another random node that's not the parent
                other_nodes = [n for n in nodes if n != (parent_x, parent_y) and n != (new_x, new_y)]
                if other_nodes:
                    other_node = random.choice(other_nodes)
                    other_idx = node_indices[other_node]
                    
                    # Connect if within reasonable distance
                    dist = math.sqrt((new_x - other_node[0])**2 + (new_y - other_node[1])**2)
                    if dist < params["max_connection_distance"]:
                        scene.add_spring(new_idx, other_idx, params["stiffness"], params["damping"])
        
        # Add extra random connections for a denser net
        for _ in range(params["num_extra_connections"]):
            if len(nodes) < 2:
                break
                
            # Pick two random distinct nodes
            idx1, idx2 = random.sample(range(len(nodes)), 2)
            node1 = nodes[idx1]
            node2 = nodes[idx2]
            
            # Check distance
            dist = math.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
            particle_idx1 = node_indices[node1]
            particle_idx2 = node_indices[node2]
            
            # Connect if within reasonable distance
            if dist < params["max_connection_distance"]:
                scene.add_spring(particle_idx1, particle_idx2, params["stiffness"], params["damping"])
        
        # Add intermediate particles for smoother connections
        for spring in scene.springs.copy():  # Use copy to avoid changing during iteration
            p1_idx = spring['p1']
            p2_idx = spring['p2']
            
            # Get the positions of the endpoints
            p1_pos = scene.x[p1_idx]
            p2_pos = scene.x[p2_idx]
            
            # Number of intermediate particles based on distance
            spring_length = np.linalg.norm(np.array(p1_pos) - np.array(p2_pos))
            n_intermediate = max(1, int(spring_length / params["particle_spacing"]))
            
            # Skip if too long to save particles
            if n_intermediate > 10:
                n_intermediate = 3
            
            # Add intermediate particles
            prev_idx = p1_idx
            for i in range(1, n_intermediate):
                t = i / (n_intermediate + 1)
                x_pos = p1_pos[0] * (1 - t) + p2_pos[0] * t
                y_pos = p1_pos[1] * (1 - t) + p2_pos[1] * t
                
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
                
                # Create springs to connect
                scene.add_spring(prev_idx, new_idx, params["stiffness"], params["damping"])
                prev_idx = new_idx
            
            # Connect last intermediate to the endpoint
            if n_intermediate > 1:
                scene.add_spring(prev_idx, p2_idx, params["stiffness"], params["damping"])
    
    # Finalize the scene
    scene.finalize()

def create_hybrid_structure(scene):
    """Create a simpler structure using both circles and rectangles"""
    # Set offset from the ground
    scene.set_offset(0.0, 0.03)
    
    # Create base rectangle
    base_x = 0.1
    base_y = 0.01
    base_width = 0.2
    base_height = 0.04
    base = scene.add_rect(base_x, base_y, base_width, base_height, 0)
    
    # Create circles on top
    circle1 = scene.add_circle(base_x + base_width*0.25, base_y + base_height + 0.03, 0.04, 1)
    circle2 = scene.add_circle(base_x + base_width*0.75, base_y + base_height + 0.03, 0.04, 2)
    
    # Add connecting rectangle
    connector = scene.add_rect(base_x + base_width*0.2, base_y + base_height + 0.06, 
                              base_width*0.6, 0.02, 3)
    
    # Add springs to connect everything
    # Connect base to circles
    for _ in range(10):
        base_idx = random.randint(base[0], base[1])
        c1_idx = random.randint(circle1[0], circle1[1])
        scene.add_spring(base_idx, c1_idx, 600.0, 0.1)
        
        base_idx = random.randint(base[0], base[1])
        c2_idx = random.randint(circle2[0], circle2[1])
        scene.add_spring(base_idx, c2_idx, 600.0, 0.1)
    
    # Connect circles to connector
    for _ in range(10):
        c1_idx = random.randint(circle1[0], circle1[1])
        conn_idx = random.randint(connector[0], connector[1])
        scene.add_spring(c1_idx, conn_idx, 600.0, 0.1)
        
        c2_idx = random.randint(circle2[0], circle2[1])
        conn_idx = random.randint(connector[0], connector[1])
        scene.add_spring(c2_idx, conn_idx, 600.0, 0.1)
    
    # Add asymmetric element to encourage movement
    side_rect = scene.add_rect(base_x + base_width*0.9, base_y, base_width*0.2, base_height*0.5, 4)
    
    # Connect side rect to base
    for _ in range(5):
        side_idx = random.randint(side_rect[0], side_rect[1])
        base_idx = random.randint(base[0], base[1])
        scene.add_spring(side_idx, base_idx, 600.0, 0.1)
    
    # Finalize
    scene.finalize()

def initialize_weights():
    """Initialize weights with values for interesting motion"""
    for i in range(n_actuators):
        for j in range(n_sin_waves):
            # Create varying frequencies for complex motion
            weights[i, j] = random.uniform(-0.5, 0.5)
        # Small biases
        bias[i] = random.uniform(-0.1, 0.1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='Path to config CSV file')
    parser.add_argument('--mode', type=str, default='hybrid', 
                      choices=['hybrid', 'net', 'custom'], 
                      help='Structure type: hybrid (circles+rects), net, or custom (from CSV)')
    parser.add_argument('--output', type=str, default='', help='Output folder for visualization')
    parser.add_argument('--frames', type=int, default=2000, help='Number of simulation frames')
    options = parser.parse_args()
    
    # Create output folder if needed
    if not options.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        options.output = f"viz_{options.mode}_{timestamp}"
    
    os.makedirs(options.output, exist_ok=True)
    
    try:
        # Create scene
        global scene
        scene = Scene()
        
        # Load or create structure based on mode
        if options.mode == 'custom' and options.config:
            # Load parameters from CSV
            params = load_config_from_csv(options.config)
            if params:
                print("Loaded parameters:")
                for key, value in params.items():
                    print(f"  {key}: {value}")
                # Create net structure from parameters
                create_random_net(scene, params)
            else:
                print("Failed to load parameters, using hybrid structure instead")
                create_hybrid_structure(scene)
        elif options.mode == 'net':
            # Create a random net structure
            params = {
                "origin_x": 0.1,
                "origin_y": 0.01,
                "y_offset": 0.03,
                "num_nodes": 50,
                "min_distance": 0.02,
                "max_distance": 0.08,
                "stiffness": 400.0,
                "damping": 0.2,
                "rightward_bias": 0.7,
                "actuator_probability": 0.4,
                "max_actuators": 10,
                "max_connection_distance": 0.15,
                "extra_connection_probability": 0.3,
                "num_extra_connections": 20,
                "particle_spacing": 0.02
            }
            create_random_net(scene, params)
        else:
            # Default to hybrid structure
            create_hybrid_structure(scene)
        
        # Allocate fields for simulation
        allocate_fields()
        
        # Initialize initial positions, velocities and deformation gradients
        for i in range(n_particles):
            x[0, i] = scene.x[i]
            v[0, i] = [0, 0]
            F[0, i] = [[1, 0], [0, 1]]
        
        # Initialize weights for actuators
        initialize_weights()
        
        # Run simulation
        print(f"Running simulation with {options.frames} frames")
        
        # Forward simulation
        for s in range(min(options.frames, max_steps-1)):
            advance(s)
            # Visualize every few steps to speed up processing
            if s % 5 == 0 or s == options.frames - 1:
                visualize(s, options.output)
                print(f"Frame {s}/{options.frames}")
        
        print(f"Simulation complete. Results saved to {options.output}")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()