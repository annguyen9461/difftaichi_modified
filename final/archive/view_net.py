import taichi as ti
import numpy as np
import os
import random
import math
from datetime import datetime

# Configure Taichi
real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, device_memory_fraction=0.6)

# Simulation parameters
dim = 2
n_particles = 2000
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
max_steps = 1024  # Reduced to save memory
steps = 512       # Reduced to save memory
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
        self.shapes = []  # To track shapes for visualization
    
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
        if ptype == 0:
            assert actuation == -1
        
        # Store start index to track rect particles
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
        
        # Store the shape info
        end_idx = self.n_particles - 1
        self.shapes.append({
            'type': 'rect',
            'start_idx': start_idx,
            'end_idx': end_idx,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'actuation': actuation
        })
        
        # Return the range of indices of the rect particles
        return (start_idx, end_idx)

    def add_circle(self, center_x, center_y, radius, actuation, ptype=1):
        # Store start index to track circle particles
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
        
        # Store the shape info
        end_idx = self.n_particles - 1
        self.shapes.append({
            'type': 'circle',
            'start_idx': start_idx,
            'end_idx': end_idx,
            'x': center_x,
            'y': center_y,
            'radius': radius,
            'actuation': actuation
        })
        
        # Return the range of indices of the circle particles
        return (start_idx, end_idx)
    
    def connect_shapes(self, shape1_indices, shape2_indices, num_springs=10, stiffness=600.0, damping=0.1):
        """Connect two shapes with multiple springs for stability"""
        start1, end1 = shape1_indices
        start2, end2 = shape2_indices
        
        # Find closest pairs of particles between the two shapes
        pairs = []
        for _ in range(num_springs):
            min_dist = float('inf')
            best_pair = (-1, -1)
            
            # Try random samples to find close pairs
            samples = 30  # Increase samples for better connections
            for _ in range(samples):
                i = random.randint(start1, end1)
                j = random.randint(start2, end2)
                
                p1 = np.array(self.x[i])
                p2 = np.array(self.x[j])
                dist = np.linalg.norm(p1 - p2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (i, j)
            
            if best_pair[0] != -1 and best_pair not in pairs:
                pairs.append(best_pair)
                self.add_spring(best_pair[0], best_pair[1], stiffness, damping)
        
        return pairs
    
    def connect_nearest_particles(self, spring_probability=0.7):
        """Connect particles that are close to each other with springs"""
        # Build a spatial grid for faster proximity checking
        grid_size = 0.05  # Grid cell size
        spatial_grid = {}
        
        # Add particles to spatial grid
        for i in range(self.n_particles):
            pos = self.x[i]
            grid_x = int(pos[0] / grid_size)
            grid_y = int(pos[1] / grid_size)
            
            if (grid_x, grid_y) not in spatial_grid:
                spatial_grid[(grid_x, grid_y)] = []
            
            spatial_grid[(grid_x, grid_y)].append(i)
        
        # Connect nearby particles with springs
        connected_pairs = set()
        for i in range(self.n_particles):
            pos = self.x[i]
            grid_x = int(pos[0] / grid_size)
            grid_y = int(pos[1] / grid_size)
            
            # Check neighbors in 3x3 grid cells
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_cell = (grid_x + dx, grid_y + dy)
                    
                    if neighbor_cell in spatial_grid:
                        for j in spatial_grid[neighbor_cell]:
                            if i != j and (i, j) not in connected_pairs and (j, i) not in connected_pairs:
                                # Calculate distance
                                dist = np.linalg.norm(np.array(self.x[i]) - np.array(self.x[j]))
                                
                                # Connect if close enough and with some probability
                                if dist < 0.03 and random.random() < spring_probability:
                                    self.add_spring(i, j, 600.0, 0.1)
                                    connected_pairs.add((i, j))
    
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

# Core simulation kernels (unchanged)
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
gui = ti.GUI("Cohesive Structure", (800, 800), background_color=0xFFFFFF)

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
    gui.circles(pos=particles, color=colors, radius=4.0)
    
    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')

def create_cohesive_structure(scene):
    """Create a cohesive structure using circles and rectangles"""
    # Set offset from the ground
    scene.set_offset(0.0, 0.03)
    
    # Create base rectangle (wider and closer to the ground)
    base_x = 0.1
    base_y = 0.01
    base_width = 0.25
    base_height = 0.04
    base_indices = scene.add_rect(base_x, base_y, base_width, base_height, 0)
    
    # Create a circle cluster on the right side
    c1_x = base_x + base_width * 0.8
    c1_y = base_y + base_height + 0.03
    c1_radius = 0.05
    c1_indices = scene.add_circle(c1_x, c1_y, c1_radius, 1)
    
    # Create a circle cluster on the left side
    c2_x = base_x + base_width * 0.2
    c2_y = base_y + base_height + 0.03
    c2_radius = 0.05
    c2_indices = scene.add_circle(c2_x, c2_y, c2_radius, 2)
    
    # Create a small rectangle connecting the two circles
    conn_rect_x = c2_x
    conn_rect_y = c1_y - 0.01
    conn_rect_width = c1_x - c2_x
    conn_rect_height = 0.02
    conn_indices = scene.add_rect(conn_rect_x, conn_rect_y, conn_rect_width, conn_rect_height, 3)
    
    # Create a rectangle on top to form a pyramid-like structure
    top_rect_x = base_x + base_width * 0.3
    top_rect_y = c1_y + c1_radius
    top_rect_width = base_width * 0.4
    top_rect_height = 0.03
    top_indices = scene.add_rect(top_rect_x, top_rect_y, top_rect_width, top_rect_height, 4)
    
    # Create a small rectangle on the right bottom for rolling
    right_rect_x = base_x + base_width * 0.9
    right_rect_y = base_y
    right_rect_width = 0.08
    right_rect_height = 0.02
    right_indices = scene.add_rect(right_rect_x, right_rect_y, right_rect_width, right_rect_height, 5)
    
    # Create additional circles for stability
    c3_x = base_x + base_width * 0.5
    c3_y = base_y + base_height + 0.05
    c3_radius = 0.03
    c3_indices = scene.add_circle(c3_x, c3_y, c3_radius, 6)
    
    c4_x = base_x + base_width * 0.5
    c4_y = c1_y + c1_radius + 0.03
    c4_radius = 0.03
    c4_indices = scene.add_circle(c4_x, c4_y, c4_radius, 7)
    
    # Connect all shapes with springs for stability
    scene.connect_shapes(base_indices, c1_indices, 15, 800.0, 0.1)
    scene.connect_shapes(base_indices, c2_indices, 15, 800.0, 0.1)
    scene.connect_shapes(base_indices, right_indices, 10, 800.0, 0.1)
    scene.connect_shapes(c1_indices, conn_indices, 10, 800.0, 0.1)
    scene.connect_shapes(c2_indices, conn_indices, 10, 800.0, 0.1)
    scene.connect_shapes(c1_indices, c3_indices, 8, 800.0, 0.1)
    scene.connect_shapes(c2_indices, c3_indices, 8, 800.0, 0.1)
    scene.connect_shapes(c3_indices, c4_indices, 8, 800.0, 0.1)
    scene.connect_shapes(c4_indices, top_indices, 10, 800.0, 0.1)
    
    # Connect particles that are near each other to create a denser network
    scene.connect_nearest_particles(spring_probability=0.5)
    
    # Create stronger connections between base and ground
    base_start, base_end = base_indices
    ground_y = base_y  # Same height as base bottom
    for i in range(base_start, base_end + 1):
        # If the particle is at the bottom of the base
        if abs(scene.x[i][1] - (ground_y + scene.offset_y)) < 0.01:
            # Add a strong spring to keep it connected to the ground
            # Create virtual ground anchor point
            ground_anchor_x = scene.x[i][0]
            ground_anchor_y = ground_y - 0.01 + scene.offset_y  # Slightly below
            scene.x.append([ground_anchor_x, ground_anchor_y])
            scene.actuator_id.append(-1)  # No actuation for ground anchors
            scene.particle_type.append(1)
            scene.n_particles += 1
            scene.n_solid_particles += 1
            
            # Connect with a very stiff spring
            anchor_idx = scene.n_particles - 1
            scene.add_spring(i, anchor_idx, 1200.0, 0.2)
    
    # Finalize the scene
    scene.finalize()

def initialize_weights():
    """Initialize weights with values that will create oscillating motion"""
    for i in range(n_actuators):
        # Multiple frequencies to create more complex motion
        weights[i, 0] = random.uniform(0.3, 0.5) * (1 if i % 2 == 0 else -1)
        weights[i, 1] = random.uniform(0.2, 0.4) * (1 if i % 3 == 0 else -1)
        weights[i, 2] = random.uniform(0.1, 0.3) * (1 if i % 5 == 0 else -1)
        weights[i, 3] = random.uniform(0.1, 0.2) * (1 if i % 7 == 0 else -1)
        
        # Small biases for asymmetry
        bias[i] = random.uniform(-0.1, 0.1)

def main():
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"cohesive_structure_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create scene and build the structure
        global scene
        scene = Scene()
        create_cohesive_structure(scene)
        
        print(f"Created structure with {scene.n_particles} particles and {len(scene.springs)} springs")
        print(f"Number of shapes: {len(scene.shapes)}")
        
        # Check if the particle count is within limits
        if scene.n_particles > n_particles:
            print(f"Warning: Too many particles ({scene.n_particles}), limit is {n_particles}")
            print("Please reduce the structure size or increase n_particles")
            return
        
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
        
        # Run simulation
        print("Running simulation...")
        num_frames = 200
        frame_interval = 5
        
        # Create simulation directory
        sim_dir = f"{output_dir}/simulation"
        os.makedirs(sim_dir, exist_ok=True)
        
        # Run simulation and capture frames
        for s in range(num_frames):
            advance(s)
            if s % frame_interval == 0 or s == num_frames - 1:
                print(f"Visualizing frame {s+1}/{num_frames}")
                visualize(s + 1, sim_dir)
                # Force CUDA synchronization to help with memory management
                ti.sync()
        
        print(f"Visualization complete. Output saved to {output_dir}")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up CUDA resources
        print("Cleaning up resources...")
        import gc
        gc.collect()
        try:
            ti.reset()
        except Exception as e:
            print(f"Warning during cleanup: {e}")

if __name__ == "__main__":
    main()