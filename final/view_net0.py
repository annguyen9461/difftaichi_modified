import taichi as ti
import numpy as np
import os
import random
import math
from datetime import datetime

# Configure Taichi with reduced memory usage
real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, device_memory_fraction=0.5)

# Simulation parameters
dim = 2
n_particles = 1000  # Reduced particle count
n_solid_particles = 0
n_actuators = 5
n_grid = 64
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 10
mu = E
la = E
max_steps = 512
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
        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        
        # Store start index to track rect particles
        start_idx = self.n_particles
        
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
        
        # Return the range of indices of the rect particles
        return (start_idx, self.n_particles - 1)

    def add_circle(self, center_x, center_y, radius, actuation, ptype=1):
        global n_particles
        spacing = dx / 2  # Adjust spacing for denser particles
        
        # Store start index to track circle particles
        start_idx = self.n_particles
        
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
        
        # Return the range of indices of the circle particles
        return (start_idx, self.n_particles - 1)
        
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

# Taichi kernels (core simulation functions)
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
gui = ti.GUI("MPM Net Structure", (800, 800), background_color=0xFFFFFF)

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
    
    # Draw springs if in early frames
    if s <= 5:
        for spring in scene.springs:
            p1_idx = spring['p1']
            p2_idx = spring['p2']
            if p1_idx < len(particles) and p2_idx < len(particles):
                p1_pos = particles[p1_idx]
                p2_pos = particles[p2_idx]
                gui.line(p1_pos, p2_pos, radius=1, color=0x777777)
    
    # Draw particles with large radius
    gui.circles(pos=particles, color=colors, radius=6.0)
    
    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')

def create_simple_structure(scene):
    """Create a simple structure using rectangles and circles"""
    # Add a small offset from the ground
    scene.set_offset(0.0, 0.03)
    
    # Create base rectangle near the origin (0,0)
    base_x = 0.1
    base_y = 0.01
    base_width = 0.25
    base_height = 0.06
    base_indices = scene.add_rect(base_x, base_y, base_width, base_height, 0)
    
    # Create a small circle on the right side of the base
    circle1_x = base_x + base_width * 0.8
    circle1_y = base_y + base_height * 1.5
    circle1_radius = 0.05
    circle1_indices = scene.add_circle(circle1_x, circle1_y, circle1_radius, 1)
    
    # Create another small circle on the left side of the base
    circle2_x = base_x + base_width * 0.2
    circle2_y = base_y + base_height * 1.5
    circle2_radius = 0.05
    circle2_indices = scene.add_circle(circle2_x, circle2_y, circle2_radius, 2)
    
    # Create a rectangle on top to connect the circles
    top_rect_x = base_x + base_width * 0.15
    top_rect_y = base_y + base_height * 2.0
    top_rect_width = base_width * 0.7
    top_rect_height = 0.03
    top_rect_indices = scene.add_rect(top_rect_x, top_rect_y, top_rect_width, top_rect_height, 3)
    
    # Create an asymmetric element to help with rolling/locomotion
    side_rect_x = base_x + base_width * 0.9
    side_rect_y = base_y + base_height * 0.5
    side_rect_width = 0.08
    side_rect_height = 0.03
    side_rect_indices = scene.add_rect(side_rect_x, side_rect_y, side_rect_width, side_rect_height, 4)
    
    # Add additional springs to connect all parts together
    # Connect base to circle1
    connect_regions_with_springs(scene, base_indices, circle1_indices, 600, 0.05)
    
    # Connect base to circle2
    connect_regions_with_springs(scene, base_indices, circle2_indices, 600, 0.05)
    
    # Connect circle1 to top rect
    connect_regions_with_springs(scene, circle1_indices, top_rect_indices, 600, 0.05)
    
    # Connect circle2 to top rect
    connect_regions_with_springs(scene, circle2_indices, top_rect_indices, 600, 0.05)
    
    # Connect side rect to base
    connect_regions_with_springs(scene, side_rect_indices, base_indices, 600, 0.05)
    
    # Finalize the scene
    scene.finalize()

def connect_regions_with_springs(scene, region1, region2, stiffness, damping, num_connections=5):
    """Connect two regions (defined by index ranges) with springs"""
    start1, end1 = region1
    start2, end2 = region2
    
    # Find closest particles between regions and connect them
    for _ in range(min(num_connections, (end1-start1+1)*(end2-start2+1))):
        min_dist = float('inf')
        best_pair = (-1, -1)
        
        # Sample random particles from each region to find close pairs
        for _ in range(10):  # Try 10 random samples
            i = random.randint(start1, end1)
            j = random.randint(start2, end2)
            
            p1 = scene.x[i]
            p2 = scene.x[j]
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            
            if dist < min_dist:
                min_dist = dist
                best_pair = (i, j)
        
        if best_pair[0] != -1:
            scene.add_spring(best_pair[0], best_pair[1], stiffness, damping)

def initialize_weights():
    """Initialize weights with values that will create oscillating motion"""
    for i in range(n_actuators):
        for j in range(n_sin_waves):
            weights[i, j] = random.uniform(-0.5, 0.5)
        bias[i] = random.uniform(-0.1, 0.1)

def main():
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"simple_structure_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create scene and build the structure
        global scene
        scene = Scene()
        create_simple_structure(scene)
        
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
        
        # Run simulation
        print("Running simulation...")
        num_frames = 100
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
                ti.sync()  # Synchronize CUDA operations
        
        print(f"Visualization complete. Output saved to {output_dir}")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        # Explicitly clean up CUDA resources
        print("Cleaning up resources...")
        import gc
        gc.collect()
        try:
            ti.reset()
        except Exception as e:
            print(f"Warning during cleanup: {e}")

if __name__ == "__main__":
    main()