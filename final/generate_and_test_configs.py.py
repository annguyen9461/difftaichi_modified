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

def evaluate_fitness(params):
    reset_fields()  # Clear all fields
    scene = Scene()
    create_snowflake_structure(scene, params)
    
    # Initialize particles
    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]
    
    # Run simulation
    try:
        forward()
    except Exception as e:
        print(f"Simulation error: {e}")
        return -1000  # Return a low fitness score for invalid structures
    
    # Calculate fitness
    return x_avg[None][0]  # Example fitness function

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
    
    def add_organic_structure(self, start_x, start_y, depth, branch_length, angle, actuation_start, ptype, params):
        """Create a more organic, asymmetric branching structure"""
        if depth <= 0:
            return

        # Add randomness to the angle
        angle_jitter = params.get("angle_jitter", math.pi/8)
        angle += random.uniform(-angle_jitter, angle_jitter)

        # Add randomness to branch length
        length_jitter = params.get("length_jitter", 0.2)
        branch_length *= (1.0 + random.uniform(-length_jitter, length_jitter))

        # Calculate end point
        end_x = start_x + branch_length * math.cos(angle)
        end_y = start_y + branch_length * math.sin(angle)

        # Add particles along branch with reduced density
        n_points = max(3, int(branch_length / (dx * (depth + 1))))

        prev_particle_idx = None
        for i in range(n_points):
            t = i / (n_points - 1)
            x_pos = start_x + t * (end_x - start_x)
            y_pos = start_y + t * (end_y - start_y)

            # Random thickness variation
            thickness_variation = params.get("thickness_variation", 0.2)
            thickness = params["thickness"] * (1.0 + random.uniform(-thickness_variation, thickness_variation))

            # For non-terminal branches, use full thickness
            if depth > 1:
                thickness_range = range(-1, 2)  # 3 particles thick
            else:
                thickness_range = range(0, 1)  # Terminal branches are single line

            for j in thickness_range:
                offset_x = -j * thickness * math.sin(angle)
                offset_y = j * thickness * math.cos(angle)

                # Add particle
                self.x.append([x_pos + offset_x + self.offset_x, y_pos + offset_y + self.offset_y])
                self.actuator_id.append(actuation_start)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)

                if actuation_start != -1:
                    self.n_actuators = max(self.n_actuators, actuation_start + 1)

                # Add springs
                if prev_particle_idx is not None and j == 0:
                    spring_stiffness = params["stiffness"] * (0.7 + 0.3 * depth/params["depth"])
                    self.add_spring(prev_particle_idx, self.n_particles - 1, spring_stiffness, params["damping"])

                if j != thickness_range.start and j < len(thickness_range) - 1:
                    self.add_spring(self.n_particles - 1, self.n_particles - 2, params["stiffness"], params["damping"])

            prev_particle_idx = self.n_particles - 1

        # Determine how many sub-branches to create
        sub_branch_count_variation = params.get("sub_branch_variation", 1)
        num_sub_branches = max(0, params["num_sub_branches"] + random.randint(-sub_branch_count_variation, sub_branch_count_variation))

        # Create sub-branches
        for i in range(num_sub_branches):
            # Randomize sub-branch position along main branch
            if params.get("position_variation", True) and depth > 1:
                branch_pos = random.uniform(0.3, 1.0)
                sub_start_x = start_x + branch_pos * (end_x - start_x)
                sub_start_y = start_y + branch_pos * (end_y - start_y)
            else:
                sub_start_x = end_x
                sub_start_y = end_y

            # Calculate sub-branch angle
            if params.get("random_angles", False):
                sub_angle = random.uniform(0, 2 * math.pi)
            else:
                if i < num_sub_branches // 2:
                    side_factor = -1.0  # Left side
                else:
                    side_factor = 1.0  # Right side

                angle_spread = params.get("angle_spread", 0.7)
                sub_angle = angle + side_factor * (params["sub_branch_angle"] * (1.0 + random.uniform(-angle_spread, angle_spread)))

            # Sub-branch length
            length_ratio = params["sub_branch_length_ratio"]
            if hasattr(params, "right_bias") and side_factor > 0:
                length_ratio *= params["right_bias"]

            sub_length_variation = params.get("sub_length_variation", 0.3)
            sub_length = branch_length * length_ratio * (1.0 + random.uniform(-sub_length_variation, sub_length_variation))

            # Recursive call with depth-1
            self.add_organic_structure(
                sub_start_x, sub_start_y, 
                depth - 1, 
                sub_length,
                sub_angle,
                (actuation_start + 1) % max(1, params.get("num_actuators", 5)),
                ptype,
                params
            )

    def finalize(self):
        global n_particles, n_solid_particles, n_actuators
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        n_actuators = self.n_actuators  # Update global n_actuators
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)
        print('n_actuators', n_actuators)
    
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



def adaptive_particle_density(params, max_particles):
    """Dynamically adjust particle density based on structural complexity"""
    # Estimate particle count with current params
    est_count = estimate_particle_count(params)
    
    # If we're close to the limit, reduce particle density
    if est_count > max_particles * 0.7:  # 70% of limit
        # Calculate reduction factor to aim for 60% of max
        target_count = max_particles * 0.6
        reduction_factor = target_count / est_count
        
        print(f"Reducing particle density: est={est_count}, target={target_count}, factor={reduction_factor:.3f}")
        
        # Apply reduction factors to different parameters based on their impact
        # Branch length reduction (has major impact)
        params["branch_length"] = max(0.03, params["branch_length"] * reduction_factor**0.5)
        
        # Thickness reduction (also significant impact)
        params["thickness"] = max(0.002, params["thickness"] * reduction_factor**0.3)
        
        # Depth reduction for very large structures
        if reduction_factor < 0.5 and params["depth"] > 1:
            params["depth"] -= 1
            print("Reducing depth to save memory")
            
        # Reduce sub-branches as a last resort
        if reduction_factor < 0.3 and params["num_sub_branches"] > 2:
            params["num_sub_branches"] -= 1
            print("Reducing sub-branches to save memory")
    
    return params

# Improved parameter generation for interesting and effective structures
def generate_optimized_params(max_particles):
    """Generate parameters optimized for horizontal movement and memory constraints"""
    # Start with parameters known to work well
    params = {
        "start_x": random.uniform(0.1, 0.2),
        "start_y": random.uniform(0.05, 0.1),  # Lower position helps with ground contact
        "depth": random.randint(2, 3),  # Control recursion depth
        "branch_length": random.uniform(0.05, 0.09),
        "angle": random.uniform(-math.pi/6, math.pi/6),  # Bias toward horizontal angles
        "thickness": random.uniform(0.003, 0.008),
        "stiffness": random.uniform(450.0, 550.0),
        "damping": random.uniform(0.03, 0.06),
        "num_sub_branches": random.randint(3, 5),
        # More horizontal sub-branches
        "sub_branch_angle": random.uniform(math.pi/6, math.pi/3),  
        "sub_branch_length_ratio": random.uniform(0.5, 0.7),
        "actuation_start": 0,
        "ptype": 1
    }
    
    # Create asymmetry to encourage directional movement
    # Add bias for right-side branches to be slightly longer
    if random.random() < 0.7:  # 70% chance of rightward bias
        params["right_bias"] = random.uniform(1.1, 1.3)
    else:
        params["right_bias"] = 1.0
    
    # Adjust particle density based on structure complexity
    return adaptive_particle_density(params, max_particles)

def create_complex_robot(scene, params):
    """Create a snowflake-like structure using the provided parameters"""
    scene.set_offset(0.0, 0.03) 
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

# Add this function to estimate particle count before creating the structure
def estimate_particle_count(params):
    """Improved estimate of particle count for a given parameter set"""
    # Base count for main branches
    points_per_branch = max(3, int(params["branch_length"] / (dx * (params["depth"] + 1))))
    
    # Adjust particle thickness based on depth
    # Terminal branches are single line, others are 3 particles thick
    avg_thickness = 1 + (params["depth"] - 1) * 2  # Approximation
    particles_per_point = avg_thickness
    
    # Calculate total branches more accurately with asymmetry consideration
    d = params["depth"]
    b = params["num_sub_branches"]
    
    # Account for asymmetry and depth reduction on left side
    if b == 1:
        total_branches = d
    else:
        # Right side branches have full depth
        right_branches = (b//2) * (b**(d-1) - 1) // (b - 1) if b > 1 else d
        
        # Left side branches have reduced depth (approximately by 1)
        left_depth = max(1, d - 1)
        left_branches = (b - b//2) * (b**(left_depth-1) - 1) // (b - 1) if b > 1 else left_depth
        
        total_branches = 1 + right_branches + left_branches  # Include main branch
    
    # Add base platform particles
    base_platform = int((0.05 / dx) * 2) * int((0.015 / dx) * 2)
    
    # Estimate total particle count
    total_particles = (total_branches * points_per_branch * particles_per_point) + base_platform
    
    # Add safety margin
    return int(total_particles * 1.1)  # 10% safety margin

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
    save_params_to_csv(params, "param-test.csv")
    print(f"Testing configuration: {params}")
    try:
        result = subprocess.run(
            ["python", "view_snow2.py"],
            check=True,
            capture_output=True,
            text=True
        )
        print("Configuration test passed.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Configuration test failed: {e.stderr}")
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
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"run_{current_time}"
    os.makedirs(run_folder, exist_ok=True)
    print(f"Created run folder: {run_folder}")

    valid_configs = []
    config_count = 0

    while len(valid_configs) < 8:
        config_count += 1
        config_file = os.path.join(run_folder, f"config{config_count}.csv")
        params = generate_parameters()
        if test_configuration(params):
            print(f"Configuration {config_count} is valid. Saving to {config_file}.")
            valid_configs.append(params)
            save_params_to_csv(params, config_file)
        else:
            print(f"Configuration {config_count} is invalid. Regenerating...")

    print(f"Successfully generated {len(valid_configs)} valid configurations:")
    for i, config in enumerate(valid_configs):
        print(f"Config {i + 1}: {config}")

def run_evolution_with_preset_population(population_folder, num_generations=5, max_particles=2000):
    """
    Run a full evolution simulation using a preset population loaded from CSV files.
    
    Args:
        population_folder: Folder containing CSV files with structure configurations
        num_generations: Number of generations to run
        max_particles: Maximum number of particles allowed
        
    Returns:
        Best structure found during evolution
    """
    print(f"Running evolution with preset population from {population_folder}")
    print(f"Number of generations: {num_generations}")
    
    # Create a folder for this run
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"preset_run_{current_time}"
    os.makedirs(run_folder, exist_ok=True)
    
    # Load population from CSV files
    population = []
    
    # Handle the nested folder structure
    # For each structure directory, look for CSV files inside it
    for item in os.listdir(population_folder):
        item_path = os.path.join(population_folder, item)
        
        # Check if it's a directory containing structure
        if os.path.isdir(item_path) and item.startswith("structure_"):
            # Look for CSV files inside this directory
            for file in os.listdir(item_path):
                if file.endswith(".csv") and "snowflake_config_" in file:
                    try:
                        file_path = os.path.join(item_path, file)
                        params = load_params_from_csv(file_path)
                        population.append(params)
                        print(f"Loaded {item}/{file}")
                    except Exception as e:
                        print(f"Error loading {item}/{file}: {e}")
    
    if not population:
        print(f"No valid structure configurations found in {population_folder}")
        print("Check if the CSV files are inside structure_X directories")
        return None
    
    print(f"Successfully loaded {len(population)} structures")
    
    # Initialize with reasonable limits
    global n_particles, n_solid_particles, n_actuators
    n_particles = max_particles
    n_solid_particles = max_particles
    n_actuators = 10
    
    try:
        # Initialize Taichi
        ti.init(default_fp=real, arch=ti.gpu, flatten_if=True, device_memory_fraction=0.7)
        allocate_fields()
        
        # Add necessary methods to Scene class
        Scene.add_branching_structure = add_branching_structure
        
        # Run the evolution with the preset population
        best_structure = evolutionary_optimization_v2(
            population_size=len(population),
            num_generations=num_generations,
            run_folder=run_folder,
            max_particles=max_particles,
            initial_population=population  # Pass the loaded population
        )
        
        # Save the best structure
        save_params_to_csv(best_structure, os.path.join(run_folder, "best_structure.csv"))
        print(f"Best structure saved to {os.path.join(run_folder, 'best_structure.csv')}")
        
        # Visualize the best structure
        try:
            reset_fields()
            
            scene = Scene()
            create_snowflake_structure(scene, best_structure)
            
            for i in range(scene.n_particles):
                x[0, i] = scene.x[i]
                F[0, i] = [[1, 0], [0, 1]]
                actuator_id[i] = scene.actuator_id[i]
                particle_type[i] = scene.particle_type[i]
            
            n_particles = scene.n_particles
            n_solid_particles = scene.n_solid_particles
            n_actuators = scene.n_actuators
            
            forward()
            
            vis_folder = os.path.join(run_folder, "visualization")
            for s in range(0, steps, steps//20):
                visualize(s, vis_folder)
                
        except Exception as e:
            print(f"Visualization error: {e}")
        
        return best_structure
        
    except Exception as e:
        print(f"Error during evolution: {e}")
        return None
    finally:
        import gc
        gc.collect()
        try:
            ti.reset()
        except:
            print("Warning: Error during ti.reset()")
        print("Evolution complete!")

# Modified evolutionary_optimization_v2 function to accept an initial population
def evolutionary_optimization_v2(population_size, num_generations, run_folder, max_particles, initial_population=None):
    """
    Improved evolutionary algorithm with memory management and specialization.
    Can start with a preset population loaded from files.
    
    Args:
        population_size: Size of the population in each generation
        num_generations: Number of generations to run
        run_folder: Folder to save results
        max_particles: Maximum number of particles allowed
        initial_population: Optional preset population to start with
        
    Returns:
        Best structure parameters found during evolution
    """
    # Initialize population either from preset or randomly
    if initial_population is not None:
        print(f"Starting with preset population of {len(initial_population)} structures")
        population = initial_population.copy()
        
        # Ensure the population size matches the expected size
        while len(population) < population_size:
            print("Adding additional random structures to reach desired population size")
            population.append(generate_optimized_params(max_particles))
            
        if len(population) > population_size:
            print(f"Warning: Preset population is larger than specified size ({len(population)} > {population_size})")
            print(f"Using the first {population_size} structures")
            population = population[:population_size]
    else:
        print(f"Generating random initial population of {population_size} structures")
        population = [generate_optimized_params(max_particles) for _ in range(population_size)]
    
    # Rest of the function remains the same as before
    all_fitness_scores = []
    best_structure = None
    best_fitness = -float('inf')
    successful_structures = []
    
    for generation in range(num_generations):
        print(f"Generation {generation + 1}/{num_generations}")
        fitness_scores = []
        gen_folder = os.path.join(run_folder, f"gen_{generation + 1}")
        os.makedirs(gen_folder, exist_ok=True)
        
        # Periodically clear memory
        if generation > 0 and generation % 2 == 0:
            import gc
            gc.collect()
            reset_fields()
            ti.sync()
        
        for idx, params in enumerate(population):
            print(f"Evaluating structure {idx + 1}/{len(population)}")
            
            # Adjust parameters to fit within memory constraints
            est_particles = estimate_particle_count(params)
            if est_particles > max_particles * 0.85:
                print(f"Estimated particles ({est_particles}) exceeds 85% of limit. Adjusting structure.")
                params = adaptive_particle_density(params, max_particles)
            
            # Try to create and evaluate the structure
            try:
                scene = Scene()
                create_snowflake_structure(scene, params)
                
                if scene.n_particles > max_particles:
                    print(f"Structure {idx + 1} has too many particles ({scene.n_particles}). Skipping.")
                    fitness_scores.append(-1000)
                    continue
                
                # Reset fields before simulation
                reset_fields()
                
                # Initialize particles
                for i in range(scene.n_particles):
                    x[0, i] = scene.x[i]
                    F[0, i] = [[1, 0], [0, 1]]
                    actuator_id[i] = scene.actuator_id[i]
                    particle_type[i] = scene.particle_type[i]
                
                # Update global counts
                global n_particles, n_solid_particles, n_actuators
                n_particles = scene.n_particles
                n_solid_particles = scene.n_solid_particles
                n_actuators = scene.n_actuators
                
                # Calculate initial position
                x_initial = 0.0
                initial_count = 0
                for i in range(scene.n_particles):
                    if particle_type[i] == 1:
                        x_initial += x[0, i][0]
                        initial_count += 1
                if initial_count > 0:
                    x_initial /= initial_count
                
                # Run simulation
                try:
                    ti.sync()
                    
                    # Use reduced steps for large structures
                    if scene.n_particles > max_particles * 0.7:
                        reduced_steps = steps // 2
                        forward(reduced_steps)
                    else:
                        forward()
                    
                    # Calculate fitness
                    x_final = x_avg[None][0]
                    horizontal_distance = x_final - x_initial
                    fitness = horizontal_distance * 10
                    
                    # Add efficiency bonus
                    efficiency_factor = 1.0 + (max_particles - scene.n_particles) / max_particles
                    fitness *= efficiency_factor
                    
                    ti.sync()
                    
                except Exception as e:
                    print(f"Simulation error: {e}")
                    fitness = -1000
                
                fitness_scores.append(fitness)
                all_fitness_scores.append((fitness, params))
                
                # Track best structure
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_structure = params.copy()
                    print(f"New best structure! Fitness: {fitness}")
                
                # Save successful structures
                if fitness > 0:
                    successful_structures.append((fitness, params.copy()))
                
                # Save configuration
                save_params_to_csv(params, os.path.join(gen_folder, f"structure_{idx + 1}.csv"))
                with open(os.path.join(gen_folder, "fitness_scores.txt"), "a") as f:
                    f.write(f"Structure {idx + 1}: {fitness}\n")
                
                # Clean up
                import gc
                gc.collect()
                import time
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Structure creation error: {e}")
                fitness_scores.append(-1000)
                continue
        
        # Check for valid structures
        if all(score <= 0 for score in fitness_scores):
            print("No effective structures found. Generating new population.")
            population = [generate_optimized_params(max_particles) for _ in range(len(population))]
            continue
        
        # Create next generation
        if successful_structures:
            # Sort by fitness
            successful_structures.sort(key=lambda x: x[0], reverse=True)
            
            # Keep only the top structures to avoid memory bloat
            if len(successful_structures) > len(population) * 2:
                successful_structures = successful_structures[:len(population) * 2]
            
            # Weight selection by fitness
            fitness_values = [s[0] for s in successful_structures]
            total_fitness = sum(fitness_values)
            
            if total_fitness > 0:
                selection_probs = [f/total_fitness for f in fitness_values]
                
                # Create new population
                new_population = []
                
                # Elitism - keep the best structure
                new_population.append(successful_structures[0][1])
                
                # Fill rest with evolved structures
                while len(new_population) < len(population):
                    # Select parents weighted by fitness
                    parent_indices = random.choices(
                        range(len(successful_structures)),
                        weights=selection_probs,
                        k=2
                    )
                    parent1 = successful_structures[parent_indices[0]][1]
                    parent2 = successful_structures[parent_indices[1]][1]
                    
                    # Create child through crossover
                    child = crossover(parent1, parent2)
                    
                    # Mutate with varying rates based on generation
                    mutation_rate = 0.1 + 0.05 * (num_generations - generation) / num_generations
                    child = mutate_v2(child, mutation_rate)
                    
                    # Make sure child is within memory constraints
                    child = adaptive_particle_density(child, max_particles)
                    
                    new_population.append(child)
                
                population = new_population
            else:
                # Fallback to random generation
                population = [generate_optimized_params(max_particles) for _ in range(len(population))]
        else:
            # No successful structures, generate new population
            population = [generate_optimized_params(max_particles) for _ in range(len(population))]
        
        # Clear memory at end of generation
        reset_fields()
        ti.sync()
        gc.collect()
    
    # Return the best structure found
    return best_structure or generate_optimized_params(max_particles)

def evolutionary_optimization(population_size, num_generations, run_folder, max_particles):
    population = [randomize_snowflake_params() for _ in range(population_size)]
    all_fitness_scores = []
    
    for generation in range(num_generations):
        print(f"Generation {generation + 1}/{num_generations}")
        fitness_scores = []
        gen_folder = os.path.join(run_folder, f"gen_{generation + 1}")
        os.makedirs(gen_folder, exist_ok=True)

        for idx in range(population_size):
            print(f"Evaluating structure {idx + 1}/{population_size}")
            
            # Retry mechanism for generating valid structures
            retry_count = 0
            max_retries = 10  # Maximum number of retries
            fitness = -1000  # Default fitness for invalid structures
            
            while retry_count < max_retries:
                # Generate new parameters
                params = randomize_snowflake_params()
                
                # Adjust parameters to fit within memory constraints
                est_particles = estimate_particle_count(params)
                if est_particles > max_particles * 0.9:  # 90% safety margin
                    print(f"Estimated particles ({est_particles}) exceeds limit. Adjusting structure.")
                    params = adaptive_particle_density(params, max_particles)
                
                # Try to create and evaluate the structure
                try:
                    scene = Scene()
                    create_snowflake_structure(scene, params)
                    
                    if scene.n_particles > max_particles:
                        print(f"Structure {idx + 1} has too many particles ({scene.n_particles}). Retrying...")
                        retry_count += 1
                        continue  # Skip to the next retry
                    
                    # Reset fields before simulation
                    reset_fields()
                    
                    # Initialize particles
                    for i in range(scene.n_particles):
                        x[0, i] = scene.x[i]
                        F[0, i] = [[1, 0], [0, 1]]
                        actuator_id[i] = scene.actuator_id[i]
                        particle_type[i] = scene.particle_type[i]
                    
                    # Update global counts
                    global n_particles, n_solid_particles, n_actuators
                    n_particles = scene.n_particles
                    n_solid_particles = scene.n_solid_particles
                    n_actuators = scene.n_actuators
                    
                    # Calculate initial position
                    x_initial = 0.0
                    initial_count = 0
                    for i in range(scene.n_particles):
                        if particle_type[i] == 1:
                            x_initial += x[0, i][0]
                            initial_count += 1
                    if initial_count > 0:
                        x_initial /= initial_count
                    
                    # Run simulation
                    try:
                        ti.sync()
                        
                        # Use reduced steps for large structures
                        if scene.n_particles > max_particles * 0.7:
                            reduced_steps = steps // 2
                            forward(reduced_steps)
                        else:
                            forward()
                        
                        # Calculate fitness
                        x_final = x_avg[None][0]
                        horizontal_distance = x_final - x_initial
                        fitness = horizontal_distance * 10
                        
                        # Add efficiency bonus
                        efficiency_factor = 1.0 + (max_particles - scene.n_particles) / max_particles
                        fitness *= efficiency_factor
                        
                        ti.sync()
                        
                    except Exception as e:
                        print(f"Simulation error: {e}")
                        fitness = -1000
                    
                    # Break out of the retry loop if a valid structure is found
                    break
                
                except Exception as e:
                    print(f"Structure creation error: {e}")
                    retry_count += 1
                    continue
            
            # If max retries reached, use the last generated structure (even if invalid)
            if retry_count >= max_retries:
                print(f"Max retries reached for structure {idx + 1}. Using last generated structure.")
            
            fitness_scores.append(fitness)
            all_fitness_scores.append((fitness, params))

            # Save the configuration and fitness score
            config_file = os.path.join(gen_folder, f"structure_{idx + 1}.csv")
            save_params_to_csv(params, config_file)
            with open(os.path.join(gen_folder, "fitness_scores.txt"), "a") as f:
                f.write(f"Structure {idx + 1}: {fitness}\n")
            
            # Clean up
            import gc
            gc.collect()
            import time
            time.sleep(0.2)

        # Check if we have any valid structures
        if all(score == -1000 for score in fitness_scores):
            print("No effective structures found. Generating new population.")
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

def generate_parameters():
    params = {
        "start_x": random.uniform(0.0, 0.2),
        "start_y": random.uniform(0.4, 0.6),
        "depth": random.randint(2, 4),
        "branch_length": random.uniform(0.05, 0.2),
        "angle": random.uniform(0.0, 2 * math.pi),
        "thickness": random.uniform(0.005, 0.02),
        "stiffness": random.uniform(400.0, 600.0),
        "damping": random.uniform(0.03, 0.07),
        "num_sub_branches": random.randint(3, 7),
        "sub_branch_angle": random.uniform(0.5, 1.5),
        "sub_branch_length_ratio": random.uniform(0.5, 0.7),
        "actuation_start": 0,
        "ptype": 1
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