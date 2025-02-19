import taichi as ti
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt

real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

dim = 2
n_particles = 8192
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

    def add_parametric_curve(self, start_x, start_y, params, n_points, actuation, ptype=1):
        """Add particles along a parametric curve
        params: dict with curve parameters (e.g., amplitude, frequency)"""
        for i in range(n_points):
            t = i / (n_points - 1)
            # Parametric equations (e.g., for a sine wave)
            x_pos = start_x + t * params.get('length', 0.5)
            y_pos = start_y + params.get('amplitude', 0.1) * math.sin(2 * math.pi * params.get('frequency', 1) * t)
            
            self.x.append([x_pos + self.offset_x, y_pos + self.offset_y])
            self.actuator_id.append(actuation)
            self.particle_type.append(ptype)
            self.n_particles += 1
            self.n_solid_particles += int(ptype == 1)
            if actuation != -1:
                self.n_actuators = max(self.n_actuators, actuation + 1)

    def add_branching_structure(self, start_x, start_y, depth, branch_length, angle, actuation_start, ptype=1):
        """Create a recursive branching structure (e.g., tree-like)"""
        if depth <= 0:
            return
            
        # Add main branch
        end_x = start_x + branch_length * math.cos(angle)
        end_y = start_y + branch_length * math.sin(angle)
        
        # Add particles along the branch
        n_points = max(3, int(branch_length / dx * 2))
        for i in range(n_points):
            t = i / (n_points - 1)
            x_pos = start_x + t * (end_x - start_x)
            y_pos = start_y + t * (end_y - start_y)
            
            self.x.append([x_pos + self.offset_x, y_pos + self.offset_y])
            self.actuator_id.append(actuation_start)
            self.particle_type.append(ptype)
            self.n_particles += 1
            self.n_solid_particles += int(ptype == 1)
            if actuation_start != -1:
                self.n_actuators = max(self.n_actuators, actuation_start + 1)
            
        # Create sub-branches
        branch_angle = math.pi / 4  # 45 degrees
        new_length = branch_length * 0.7  # Shorter sub-branches
        
        self.add_branching_structure(end_x, end_y, depth - 1, new_length, 
                                   angle + branch_angle, actuation_start + 1, ptype)
        self.add_branching_structure(end_x, end_y, depth - 1, new_length, 
                                   angle - branch_angle, actuation_start + 2, ptype)

    def add_spring_chain(self, start_x, start_y, n_segments, segment_length, actuation_pattern='alternating'):
        """Create a chain of segments connected by springs"""
        prev_end_idx = None
        
        for i in range(n_segments):
            # Determine actuation ID based on pattern
            if actuation_pattern == 'alternating':
                act_id = i % 2
            elif actuation_pattern == 'sequential':
                act_id = i
            else:
                act_id = -1
                
            # Add segment
            x_pos = start_x + i * segment_length
            segment_start_idx = self.n_particles
            
            # Add particles for current segment
            self.add_rect(x_pos, start_y, segment_length * 0.8, segment_length * 0.2, act_id)
            
            # Connect with previous segment
            if prev_end_idx is not None:
                self.connections.append({
                    'type': 'spring',
                    'start': prev_end_idx,
                    'end': segment_start_idx,
                    'stiffness': 1000.0
                })
                
            prev_end_idx = self.n_particles - 1

    def add_chain(self, start_x, start_y, n_segments, segment_radius, spacing):
        """Create a chain of circles"""
        for i in range(n_segments):
            x_pos = start_x + i * (segment_radius * 2 + spacing)
            actuation = i % 2  # Alternate actuation between segments
            self.add_circle(x_pos, start_y, segment_radius, actuation)
            self.n_actuators = max(self.n_actuators, actuation + 1)

    def add_tree(self, start_x, start_y, depth, size):
        """Create a tree-like structure"""
        if depth <= 0:
            return
        
        # Add main node
        self.add_circle(start_x, start_y, size, depth % 2)
        
        if depth > 1:
            # Add two branches
            angle = math.pi / 4  # 45 degrees
            new_size = size * 0.7
            spacing = size * 3
            
            # Left branch
            left_x = start_x - spacing * math.cos(angle)
            left_y = start_y + spacing * math.sin(angle)
            self.add_tree(left_x, left_y, depth - 1, new_size)
            
            # Right branch
            right_x = start_x + spacing * math.cos(angle)
            right_y = start_y + spacing * math.sin(angle)
            self.add_tree(right_x, right_y, depth - 1, new_size)

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

def create_complex_robot(scene):
    """Create a more complex robot structure"""
    # Base structure - a chain of circles
    # WORKs: a line of hori balls
    # scene.add_chain(0.3, 0.5, 5, 0.05, 0.02)       
    
    # Add a tree structure on top
    # WORKS: a bunch of symmetric circles
    # scene.add_tree(0.5, 0.7, 3, 0.05)           
    
    # Add a parametric curve (e.g., a sine wave)
    # DOES NOT WORK
    # scene.add_parametric_curve(0.2, 0.3, {'length': 0.4, 'amplitude': 0.1, 'frequency': 2}, 10, 0)
    
    # Add a branching structure (e.g., a tree)
    # WORKS BUT: squiggly lines drop on the ground
    # scene.add_branching_structure(0.7, 0.3, 3, 0.1, -math.pi / 2, 1)
    
    # Add a spring chain (e.g., a flexible limb)
    # WORKS: hori rectangles in a line
    # scene.add_spring_chain(0.1, 0.2, 4, 0.1, 'alternating')
    
    scene.finalize()


def fish(scene):
    scene.add_rect(0.025, 0.025, 0.95, 0.1, -1, ptype=0)
    scene.add_rect(0.1, 0.2, 0.15, 0.05, -1)
    scene.add_rect(0.1, 0.15, 0.025, 0.05, 0)
    scene.add_rect(0.125, 0.15, 0.025, 0.05, 1)
    scene.add_rect(0.2, 0.15, 0.025, 0.05, 2)
    scene.add_rect(0.225, 0.15, 0.025, 0.05, 3)
    scene.set_n_actuators(4)


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100)
    options = parser.parse_args()

    # Initialize scene with complex robot
    scene = Scene()
    create_complex_robot(scene)  # This will set n_actuators
    # fish(scene)  # This will set n_actuators
    scene.finalize()  # Finalize the scene to update n_actuators
    allocate_fields()  # Allocate fields after n_actuators is set

    # Initialize particle positions, deformation gradient, and velocity
    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]

    # Optimization loop
    losses = []
    for iter in range(options.iters):
        with ti.ad.Tape(loss):
            forward()
        l = loss[None]
        losses.append(l)
        print('i=', iter, 'loss=', l)
        learning_rate = 0.1

        # Update weights and biases
        for i in range(n_actuators):
            for j in range(n_sin_waves):
                weights[i, j] -= learning_rate * weights.grad[i, j]
            bias[i] -= learning_rate * bias.grad[i]

        # Visualize every 10 iterations
        if iter % 10 == 0:
            forward(1500)
            for s in range(15, 1500, 16):
                visualize(s, 'diffmpm/iter{:03d}/'.format(iter))

    # Plot loss
    plt.title("Optimization of Initial Velocity")
    plt.ylabel("Loss")
    plt.xlabel("Gradient Descent Iterations")
    plt.plot(losses)
    plt.show()

if __name__ == '__main__':
    main()