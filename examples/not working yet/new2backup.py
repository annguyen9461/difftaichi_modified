import taichi as ti
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Initialize Taichi
real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

# Constants
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
mu = E
la = E
max_steps = 2048
steps = 1024
gravity = 3.8
target = [0.8, 0.2]

# Fields
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


class MPMSimulator:
    def __init__(self):
        self.allocate_fields()

    def allocate_fields(self):
        ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
        ti.root.dense(ti.i, n_actuators).place(bias)
        ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
        ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
        ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
        ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
        ti.root.place(loss, x_avg)
        ti.root.lazy_grad()

    @ti.kernel
    def clear_grid(self):
        for i, j in grid_m_in:
            grid_v_in[i, j] = [0, 0]
            grid_m_in[i, j] = 0
            grid_v_in.grad[i, j] = [0, 0]
            grid_m_in.grad[i, j] = 0
            grid_v_out.grad[i, j] = [0, 0]

    @ti.kernel
    def p2g(self, f: ti.i32):
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

    @ti.kernel
    def grid_op(self):
        bound = 3
        coeff = 0.5
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
            if j > n_grid - bound and v_out[1] > 0:
                v_out[0] = 0
                v_out[1] = 0
            grid_v_out[i, j] = v_out

    @ti.kernel
    def g2p(self, f: ti.i32):
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

    def advance(self, s):
        self.clear_grid()
        self.p2g(s)
        self.grid_op()
        self.g2p(s)


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0

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

    def add_circle(self, center_x, center_y, radius, actuation, ptype=1):
        global n_particles
        spacing = dx / 2
        for i in range(int((center_x - radius) / spacing), int((center_x + radius) / spacing) + 1):
            for j in range(int((center_y - radius) / spacing), int((center_y + radius) / spacing) + 1):
                x_pos = center_x + (i * spacing - center_x)
                y_pos = center_y + (j * spacing - center_y)
                if (x_pos - center_x) ** 2 + (y_pos - center_y) ** 2 <= radius ** 2:
                    self.x.append([x_pos + self.offset_x, y_pos + self.offset_y])
                    self.actuator_id.append(actuation)
                    self.particle_type.append(ptype)
                    self.n_particles += 1
                    self.n_solid_particles += int(ptype == 1)

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100)
    options = parser.parse_args()

    # Initialize simulation
    simulator = MPMSimulator()

    # Create scene
    scene = Scene()
    scene.add_circle(0.5, 0.9, 0.05, -1, ptype=1)
    scene.add_circle(0.5, 0.8, 0.05, -1, ptype=1)
    scene.add_circle(0.5, 0.7, 0.05, -1, ptype=1)
    scene.add_circle(0.5, 0.6, 0.05, -1, ptype=1)
    scene.add_circle(0.5, 0.5, 0.05, -1, ptype=1)
    scene.set_n_actuators(1)
    scene.finalize()

    # Initialize particles and fields
    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]

    # Optimization loop
    losses = []
    for iter in range(options.iters):
        with ti.ad.Tape(loss):
            simulator.advance(iter)
        losses.append(loss[None])
        print(f'i={iter}, loss={loss[None]}')

    # Plot losses
    plt.title("Optimization of Initial Velocity")
    plt.ylabel("Loss")
    plt.xlabel("Gradient Descent Iterations")
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    main()
