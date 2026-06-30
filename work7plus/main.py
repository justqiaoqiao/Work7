import taichi as ti

ti.init(arch=ti.gpu)

# --- 物理参数 ---
N = 20
mass = 1.0
dt = 1e-4
k_s = 8000.0   # 结构刚度
k_sh = 4000.0  # 剪切刚度
k_b = 2000.0   # 弯曲刚度
k_d = 20.0     # 阻尼
gravity = ti.Vector([0.0, -9.8, 0.0])

# --- 球体配置 ---
sphere_pos = ti.Vector.field(3, dtype=float, shape=1)
sphere_radius = 0.3

# --- 数据场 ---
x = ti.Vector.field(3, dtype=float, shape=N * N)
v = ti.Vector.field(3, dtype=float, shape=N * N)
f = ti.Vector.field(3, dtype=float, shape=N * N)

@ti.dataclass
class Spring:
    p1: ti.i32; p2: ti.i32
    length: ti.f32; stiffness: ti.f32

max_springs = N * N * 16
springs = Spring.field(shape=max_springs)
num_springs = ti.field(dtype=int, shape=())

# 渲染网格索引
indices = ti.field(dtype=int, shape=(N - 1) * (N - 1) * 6)

@ti.kernel
def init_cloth():
    sphere_pos[0] = ti.Vector([0.0, -0.2, 0.0])
    num_springs[None] = 0
    # 初始化点
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.5, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        
    # 初始化三种弹簧 (结构、剪切、弯曲)
    for i, j in ti.ndrange(N, N):
        for dx, dy, ks in ti.static([(0,1,k_s), (1,0,k_s), (1,1,k_sh), (1,-1,k_sh), (0,2,k_b), (2,0,k_b)]):
            ni, nj = i + dx, j + dy
            if 0 <= ni < N and 0 <= nj < N:
                idx = i * N + j; nidx = ni * N + nj
                c = ti.atomic_add(num_springs[None], 1)
                springs[c].p1, springs[c].p2 = idx, nidx
                springs[c].length = (x[idx] - x[nidx]).norm()
                springs[c].stiffness = ks

    # 初始化渲染索引
    for i, j in ti.ndrange(N - 1, N - 1):
        idx = (i * (N - 1) + j) * 6
        v1, v2, v3, v4 = i*N+j, i*N+j+1, (i+1)*N+j, (i+1)*N+j+1
        for k, val in ti.static(enumerate([v1, v2, v3, v2, v4, v3])):
            indices[idx + k] = val

@ti.func
def compute_forces():
    for i in range(N * N): f[i] = gravity * mass - k_d * v[i]
    for i in range(num_springs[None]):
        s = springs[i]
        d = x[s.p1] - x[s.p2]
        dist = d.norm()
        if dist > 1e-5:
            force = -s.stiffness * (dist - s.length) * (d / dist)
            ti.atomic_add(f[s.p1], force); ti.atomic_add(f[s.p2], -force)

@ti.kernel
def step():
    compute_forces()
    for i in range(N * N):
        if not (i == 0 or i == N - 1): # 固定布料两角
            v[i] += f[i] / mass * dt
            x[i] += v[i] * dt
            # 碰撞检测：增加微小偏移防止 Z-fighting
            dist_to_sphere = (x[i] - sphere_pos[0]).norm()
            if dist_to_sphere < sphere_radius:
                x[i] = sphere_pos[0] + (x[i] - sphere_pos[0]).normalized() * (sphere_radius + 0.01)
                v[i] *= 0.5 # 简单的碰撞摩擦

def main():
    init_cloth()
    window = ti.ui.Window("Cloth Simulation", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene(); camera = ti.ui.Camera()
    camera.position(0, 0.0, 2.5); camera.lookat(0, 0, 0)
    
    while window.running:
        # --- 交互逻辑：按 R 重置 ---
        while window.get_event(ti.ui.PRESS):
            if window.event.key == 'r':
                init_cloth()
        
        # --- GUI 面板 ---
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.3, 0.1)
        window.GUI.text("Press 'R' to reset the cloth")
        window.GUI.end()

        # --- 物理步进 ---
        for _ in range(50): step()
        
        # --- 渲染 ---
        scene.set_camera(camera)
        scene.point_light(pos=(1, 1, 1), color=(1, 1, 1))
        scene.ambient_light((0.3, 0.3, 0.3))
        
        scene.mesh(x, indices=indices, color=(0.2, 0.8, 1.0), two_sided=True)
        scene.particles(sphere_pos, radius=sphere_radius, color=(0.9, 0.2, 0.2))
        
        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()