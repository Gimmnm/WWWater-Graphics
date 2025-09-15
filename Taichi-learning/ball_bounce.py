import numpy as np
import taichi as ti  # 导入 Taichi 框架（数值并行、GUI）

# 初始化后端（兼容不同平台/版本）：
# - 优先尝试 GPU（统一用 ti.gpu 让 Taichi 自选可用 GPU 后端，如 metal/vulkan/cuda）
# - 若失败则自动回退到 CPU
try:
    ti.init(arch=ti.gpu)
except Exception:
    ti.init(arch=ti.cpu)

# ----------------------
# 仿真参数（可按需调整）
# ----------------------
# 时间离散设置：总的仿真步长为 substeps 次子步，每次子步的时间为 dt
dt = 1.0 / 360.0         # 单个子步的时间步长（秒）：越小越稳定，但计算越多
substeps = 8               # 每帧进行多少个物理子步：增大可减少穿透/爆震

# 场景与物理参数
gravity = ti.Vector([0.0, -9.8])  # 重力加速度（m/s^2）
radius = 0.02                     # 小球半径（归一化坐标：画布高度视为 1）
restitution = 0.7                 # 小球与地面/墙的恢复系数（0~1）
ground_y = 0.05                   # 地面高度（归一化坐标，留出边距防止视觉穿透）
ground_friction = 0.1             # 地面摩擦（碰撞瞬间对水平速度的衰减百分比）

# 横向控制参数（键盘 WASD 控制）
control_accel = 6.0               # 水平“加速度”强度（m/s^2 的等效值）
control_max_speed = 3.0           # 水平最大速度（防止过快冲出画面）

# 多球参数：总数量与可控球索引（默认控制第 0 个）
num_balls = 8
player_idx = 0

# ----------------------
# 障碍物（静态圆）参数
# ----------------------
num_obstacles = 3
obstacle_restitution = 0.7  # 小球与障碍物碰撞的恢复系数（法向）
obstacle_friction = 0.05    # 小球与障碍物的切向摩擦（越大切向速度衰减越明显）
obst_pos = ti.Vector.field(2, dtype=ti.f32, shape=num_obstacles)  # 每个障碍的圆心
obst_rad = ti.field(dtype=ti.f32, shape=num_obstacles)            # 每个障碍的半径

# ----------------------
# 球-球碰撞参数（同半径、等质量）
# ----------------------
ball_restitution = 0.75     # 球-球法向恢复系数（略低更稳）
ball_tangent_damping = 0.1  # 碰撞后的切向速度衰减比例（简化摩擦）
collision_iterations = 3    # 每个子步内球-球碰撞的求解迭代次数（提高贴合度）
restitution_threshold = 0.3 # 法向相对速度阈值：低于它视为静止接触（不反弹）
global_damping = 0.995      # 每子步施加全局速度衰减
max_speed = 6.0             # 速度夹限，防止数值发散

# ----------------------
# 状态量（多小球）
# ----------------------
pos = ti.Vector.field(2, dtype=ti.f32, shape=num_balls)  # 小球位置 [x, y]
vel = ti.Vector.field(2, dtype=ti.f32, shape=num_balls)  # 小球速度 [vx, vy]


@ti.kernel
def reset():
    """
    重置场景：
    - 将小球放在中央偏上的位置，并给予一个向右的初始速度；
    - 放置若干静态圆形障碍物（可根据需要调整位置与半径）。
    """
    # 1) 初始化所有小球：第一个为玩家控制球，其余随机
    for i in range(num_balls):
        if i == ti.static(player_idx):
            pos[i] = ti.Vector([0.5, 0.8])   # 玩家球：居中偏上
            vel[i] = ti.Vector([0.6, 0.0])   # 玩家球：初速度向右
        else:
            # 其他球随机分布与速度（范围内避免生成重叠边界）
            px = 0.1 + 0.8 * ti.random(ti.f32)
            py = 0.2 + 0.7 * ti.random(ti.f32)
            vx = (ti.random(ti.f32) * 2.0 - 1.0) * 0.5
            vy = (ti.random(ti.f32) * 2.0 - 1.0) * 0.5
            pos[i] = ti.Vector([px, py])
            vel[i] = ti.Vector([vx, vy])

    # 初始化静态圆形障碍物
    obst_pos[0] = ti.Vector([0.30, 0.25]); obst_rad[0] = 0.07
    obst_pos[1] = ti.Vector([0.70, 0.35]); obst_rad[1] = 0.06
    obst_pos[2] = ti.Vector([0.50, 0.55]); obst_rad[2] = 0.05


@ti.kernel
def substep():
    """
    单个物理子步：
    1) 施加重力更新速度（显式欧拉）：v ← v + g*dt
    2) 积分位置：x ← x + v*dt
    3) 处理与地面/墙面的碰撞反弹
    4) 处理与圆形障碍物的碰撞反弹
    """
    # 遍历每个小球，独立更新一个物理子步
    for i in range(num_balls):
        # 1) 重力：改变速度（仅 y 方向）
        vel[i] += gravity * dt

        # 2) 速度积分到位置
        pos[i] += vel[i] * dt

        # 3) 与地面（底边 y=ground_y）碰撞
        if pos[i].y - radius < ground_y:
            pos[i].y = ground_y + radius
            if vel[i].y < 0:
                vel[i].y = -vel[i].y * restitution
                vel[i].x *= (1.0 - ground_friction)

        # 顶部墙（上边 y=1）碰撞
        if pos[i].y + radius > 1.0:
            pos[i].y = 1.0 - radius
            if vel[i].y > 0:
                vel[i].y = -vel[i].y * restitution

        # 左墙（x=0）碰撞
        if pos[i].x - radius < 0.0:
            pos[i].x = radius
            if vel[i].x < 0:
                vel[i].x = -vel[i].x * restitution

        # 右墙（x=1）碰撞
        if pos[i].x + radius > 1.0:
            pos[i].x = 1.0 - radius
            if vel[i].x > 0:
                vel[i].x = -vel[i].x * restitution

        # 4) 与圆形障碍物碰撞
        for k in range(num_obstacles):
            rel = pos[i] - obst_pos[k]
            dist = rel.norm()
            R = radius + obst_rad[k]
            if dist < R:
                n = rel / (dist + 1e-6)
                pos[i] = obst_pos[k] + n * R
                v = vel[i]
                vn = v.dot(n)
                vt = v - vn * n
                if vn < 0:
                    vn = -vn * obstacle_restitution
                vt *= (1.0 - obstacle_friction)
                vel[i] = vt + vn * n

    # 5) 球-球碰撞（等质量、同半径）：位移校正 + 冲量法，迭代多次以减少“悬空”缝隙
    for _ in range(collision_iterations):
        for i in range(num_balls):
            for j in range(i + 1, num_balls):
                rel = pos[j] - pos[i]
                dist = rel.norm()
                min_dist = 2.0 * radius
                if dist < min_dist:
                    # 碰撞法线
                    n = rel / (dist + 1e-6)
                    # 位置修正：各自分担一半穿透量，使其正好相切
                    penetration = min_dist - dist
                    pos[i] -= 0.5 * penetration * n
                    pos[j] += 0.5 * penetration * n

                    # 法向相对速度（j 相对 i）
                    vrel_n = (vel[j] - vel[i]).dot(n)
                    # 低速接触不反弹：小于阈值时采用 e=0，提高“贴合”观感
                    e = 0.0 if -vrel_n < restitution_threshold else ball_restitution
                    if vrel_n < 0:  # 仅在相互接近时施加法向冲量
                        j_imp = -(1.0 + e) * vrel_n / 2.0  # 等质量简化
                        vel[i] -= j_imp * n
                        vel[j] += j_imp * n

                    # 切向相对速度与简化摩擦
                    t = ti.Vector([-n.y, n.x])
                    vrel_t = (vel[j] - vel[i]).dot(t)
                    j_t = -ball_tangent_damping * vrel_t / 2.0
                    vel[i] -= j_t * t
                    vel[j] += j_t * t

    # 6) 子步收尾：为所有小球施加轻微阻尼并限制最高速度，抑制“乱飞”
    for i in range(num_balls):
        # 速度夹限
        speed = vel[i].norm()
        if speed > max_speed:
            vel[i] = vel[i] / (speed + 1e-6) * max_speed
        # 全局轻微阻尼
        vel[i] *= global_damping


def main():
    # GUI 画布像素分辨率：宽 × 高
    res = (800, 600)
    gui = ti.GUI("Ball Bounce (Taichi)", res=res)

    reset()  # 初始化场景

    while gui.running:
        # —— 输入处理 ——
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key in [ti.GUI.ESCAPE, 'q']:
                break  # 退出
            if gui.event.key in ['r', 'R']:
                reset()  # 重置
            if gui.event.key in ['w', 'W']:  # W：给予向上的瞬时冲量（跳跃）
                vel_np = vel.to_numpy()
                vel_np[player_idx, 1] += 3.0
                vel.from_numpy(vel_np)

        # 连续按键检测：WASD（A/D 左右加速，S 刹车）
        frame_dt = substeps * dt  # 本帧等效的物理时间
        left_pressed = gui.is_pressed('a')
        right_pressed = gui.is_pressed('d')
        brake_pressed = gui.is_pressed('s')
        if left_pressed or right_pressed:
            vel_np = vel.to_numpy()
            ax = 0.0
            if left_pressed:
                ax -= control_accel
            if right_pressed:
                ax += control_accel
            # v_x ← v_x + a*Δt（Δt 用本帧时间）
            vel_np[player_idx, 0] += ax * frame_dt
            # 限制水平最大速度，避免数值发散
            vel_np[player_idx, 0] = float(np.clip(vel_np[player_idx, 0], -control_max_speed, control_max_speed))
            vel.from_numpy(vel_np)
        if brake_pressed:
            vel_np = vel.to_numpy()
            vel_np[player_idx, 0] *= 0.90
            vel.from_numpy(vel_np)

        # —— 物理更新 ——
        for _ in range(substeps):  # 多子步提高稳定性
            substep()

        # —— 绘制 ——（Taichi GUI 使用归一化坐标 [0,1]×[0,1]）
        gui.clear(color=0x112F41)

        # 地面（粗线模拟）
        y = ground_y
        gui.rect([0.0, y - 0.003], [1.0, y + 0.003], color=0x445566, radius=1)

        # 障碍物（静态圆）
        centers = obst_pos.to_numpy()
        radii = obst_rad.to_numpy()
        for k in range(num_obstacles):
            gui.circle(pos=centers[k], radius=int(np.round(radii[k] * res[1])), color=0x5DADE2)

        # 小球（多颗）：玩家球与其他球使用不同颜色
        pos_np = pos.to_numpy()
        r_px = int(np.round(radius * res[1]))
        for i in range(num_balls):
            color = 0xFDB863 if i == player_idx else 0x9B59B6
            gui.circle(pos=pos_np[i], radius=r_px, color=color)

        # HUD 文本（英文界面，但注释为中文）：WASD 控制与玩家球状态
        gui.text(content="R Reset / W Jump / A Left / S Brake / D Right / Q Quit",
                 pos=(0.02, 0.96), color=0xFFFFFF, font_size=18)
        player_p = pos_np[player_idx]
        player_v = vel.to_numpy()[player_idx]
        gui.text(content=f"player pos=({player_p[0]:.2f},{player_p[1]:.2f})  vel=({player_v[0]:.2f},{player_v[1]:.2f})",
                 pos=(0.02, 0.92), color=0xDDEEFF, font_size=16)

        gui.show()


if __name__ == "__main__":
    main()
