import taichi as ti  # 导入 Taichi 数值计算与并行框架

# 初始化 Taichi 运行后端：优先使用 GPU（如你的机器/驱动不支持，可改为 ti.cpu）
ti.init(arch=ti.gpu)

# 基础分辨率；最终画布大小为 (2n, n)
n = 320
# 声明一个标量场（二维网格），用于存储每个像素的灰度值
# dtype=float 会被 Taichi 视为 ti.f32（32 位浮点）
pixels = ti.field(dtype=float, shape=(n * 2, n))

@ti.func  # Taichi 函数：可被 kernel 调用，通常内联，无独立并行调度
def complex_sqr(z):
    # 复数平方：若 z = x + i y，则 z^2 = (x^2 - y^2) + i(2xy)
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])

@ti.kernel  # Taichi kernel：在后端（GPU/CPU）并行执行的核函数
def paint(t: float):
    # 并行遍历所有像素坐标 (i, j)，该循环由 Taichi 自动并行化
    for i, j in pixels:  # Parallelized over all pixels
        # Julia 集参数 c：实部固定 -0.8，虚部随时间 t 在 [-0.2, 0.2] 内摆动
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        # 将像素坐标 (i, j) 映射到复平面坐标系
        # i ∈ [0, 2n) -> i/n ∈ [0, 2) -> i/n - 1 ∈ [-1, 1)
        # j ∈ [0,  n) -> j/n ∈ [0, 1) -> j/n - 0.5 ∈ [-0.5, 0.5)
        # 再整体乘以 2，得到约 [-2, 2) × [-1, 1) 的视野范围
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        # 迭代计数器，用于逃逸时间（escape time）着色
        iterations = 0
        # 逃逸条件：|z| < 20 且迭代次数 < 50 时继续迭代
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c  # 迭代公式：z ← z^2 + c
            iterations += 1         # 记录迭代次数
        # 根据迭代次数映射灰度：迭代越多（不易发散），像素越亮
        # 1 - iterations * 0.02 ≈ 1 - (iterations / 50)
        pixels[i, j] = 1 - iterations * 0.02


# 创建 GUI 窗口，标题为 "Julia Set"，分辨率与像素场一致
gui = ti.GUI("Julia Set", res=(n * 2, n))

# 主渲染循环：持续更新参数 t 并刷新画面，形成动画
for i in range(1000000):
    paint(i * 0.03)       # 随时间推进参数 t（控制 c 的虚部），渲染一帧
    gui.set_image(pixels)  # 将像素场传给 GUI 作为图像显示
    gui.show()             # 显示当前帧（可传文件名以保存序列帧）
