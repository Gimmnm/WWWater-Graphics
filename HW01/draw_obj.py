import os
import colorsys
import numpy as np
import tqdm

# 兼容新旧 util 接口（优先使用新命名）
from util import save_np_array_to_img, ndc_to_pixel, pixel_to_ndc, save_images_to_video

def get_vertices_from_obj(obj_path: str) -> np.ndarray:
    """
    从 .obj 文件中提取顶点坐标（以 'v ' 开头的行）。

    参数:
        obj_path: .obj 文件路径

    返回:
        (N, 3) 的 float32 数组，每行是一个顶点的 (x, y, z)
    """
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ 文件不存在: {obj_path}")

    vertices = []

    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):  # 顶点行以 'v ' 开头
                parts = line.split()
                # 跳过 'v'，取接下来的三个浮点数作为 x, y, z
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])

    return np.array(vertices, dtype=np.float32)


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    """
    将顶点居中并按最大半径归一化到单位尺度，提升数值稳定性。

    参数:
        vertices: (N, 3)

    返回:
        归一化后的 (N, 3)
    """
    if vertices.size == 0:
        raise ValueError("OBJ 文件未解析到任何顶点（v 行为空）。")
    center = vertices.mean(axis=0, keepdims=True)
    v = vertices - center
    radii = np.linalg.norm(v, axis=1)
    scale = float(np.max(radii))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    return (v / scale).astype(np.float32)


def rotate_vertices_around_z(vertices: np.ndarray, angle: float) -> np.ndarray:
    """
    围绕 z 轴旋转顶点。

    参数:
        vertices: (N, 3)
        angle: 旋转弧度（rad）
    """
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c,  s, 0.0],
                  [-s, c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return vertices @ R

if __name__ == "__main__":
    # 基于脚本位置构造稳定路径，避免相对路径受运行目录影响
    # 可在下方三者中任选其一
    obj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "white_oak.obj"))
    # obj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "teapot.obj"))
    # obj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "bunny.obj"))
    raw_vertices = get_vertices_from_obj(obj_path)
    raw_vertices = normalize_vertices(raw_vertices)

    N = 100
    img_list = []
    for frame_idx in tqdm.trange(N+1,desc="Generating images"):
        alpha = frame_idx/N
        angle = alpha * 2 * np.pi

        vertices = rotate_vertices_around_z(raw_vertices, angle)
        
        h = 800
        w = 800
        image = np.zeros((h, w, 3), dtype=np.float32)

        # (x,y,z) --> 2D (x,y) 并映射到像素坐标，同时着色
        for vi in range(vertices.shape[0]):
            x, y, z = vertices[vi]
            pixel_x, pixel_y = ndc_to_pixel(x, y, h, w)
            if pixel_x < 0 or pixel_x >= w or pixel_y < 0 or pixel_y >= h:
                continue

            # 动态颜色：三通道正弦流光，随时间与深度变化
            t = alpha * 2*np.pi
            r = 0.5 + 0.5*np.sin(t + 0.0 + 3.0*z)
            g = 0.5 + 0.5*np.sin(t + 2.1 + 2.5*z)
            b = 0.5 + 0.5*np.sin(t + 4.2 + 2.0*z)
            color = np.array([r, g, b], dtype=np.float32)
            # 同一像素被多次命中时，取最大值避免覆盖变灰
            image[pixel_y, pixel_x, :] = np.maximum(image[pixel_y, pixel_x, :], color)

        img_list.append(image)

    save_images_to_video(img_list, 'test.mp4', 25)
