import os
import numpy as np
from util import save_np_array_to_img, ndc2pixel, pixel2ndc,save_imgs_to_video
import tqdm

def get_vertices_from_obj(obj_path):
    """
    从 .obj 文件中提取顶点坐标。

    参数:
        obj_path (str): .obj 文件路径

    返回:
        np.ndarray: 形状为 (N, 3) 的数组，每行是一个顶点的 (x, y, z) 坐标
    """
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

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


def normalize_vertices(vertices):
    '''
    vertices: (N, 3)
    '''
    center = vertices.mean(axis=0)
    vertices = vertices - vertices.mean(axis=0)
    vertices = vertices / vertices.max()
    return vertices


def rotate_vertices_around_z(vertices, angle):
    '''
    vertices: (N, 3)
    angle: float
    '''
    vertices = vertices @ np.array([[np.cos(angle), np.sin(angle), 0],
                                   [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return vertices

if __name__ == "__main__":
    # obj_path = '../data/teapot.obj'
    # obj_path = '../data/bunny.obj'
    obj_path = '../data/white_oak.obj'
    raw_vertices = get_vertices_from_obj(obj_path)
    raw_vertices = normalize_vertices(raw_vertices)

    N = 100
    img_list = []
    for i in tqdm.trange(N+1,desc="Generating images"):
        alpha = i/N
        angle = alpha * 2 * np.pi

        vertices = rotate_vertices_around_z(raw_vertices, angle)
        
        h = 800
        w = 800
        image = np.zeros((h, w, 3), dtype=np.float32)

        # (x,y,z) --> 2k (x,y)
        for i in range(vertices.shape[0]):
            x, y, z = vertices[i]
            pixel_x, pixel_y = ndc2pixel(x, y, h, w)
            if pixel_x < 0 or pixel_x >= w or pixel_y < 0 or pixel_y >= h:
                continue

            t = alpha * 2*np.pi
            r = 0.5 + 0.5*np.sin(t + 0.0 + 3.0*z)
            g = 0.5 + 0.5*np.sin(t + 2.1 + 2.5*z)
            b = 0.5 + 0.5*np.sin(t + 4.2 + 2.0*z)
            color = np.array([r, g, b], dtype=np.float32)
            image[pixel_y, pixel_x, :] = np.maximum(image[pixel_y, pixel_x, :], color)

        img_list.append(image)

    save_imgs_to_video(img_list, 'test.mp4', 25)
