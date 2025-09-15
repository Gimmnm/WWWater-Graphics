import os
from typing import List

import numpy as np
import imageio

def save_np_array_to_img(image: np.ndarray, path: str) -> None:
    """
    将取值范围为 [0, 1] 的 RGB 图像数组保存为文件。

    参数:
        image: np.ndarray, 形状 (H, W, 3)，RGB，数值范围 [0, 1]
        path: 输出图像路径（支持常见图像格式，如 .png/.jpg）
    """
    image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    imageio.imwrite(path, image)


def ndc_to_pixel(ndc_x: float, ndc_y: float, height: int, width: int) -> tuple[int, int]:
    """
    将标准化设备坐标系 (NDC) 中的点从 [-1, 1]×[-1, 1]
    映射到像素坐标系 [0, W)×[0, H)。

    说明:
        - NDC 原点在中心，x 向右为正，y 向上为正。
        - 像素坐标原点在左上角，x 向右，y 向下。

    参数:
        ndc_x, ndc_y: NDC 坐标，范围通常在 [-1, 1]
        height, width: 图像高和宽（像素）

    返回:
        (pixel_x, pixel_y): 整型像素坐标。
    """
    pixel_x = (ndc_x + 1.0) * width / 2.0
    pixel_y = (ndc_y + 1.0) * height / 2.0
    # 将 y 轴翻转到以左上角为原点的像素坐标系
    pixel_y = height - 1 - pixel_y
    return int(pixel_x), int(pixel_y)


def pixel_to_ndc(pixel_x: float, pixel_y: float, height: int, width: int) -> tuple[float, float]:
    """
    将像素坐标 [0, W)×[0, H) 映射回标准化设备坐标系 [-1, 1]×[-1, 1]。

    参数:
        pixel_x, pixel_y: 像素坐标（左上角为原点）
        height, width: 图像高和宽（像素）

    返回:
        (ndc_x, ndc_y): NDC 坐标。
    """
    ndc_x = (pixel_x / width - 0.5) * 2.0
    ndc_y = (pixel_y / height - 0.5) * 2.0
    return float(ndc_x), float(ndc_y)


def save_images_to_video(images: List[np.ndarray], save_video_path: str, fps: int) -> None:
    """
    将多帧 RGB 图像序列保存为视频或动图（自动根据扩展名选择编码）。

    参数:
        images: 图像序列，列表中的每项为 (H, W, 3) 且数值范围 [0, 1]
        save_video_path: 输出路径，支持如 .mp4/.gif 等扩展名。
        fps: 帧率（每秒帧数）

    """
    imgs_u8 = [(np.clip(img, 0.0, 1.0) * 255).astype(np.uint8) for img in images]
    imageio.mimsave(save_video_path, imgs_u8, fps=fps)



if __name__ == "__main__":
    # 简单自检：生成一段从黑到白的演示动图
    H, W = 320, 320
    frames = []
    num_frames = 60
    for i in range(num_frames + 1):
        t = i / num_frames  # [0, 1]
        xx = int(t * (W - 1))
        img = np.zeros((H, W, 3), dtype=np.float32)
        img[:, :xx, :] = 1.0
        frames.append(img)
    save_images_to_video(frames, "test.mp4", 25)
