import numpy as np
import imageio


def save_np_array_to_img(image, path):
    '''
    image: np.array, shape: (h, w, 3), value is in [0,1], RGB
    '''
    image = (image * 255).astype(np.uint8)
    imageio.imwrite(path, image)


def ndc2pixel(ndc_x, ndc_y, h, w):
    '''
    [-1,1] x [-1,1] -> [0,w] x [0,h]
    '''
    pixel_x = (ndc_x + 1) * w / 2
    pixel_y = (ndc_y + 1) * h / 2
    pixel_y = h-1-pixel_y
    return int(pixel_x), int(pixel_y)


def pixel2ndc(pixel_x, pixel_y, h, w):
    '''
    [0,w] x [0,h] -> [-1,1] x [-1,1]
    '''
    ndc_x = (pixel_x / w - 0.5) * 2
    ndc_y = (pixel_y / h - 0.5) * 2
    return ndc_x, ndc_y


def save_imgs_to_video(img_list, save_video_path, fps):
    '''
    img_list: list of np.array, shape: (h, w, 3), value is in [0,1], RGB
    save_video_path: str, video save path
    fps: int, video fps
    '''
    img_list = [(img * 255).astype(np.uint8) for img in img_list]
    imageio.mimsave(save_video_path, img_list, fps=fps)


if __name__ == "__main__":
    h = 320
    w = 320

    img_list = []
    N = 100
    for i in range(N+1):
        alpha = i/N
        xx = int(alpha * (w-1)) # 0,1,2,...,w-1

        image = np.zeros((h, w, 3), dtype=np.float32)
        image[:xx, :, 0] = 1
        image[:xx, :, 1] = 1
        image[:xx, :, 2] = 1
        img_list.append(image)
    save_imgs_to_video(img_list, "test.mp4", 25)
