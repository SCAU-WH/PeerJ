import inspect
import os
import random
from matplotlib import pyplot as plt
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from timeit import default_timer as timer


current_time = datetime.now().strftime("%Y%m%d_%H%M%S")


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def Clahe(gray_image):
    gray_image = cv2.convertScaleAbs(gray_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(gray_image)
    return img_clahe


def retinex_multiscale(gray_image, sigma_list):
    result = np.zeros_like(gray_image, dtype=np.float32)
    for sigma in sigma_list:
        blur_image = cv2.GaussianBlur(gray_image, (0, 0), sigma)
        reflectance = np.log1p(gray_image) - np.log1p(blur_image)
        result += reflectance

    result /= len(sigma_list)
    result = np.expm1(result)
    img_retinex = (result * 255).astype(np.uint8)
    return img_retinex


def get_variable_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    for var_name, var_val in callers_local_vars:
        if var_val is var:
            return var_name
    return None


def extract_filename(image_path):
    # Split the image path using the directory separator '/'
    filename = image_path.split(os.path.sep)[-1]
    return filename


def fill(binary_img, thresh):
    filled_img = binary_img.copy()
    inverted_binary = cv2.bitwise_not(filled_img)
    contours, _ = cv2.findContours(inverted_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area <= thresh:
            cv2.drawContours(filled_img, [contour], -1, (0, 0, 0), cv2.FILLED)
    return filled_img


def closing(binary_img, kernel_size=11):
    # 定义一个闭运算的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # 使用闭运算填充缺失的轮廓中心
    closing_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    return closing_img


def process_image_2(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    B, G, R = cv2.split(image)
    mask = (R > G) & (R > B)
    R = R * mask
    G = G * mask
    B = B * mask
    new_img = cv2.merge([B, G, R])
    new_gray_image = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    new_gray_image = cv2.GaussianBlur(new_gray_image, (5, 5), 2)
    ret4, threshold_rgb = cv2.threshold(new_gray_image, 0, 255, cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold_rgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_contour_area = 0
    colored_image = np.ones_like(image) * 255
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        total_contour_area += contour_area
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # cv2.drawContours(colored_image, [contour], -1, color, -1)

    show_list = [image, threshold_rgb]

    return show_list, colored_image, threshold_rgb, total_contour_area


def process_image(image):
    B, G, R = cv2.split(image)
    mask = (R > G) & (R > B)
    R = R * mask
    G = G * mask
    B = B * mask
    new_img = cv2.merge([B, G, R])

    image_float = new_img.astype(np.float32) / 255.0
    sigma_list = [15, 80, 250]
    enhanced_image_2 = retinex_multiscale(image_float, sigma_list)
    _, _, enhanced_gray_image_2 = cv2.split(enhanced_image_2)
    ret6, img_retinex_rgb = cv2.threshold(enhanced_gray_image_2, 0, 255, cv2.THRESH_OTSU)

    retinex_rgb_fill = fill(img_retinex_rgb, 90000)
    rt_rgb_fill_close = closing(retinex_rgb_fill, 3)

    inverted_image = cv2.bitwise_not(rt_rgb_fill_close)
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_contour_area = 0
    colored_image = np.ones_like(image) * 255
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        total_contour_area += contour_area
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # cv2.drawContours(colored_image, [contour], -1, color, -1)

    show_list = [image, rt_rgb_fill_close]

    return show_list, colored_image, img_retinex_rgb, total_contour_area


def plt_save(n, m, show_list, save_path, image_name):
    show_x = 3
    num_images = len(show_list)
    show_y = num_images // show_x
    if num_images % show_x != 0:
        show_y += 1

    if num_images == 0:
        print("No images to show.")
        return

    for i, image in enumerate(show_list):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.suptitle(image_name + str(n) + "_" + str(m), fontsize=15)
        plt.subplot(show_y, show_x, i+1)
        plt.imshow(image)

        iname = get_variable_name(show_list[i])
        if iname:
            plt.title(f"{i + 1}: {iname}")
        else:
            plt.title(f"{i + 1}")
        plt.xticks([]), plt.yticks([])

    new_folder_path = save_path + image_name + "plt"
    os.makedirs(new_folder_path, exist_ok=True)
    filename = f"{new_folder_path}/leaf_{n}_{m}.png"
    print("save: ", filename)
    plt.savefig(filename)
    # plt.show()
    show_list.clear()
    plt.close()


results = []
def main(root, save_path, image_name, img_len, roi_len, image_list):
    for i in image_list:
        for j in range(roi_len):
            image_path = f"{save_path}/{image_name}{i}/{image_name}{i}_{j}.jpg"
            image = cv2.imread(image_path)

            image_path_2 = f"{save_path}/{image_name}{i}/{image_name}_leaf.jpg"
            image_2 = cv2.imread(image_path_2)

            image_path_3 = f"{save_path}/{image_name}{i}/{image_name}_mask.jpg"
            image_3 = cv2.imread(image_path_3)
            img_gray = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)
            ret, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
            kernel = np.ones((13, 13), np.uint8)
            img_bin = cv2.erode(img_bin, kernel, iterations=1)
            contours, _ = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            con_area = cv2.contourArea(contours[0])

            leaf_start_time = timer()
            show_list_2, colored_image_2, rt_rgb_fill_close_2, leaf_area = process_image_2(image_2)
            leaf_end_time = timer()
            leaf_time = leaf_end_time - leaf_start_time

            droplet_start_time = timer()
            show_list, colored_image, rt_rgb_fill_close, droplet_area = process_image(image)
            droplet_end_time = timer()
            droplet_time = droplet_end_time - droplet_start_time

            droplet_area_rate = droplet_area / (300 * 300)
            leaf_area_rate = leaf_area / con_area

            print("droplet_area rate:", droplet_area_rate)
            print("leaf_area rate:", leaf_area_rate)

            results.append([image_name+str(i), droplet_area, leaf_area, droplet_area_rate * 100, leaf_area_rate * 100, droplet_time, leaf_time])



