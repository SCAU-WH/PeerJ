import os
import largestinteriorrectangle as lir
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from datetime import datetime
import pandas as pd

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")


def is_empty_image(image):
    return image.shape[0] == 0 and image.shape[1] == 0


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


def maximum_internal_rectangle(mask, scale_percent):
    resized = mask.copy()

    img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((13, 13), np.uint8)
    img_bin = cv2.erode(img_bin, kernel, iterations=1)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    x1, x2, y1, y2 = 0, 0, 0, 0
    if len(contours) > 0:
        contour = contours[0].reshape(len(contours[0]), 2)
        rect = []
        for i in range(len(contour)):
            x1, y1 = contour[i]
            for j in range(len(contour)):
                x2, y2 = contour[j]
                area = abs(y2 - y1) * abs(x2 - x1)
                rect.append(((x1, y1), (x2, y2), area))

        all_rect = sorted(rect, key=lambda x: x[2], reverse=True)

        if all_rect:
            best_rect_found = False
            index_rect = 0
            nb_rect = len(all_rect)

            while not best_rect_found and index_rect < nb_rect:

                rect = all_rect[index_rect]
                (x1, y1) = rect[0]
                (x2, y2) = rect[1]

                valid_rect = True

                x = min(x1, x2)
                while x < max(x1, x2) + 1 and valid_rect:
                    if any(resized[y1, x]) == 0 or any(resized[y2, x]) == 0:
                        valid_rect = False
                    x += 1

                y = min(y1, y2)
                while y < max(y1, y2) + 1 and valid_rect:
                    if any(resized[y, x1]) == 0 or any(resized[y, x2]) == 0:
                        valid_rect = False
                    y += 1

                if valid_rect:
                    best_rect_found = True

                index_rect += 1

            if best_rect_found:
                cv2.rectangle(resized, (x1, y1), (x2, y2), (255, 0, 0), 3)
                # cv2.imshow("rec", img)
                # cv2.waitKey(0)

            else:
                print("No rectangle fitting into the area")

        else:
            print("No rectangle found")

    else:
        print("No contours found.")


    return resized, x1, y1, x2, y2


def maximum_internal_rectangle_2(mask, path):
    resized = mask.copy()

    img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((13, 13), np.uint8)
    img_bin = cv2.erode(img_bin, kernel, iterations=1)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0][:, 0, :]
    con_area = cv2.contourArea(contour)

    grid = np.array(img_bin, dtype=bool)

    rectangle = lir.lir(grid, contour)
    cv2.rectangle(resized, lir.pt1(rectangle), lir.pt2(rectangle), (255, 0, 0), 10)
    # cv2.imwrite(path, resized)

    pt1 = lir.pt1(rectangle)
    pt2 = lir.pt2(rectangle)

    y1 = pt1[1]
    y2 = pt2[1]
    x1 = pt1[0]
    x2 = pt2[0]

    return resized, x1, y1, x2, y2


def random_crop_inner_rectangle(img, x1, y1, x2, y2, crop_width, crop_height):
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    max_x = x2 - crop_width // 2
    min_x = x1 + crop_width // 2
    max_y = y2 - crop_height // 2
    min_y = y1 + crop_height // 2

    cropped_img = np.zeros((crop_height, crop_width), np.uint8)

    if min_x < max_x and min_y < max_y:
        crop_x = np.random.randint(min_x, max_x)
        crop_y = np.random.randint(min_y, max_y)

        cropped_img = img[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

    return cropped_img


click_x, click_y = -1, -1
clicked_points = [] 
def mouse_callback(event, x, y, flags, param):
    global click_x, click_y
    if event == cv2.EVENT_LBUTTONDOWN:
        click_x, click_y = x, y

def mouse_click(image):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", mouse_callback)
    global click_x, click_y
    click_x, click_y = -1, -1
    clicked_points.clear()
    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1)
        if click_x != -1 and click_y != -1:
            clicked_points.append((click_x, click_y))
            print(click_x, ",", click_y)

        if key == 27:
            break


def get_masks(predictor, image, image_name, input_point, input_label):
    predictor.set_image(image)

    masks, scores, logits = predictor.predict(input_point,
                                              point_labels=input_label,
                                              multimask_output=False, )

    return masks, scores


def show_mask_in_image(masks, scores, input_point, input_label, image, save_path, image_name):
    for k, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(masks[k], plt.gca())
        for j in range(len(input_point)):
            show_points(input_point[j], input_label[j], plt.gca())
        plt.title(f"Mask {image_name}, Score: {score:.3f}", fontsize=10)
        plt.axis('off')

        plt_name = "marked/point/"
        os.makedirs(plt_name, exist_ok=True)

        plt.close()


def save_leaf(new_folder_path, res, image, image_rect, src, mask, image_name):


    filename6 = f"{new_folder_path}/{image_name}_mask.jpg"



def dispose(image):
    input_point = np.array([[click_x, click_y]])

    input_label = np.array([i for i in range(1, len(input_point) + 1)])
    masks, scores = get_masks(predictor, image, image_name, input_point, input_label)
    mask = ~masks[0] + 255
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
    res = cv2.bitwise_and(image, mask)
    scale_percent = 9
    src, x1, y1, x2, y2 = maximum_internal_rectangle_2(mask, scale_percent)
    new_folder_path = save_path
    os.makedirs(new_folder_path, exist_ok=True)
    image_rect = image.copy()
    cv2.rectangle(image_rect, (x1, y1), (x2, y2), (255, 0, 0), 10)

    save_leaf(new_folder_path, res, image, image_rect, src, mask, image_name)
    show_mask_in_image(masks, scores, input_point, input_label, image, save_path, image_name)
    resluts.append([image_name, click_x, click_y])



image_names = []
resluts = []
def main(predictor, root, save_path, image_name, img_len, roi_len):

    file_path = 'marked/2.xlsx'
    sheet_name = 'Sheet1'
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    sheet_name2 = 'Sheet2'
    data2 = pd.read_excel(file_path, sheet_name=sheet_name2)

    image_names = data['Image Name']
    image_names2 = data2['Image Name']
    # print(len(image_names))
    f = open("marked\\3.txt")

    for image_name in image_names2:
        image_name = '1688125653033'
        print(image_name)
        image_path = "marked\\JPEGImages\\2\\" + str(image_name) + ".jpg"
        image = cv2.imread(image_path)
        mouse_click(image)
        dispose(image)

    for image_name in image_names:
        print(image_name)
        image_path = "marked\\JPEGImages\\1\\" + str(image_name) + ".jpg"
        image = cv2.imread(image_path)
        mouse_click(image)
        dispose(image)

    f.close()
    excel_path = 'marked/point_results_2.xlsx'
    df = pd.DataFrame(resluts, columns=['Image Name', 'click_x', 'click_y'])

    df.to_excel(excel_path, index=False)


























