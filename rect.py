import os
import largestinteriorrectangle as lir
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from datetime import datetime
import pandas as pd
from timeit import default_timer as timer

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


def maximum_internal_rectangle(mask, scale_percent, path):
    resized = mask.copy()

    # 求最小内接矩形
    img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((13, 13), np.uint8)
    img_bin = cv2.erode(img_bin, kernel, iterations=1)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    con_area = cv2.contourArea(contours[0])

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
                # 如果要在灰度图img_gray上画矩形，请用黑色画（0,0,0）
                cv2.rectangle(resized, (x1, y1), (x2, y2), (255, 0, 0), 3)
                # 在mask上画最小内接矩形
                # cv2.imshow("rec", resized)
                # cv2.imwrite(path, resized)
                # cv2.waitKey(0)

            else:
                print("No rectangle fitting into the area")

        else:
            print("No rectangle found")

    else:
        print("No contours found.")



    return resized, x1, y1, x2, y2


def maximum_internal_rectangle_2(mask, path):
    # scale_percent = 9

    resized = mask.copy()
    # 求最小内接矩形
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


    return pt1, pt2, con_area


def get_masks(predictor, image, image_name, input_point, input_label, n):
    predictor.set_image(image)


    masks, scores, logits = predictor.predict(input_point,
                                              point_labels=input_label,
                                              multimask_output=False, )

    return masks, scores


results = []
def main(predictor, root, save_path, image_name, img_len, roi_len):

    for i in range(1, img_len):
        image_path = root + image_name + str(i) + ".jpg"
        image = cv2.imread(image_path)

        if image is not None:
            print("!!!!!!!!!!!!!!!!!", image_path)
            pass
        else:
            print("图像读取失败", image_path)
            continue

        input_point = np.array([[1600, 1800]])
        input_label = np.array([i for i in range(1, len(input_point) + 1)])
        masks, scores = get_masks(predictor, image, image_name, input_point, input_label, i)

        mask = ~masks[0] + 255
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        res = cv2.bitwise_and(image, mask)

        path_rect_1 = "rect/1/"+image_name+str(i)+".jpg"
        path_rect_2 = "rect/2/" + image_name + str(i) + ".jpg"

        start_time = timer()
        scale_percent = 9
        src, x1, y1, x2, y2 = maximum_internal_rectangle(mask, scale_percent, path_rect_1)
        end_time = timer()
        time = end_time - start_time

        start_time_2 = timer()

        pt1, pt2, con_area = maximum_internal_rectangle_2(mask, path_rect_2)
        end_time_2 = timer()
        time_2 = end_time_2 - start_time_2


        area_1 = (abs(x2 - x1) * abs(y2 - y1)) / con_area
        area_2 = (abs(pt2[1] - pt1[1]) * abs(pt2[0] - pt1[0])) / con_area

        print(area_1, area_2)

        results.append([image_name+str(i),  area_1,  area_2, time, time_2])

        # print("已处理： " + image_path)
    total_area_1 = 0
    total_area_2 = 0
    total_time = 0
    total_time_2 = 0
    num_sublists = len(results)

    # 遍历 results 列表，累加各个值
    for sublist in results:
        total_area_1 += sublist[1]
        total_area_2 += sublist[2]
        total_time += sublist[3]
        total_time_2 += sublist[4]

    # 计算平均值
    avg_area_1 = total_area_1 / num_sublists
    avg_area_2 = total_area_2 / num_sublists
    avg_time = total_time / num_sublists
    avg_time_2 = total_time_2 / num_sublists

    # 打印平均值
    print("Average area 1:", avg_area_1)
    print("Average area 2:", avg_area_2)
    print("Average time:", avg_time)
    print("Average time 2:", avg_time_2)
    print(num_sublists)


if __name__ == "__main__":
    model_type = "vit_t"
    sam_checkpoint = "mobile_sam.pt"
    device = "cuda"

    root = "paper_rect_image/"
    save_path = "outputs_rect_2/"
    image_names = ["img_", "img_single_"]
    img_len = 73
    roi_len = 5

    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    predictor = SamPredictor(mobile_sam)

    for image_name in image_names:

        main(predictor, root, save_path, image_name, img_len, roi_len)

    df = pd.DataFrame(results, columns=['image_name',  'area_1',  'area_2', 'time', 'time_2'])

    excel_path = 'excel_2/area_resize_results.xlsx'
    df.to_excel(excel_path, index=False)
    print("Results saved to:", excel_path)










