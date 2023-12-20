import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from timeit import default_timer as timer


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


def leaf_eval(gt_mask, pred_mask):
    assert gt_mask.shape == pred_mask.shape

    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()

    true_mask = (gt_mask > 0).astype(np.uint8) 
    predicted_mask = (pred_mask > 0).astype(np.uint8)

    tp = np.sum(np.logical_and(predicted_mask == 1, true_mask == 1))
    fp = np.sum(np.logical_and(predicted_mask == 1, true_mask == 0))
    fn = np.sum(np.logical_and(predicted_mask == 0, true_mask == 1))
    tn = np.sum(np.logical_and(predicted_mask == 0, true_mask == 0))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    pa = (tp + tn) / (tp + tn + fp + fn)
    iou = tp / (tp + fp + fn)
    # print(iou)
    # iou = intersection / union
    # f1 = 2 * tp / (2 * tp + fp + fn + fp + fn)
    # print(f1)
    f1 = 2 * precision * recall / (precision + recall)
    # dice = 2 * tp / (2 * tp + fp + fn)
    # print(dice)
    dice = (2 * iou) / (1 + iou)

    # print(pa, iou, f1, dice)

    return pa, iou, f1, dice


if __name__ == '__main__':
    directory_path = "marked\\JPEGImages\\1"
    excel_path = 'excel_2/evaluation_results_IOC.xlsx'

    mobile_sam_type = "vit_t"
    mobile_sam_checkpoint = "mobile_sam.pt"
    device = "cuda"

    sam_type = "vit_b"
    sam_checkpoint = "sam_vit_b_01ec64.pth"

    f = open("marked\\3.txt")
    file_path = 'marked/point_results3.xlsx'
    sheet_name = 'Sheet1'
    data = pd.read_excel(file_path, sheet_name=sheet_name)

    image_names = data['Image Name']
    click_x = data['click_x']
    click_y = data['click_y']


    results = []
    for filename in os.listdir(directory_path):
        image_name = os.path.splitext(filename)[0]

        image_path = directory_path + "\\" + image_name + ".jpg"
        gt_image_path = "marked\\SegmentationClassPNG\\" + image_name + ".png"
        image = cv2.imread(image_path)
        gt_image = cv2.imread(gt_image_path)

        _, _, gt_mask = cv2.split(gt_image)
        ret, gt_mask = cv2.threshold(gt_mask, 0, 255, cv2.THRESH_BINARY)

        gt_mask = (gt_mask > 0).astype(np.uint8)

        input_point = np.array([[1800, 1700]])
        print(image_name, input_point)
        for tmp_name in f:
            if tmp_name[:-1] == image_name:
                input_point = np.array([[1600, 1500]])
                print(tmp_name[:-1], input_point)
        for tmp_name in image_names:
            if tmp_name == image_name:
                i = image_names[image_names.values == image_name].index
                input_point = np.array([[int(click_x[i].values), int(click_y[i].values)]])
                print(tmp_name, input_point)

        input_label = np.array([1])

        start_time = timer()
        mobile_sam = sam_model_registry[mobile_sam_type](checkpoint=mobile_sam_checkpoint)
        mobile_sam.to(device=device)
        mobile_sam.eval()
        mobile_sam_predictor = SamPredictor(mobile_sam)
        mobile_sam_predictor.set_image(image)
        mobile_sam_masks, _, _ = mobile_sam_predictor.predict(input_point,
                                                  point_labels=input_label,
                                                  multimask_output=False, )
        end_time = timer()
        time = end_time - start_time

        sam_start_time = timer()
        sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam.eval()
        sam_predictor = SamPredictor(sam)
        sam_predictor.set_image(image)
        sam_masks, scores, _ = sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sam_end_time = timer()
        sam_time = sam_end_time - sam_start_time

        mobile_sam_pred_mask = np.array(mobile_sam_masks[0], dtype=int)
        sam_pred_mask = np.array(sam_masks[1], dtype=int)

        pixel_acc, iou, f1, dice = leaf_eval(gt_mask, mobile_sam_pred_mask)
        sam_pixel_acc, sam_iou, sam_f1, sam_dice = leaf_eval(gt_mask, sam_pred_mask)


        results.append([image_name, pixel_acc, sam_pixel_acc, iou, sam_iou, dice, sam_dice, time, sam_time])

    df = pd.DataFrame(results, columns=['Image Name', 'Pixel Accuracy', 'sam_pixel_acc', 'IoU', 'sam_iou',
                                         'Dice Coefficient', 'sam_dice', 'time', 'sam_time'])

    df.to_excel(excel_path, index=False)
    print("Results saved to:", excel_path)

