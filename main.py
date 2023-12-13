
import rangeROIs
from mobile_sam import sam_model_registry, SamPredictor

if __name__ == '__main__':
    model_type = "vit_t"
    sam_checkpoint = "mobile_sam.pt"
    device = "cuda"

    root = ""
    save_path = "paper_test/"
    image_names = ["img_", "img_single_"]
    img_len = 73
    roi_len = 1

    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.eval()
    predictor = SamPredictor(mobile_sam)

    rangeROIs.main(predictor, root, save_path, "img_single_", img_len, roi_len)

    print('PyCharm')


