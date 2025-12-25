# helpers.py -- all functions and classes taken from the notebooks and necessary for the app

from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import clip_boxes_to_image
from torchvision.models import resnet34
import segmentation_models_pytorch as smp
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUNDLES_DIR = Path(__file__).resolve().parent.parent / "bundles"

# helper for detector (no normalization bc already implicit from pckg)
PREPROCESS_DET = T.ToTensor()

# preprocess
def pad_to_square(img):
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    out = Image.new("RGB", (s, s), (0, 0, 0))
    out.paste(img, ((s - w)//2, (s - h)//2))
    return out

CLS_TRANSFORM = T.Compose([
    T.Lambda(pad_to_square),
    T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# box detector
def detect_person_boxes(pil_img, model, threshold=0.5, max_boxes=5):
    img_tensor = PREPROCESS_DET(pil_img).to(device)
    with torch.no_grad():
        out = model([img_tensor])[0]

    keep = (out["labels"] == 1) & (out["scores"] >= threshold)
    if keep.sum() == 0:
        raise RuntimeError("No humans detected.")

    boxes, scores = out["boxes"][keep], out["scores"][keep]
    h, w = pil_img.size[1], pil_img.size[0]
    boxes = clip_boxes_to_image(boxes, (h, w))
    scores, order = scores.sort(descending=True)
    boxes = boxes[order][:max_boxes]
    return boxes.cpu()

# Model leading
# Detector (COCO)
det_model = fasterrcnn_resnet50_fpn(pretrained=True)
det_model.roi_heads.nms_thresh = 0.4
det_model.eval().to(device)

# Segmentation bundle
SEG_BUNDLE = torch.load(BUNDLES_DIR / "dress_segmentation_bundle_part_2.pth",
                        map_location=device)
seg_model = smp.Unet(
    encoder_name=SEG_BUNDLE["encoder"],
    encoder_weights="imagenet",
    classes=1,
    activation=None
).to(device)
seg_model.load_state_dict(SEG_BUNDLE["model_state"])
seg_model.eval()
SEG_MEAN, SEG_STD = SEG_BUNDLE["mean"], SEG_BUNDLE["std"]
SEG_TRANSFORM = T.Compose([
    T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(SEG_MEAN, SEG_STD),
])

# Classifier (10 classes)
cls_model = resnet34(weights=None)
cls_model.fc = torch.nn.Linear(cls_model.fc.in_features, 10)
ckpt = torch.load(BUNDLES_DIR / "best_resnet.pt", map_location=device)
state = ckpt["model_state"] if "model_state" in ckpt else ckpt
state = { (k.replace("backbone.", "") if k.startswith("backbone.") else k): v
          for k, v in state.items() }
missing, unexpected = cls_model.load_state_dict(state, strict=False)
cls_model.eval().to(device)

label_to_idx = {
    'casual_dress': 0, 'denim_dress': 1, 'evening_dress': 2,
    'jersey_dress': 3, 'knitted_dress': 4, 'maxi_dress': 5,
    'occasion_dress': 6, 'shift_dress': 7, 'shirt_dress': 8,
    'work_dress': 9
}
IDX_TO_CLASS = {v: k for k, v in label_to_idx.items()}

# Segmentation inference
def segment_crop(crop_pil, thr=0.5):
    # Resize+normalize like training, then resize mask back to crop size
    x = SEG_TRANSFORM(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = seg_model(x)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    # back to original crop size
    prob_img = Image.fromarray((prob * 255).astype(np.uint8)).resize(
        crop_pil.size, Image.BILINEAR
    )
    prob = np.array(prob_img, dtype=np.float32) / 255.0
    mask = (prob >= thr).astype(np.uint8)
    return mask

# Full pipeline
def process_image(img_pil, process_all=True):
    results = []
    try:
        boxes = detect_person_boxes(img_pil, det_model)
    except RuntimeError as e:
        print(str(e))
        return results

    target_layers = [cls_model.layer4[-1]]
    with GradCAM(model=cls_model, target_layers=target_layers) as cam:
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.tolist())
            crop = img_pil.crop((x1, y1, x2, y2))

            mask = segment_crop(crop, thr=0.5)
            if mask.sum() == 0:
                results.append({"box": b, "msg": "No dress detected.",
                                "mask": None, "label": None, "heatmap": None})
                if not process_all:
                    break
                continue

            crop_np = np.array(crop).copy()
            crop_np[mask == 0] = 0
            masked = Image.fromarray(crop_np)

            x_cls = CLS_TRANSFORM(masked).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = cls_model(x_cls)
            pred_idx = int(torch.argmax(logits, dim=1).item())
            pred_label = IDX_TO_CLASS[pred_idx]

            grayscale = cam(input_tensor=x_cls,
                            targets=[ClassifierOutputTarget(pred_idx)])[0]
            rgb = np.array(masked.resize((224, 224))).astype(np.float32) / 255.0
            heat = show_cam_on_image(rgb, grayscale, use_rgb=True)

            results.append({"box": b, "label": pred_label,
                            "mask": mask, "heatmap": heat})
            if not process_all:
                break
    return results