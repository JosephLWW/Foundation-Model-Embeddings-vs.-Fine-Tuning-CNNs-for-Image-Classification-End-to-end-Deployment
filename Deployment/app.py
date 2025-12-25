import os
import io
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch import nn
import numpy as np
from PIL import Image, ImageDraw

import torchvision
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import clip_boxes_to_image

import segmentation_models_pytorch as smp  # segmentation library used in the notebooks
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import gradio as gr

"""
This module implements a Gradio application that mirrors the pipeline used
in the practical assignments.  The functions defined here have been
copied verbatim from the original notebooks (``Assignment 3-1.ipynb``,
``Assignment 3-2.ipynb`` and ``Assignment 3-3.ipynb``) so that their
behaviour is identical to the code you already wrote.  In particular,
we reuse the preprocessing functions and model loading logic exactly as
they appear in the notebooks.

The pipeline works in four stages:

1. **Detection** – Locate up to five people in the image using a
   pre‐trained Faster R‑CNN.  Only persons are kept and boxes are
   clipped to the image boundaries.  When no humans are detected the
   original function raises a ``RuntimeError`` with the message
   ``"No humans detected."``; we propagate this exception to the UI.

2. **Segmentation** – For each detected person we crop the image
   according to the bounding box and feed it through a UNet to obtain
   a binary mask of the dress.  The crops are resized to ``256×256``
   and normalised using the mean and standard deviation stored in the
   segmentation bundle.  After inference the mask is resized back to
   the original crop size and thresholded at ``0.5``.

3. **Classification** – The masked crop is padded to a square,
   resized to ``224×224``, converted to a tensor and normalised with
   ImageNet statistics.  A fine–tuned ResNet returns logits over the
   ten dress categories.  The predicted class label is looked up from
   the mapping used in the notebooks.

4. **Grad‑CAM** – A class activation map is produced to visualise
   which areas of the masked crop contribute most to the prediction.
   The heatmap is overlaid on the resized crop.

To deploy the application as a Hugging Face Space, simply upload
``app.py``, ``requirements.txt`` and the weight files used in the
assignments to a new Space repository.  The Space will
automatically install the dependencies and run the Gradio interface.
"""

# ----------------------------------------------------------------------------
# Global configuration and device selection
# ----------------------------------------------------------------------------
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The detection preprocessing used in the original notebook consisted only
# of converting the PIL image to a tensor.  Faster R‑CNN applies its own
# normalisation internally, so we do not normalise here.  See notebook
# ``Assignment 3-1.ipynb`` for the exact definition.
PREPROCESS_DET: T.Compose = T.Compose([
    T.ToTensor(),
])

# Load the person detector.  The model is configured with a lower NMS
# threshold in the original assignment; however this is not strictly
# necessary for the inference pipeline.  The model weights are the
# standard ``pretrained=True`` weights from TorchVision.
def load_detector() -> torchvision.models.detection.faster_rcnn.FasterRCNN:
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.roi_heads.nms_thresh = 0.4  # as in the notebook
    model.eval()
    model.to(device)
    return model


def detect_person_boxes(
    pil_img: Image.Image,
    model: torchvision.models.detection.faster_rcnn.FasterRCNN,
    threshold: float = 0.5,
    max_boxes: int = 5,
) -> torch.Tensor:
    """
    Detect up to ``max_boxes`` persons in ``pil_img`` using ``model``.

    This function is copied exactly from the helpers cell in
    ``Assignment 3-3.ipynb``.  It accepts a PIL image and returns a
    tensor of shape ``(N, 4)`` containing the bounding boxes for the
    top‐``N`` detections where ``N <= max_boxes``.  If no persons are
    detected it raises a ``RuntimeError`` with the message
    ``"No humans detected."``.
    """
    # Convert the PIL image to tensor (no normalisation)
    img_tensor = PREPROCESS_DET(pil_img).to(device)
    with torch.no_grad():
        out = model([img_tensor])[0]

    keep = (out["labels"] == 1) & (out["scores"] >= threshold)
    if keep.sum() == 0:
        # This mirrors the behaviour in the original notebook.
        raise RuntimeError("No humans detected.")

    boxes, scores = out["boxes"][keep], out["scores"][keep]

    # Clip boxes to image boundaries (height, width)
    h, w = pil_img.size[1], pil_img.size[0]
    boxes = clip_boxes_to_image(boxes, (h, w))
    scores, order = scores.sort(descending=True)
    boxes = boxes[order][:max_boxes]
    return boxes.cpu()


def draw_boxes(pil_img: Image.Image, boxes: torch.Tensor) -> Image.Image:
    """Return a copy of ``pil_img`` with the bounding boxes drawn in green."""
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.tolist())
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
    return img


# ----------------------------------------------------------------------------
# Segmentation model and utilities
# ----------------------------------------------------------------------------

# Path to the segmentation bundle.  Adjust this if your files are located
# elsewhere.  The bundle contains the encoder name, model weights, and
# the mean/standard deviation used for normalisation during training.
SEG_BUNDLE_PATH: str = os.getenv(
    "SEG_BUNDLE_PATH",
    "Practical Deep Learning/Assignment 3/CACHE/dress_segmentation_bundle_part_2.pth",
)


def load_segmentation_model() -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """
    Load the UNet segmentation model and associated normalisation stats.

    Returns a tuple ``(model, mean, std)`` where ``model`` is on the
    correct device and ready for inference, and ``mean`` and ``std``
    are one‐dimensional tensors used by the segmentation transform.
    """
    bundle = torch.load(SEG_BUNDLE_PATH, map_location=device)
    mean = bundle["mean"]
    std = bundle["std"]
    # Create the UNet with the same encoder as used in training
    seg_model = smp.Unet(
        encoder_name=bundle["encoder"],
        encoder_weights="imagenet",
        classes=1,
        activation=None,
    )
    seg_model.load_state_dict(bundle["model_state"])
    seg_model.eval()
    seg_model.to(device)
    return seg_model, mean, std


def get_segmentation_transform(mean: torch.Tensor, std: torch.Tensor) -> T.Compose:
    """Return the transformation pipeline used before feeding crops to the UNet."""
    return T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean.tolist(), std.tolist()),
    ])


def segment_crop(
    crop_pil: Image.Image,
    seg_model: nn.Module,
    seg_transform: T.Compose,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Produce a binary mask for the dress in ``crop_pil``.

    The crop is resized and normalised using ``seg_transform``.  The
    UNet returns a logit map; applying a sigmoid followed by a
    threshold yields a binary mask.  Finally, the mask is resized back
    to the original crop size.

    Returns a NumPy array of shape ``(H, W)`` with dtype ``uint8``.
    """
    x_in = seg_transform(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = seg_model(x_in)
    # Remove batch and channel dimensions
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    # Resize the probability map back to the crop size
    prob_img = Image.fromarray((prob * 255).astype(np.uint8)).resize(
        crop_pil.size, Image.BILINEAR
    )
    prob = np.array(prob_img, dtype=np.float32) / 255.0
    mask = (prob >= threshold).astype(np.uint8)
    return mask


# ----------------------------------------------------------------------------
# Classification model and utilities
# ----------------------------------------------------------------------------

# Path to the classification checkpoint (fine–tuned ResNet).  Adjust this as
# required to point at the correct file in your repository.
CLS_CKPT_PATH: str = os.getenv(
    "CLS_CKPT_PATH",
    "Practical Deep Learning/Assignment 2/best_resnet.pt",
)


def load_classifier(num_classes: int = 10) -> nn.Module:
    """Load the fine–tuned ResNet used for dress classification."""
    model = torchvision.models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    ckpt = torch.load(CLS_CKPT_PATH, map_location=device)
    # The checkpoint may store the weights under different keys; strip
    # any leading 'backbone.' prefixes if necessary.
    state = ckpt.get("model_state", ckpt)
    new_state = {}
    for k, v in state.items():
        if k.startswith("backbone."):
            new_state[k.replace("backbone.", "")] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=False)
    model.eval()
    model.to(device)
    return model


def pad_to_square(pil_img: Image.Image, fill: int = 0) -> Image.Image:
    """
    Pad a rectangular image to a square by adding borders of colour
    ``fill``.  This helper is copied from the classification notebook.
    """
    w, h = pil_img.size
    if w == h:
        return pil_img
    s = max(w, h)
    new_img = Image.new("RGB", (s, s), (fill, fill, fill))
    new_img.paste(pil_img, ((s - w) // 2, (s - h) // 2))
    return new_img


def get_classification_transform() -> T.Compose:
    """Return the preprocessing pipeline used before feeding the masked crop to the classifier."""
    return T.Compose([
        T.Lambda(lambda img: pad_to_square(img, fill=0)),
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


label_to_idx = {
    'casual_dress': 0,
    'denim_dress': 1,
    'evening_dress': 2,
    'jersey_dress': 3,
    'knitted_dress': 4,
    'maxi_dress': 5,
    'occasion_dress': 6,
    'shift_dress': 7,
    'shirt_dress': 8,
    'work_dress': 9,
}
IDX_TO_CLASS = {v: k for k, v in label_to_idx.items()}


def classify_masked_crop(
    crop_pil: Image.Image,
    mask_np: np.ndarray,
    cls_model: nn.Module,
    cls_transform: T.Compose,
) -> Tuple[Image.Image, str, int, torch.Tensor]:
    """
    Given a crop and its binary mask, mask out the background, run the
    classifier and return the masked image, predicted label, label index
    and the tensor that was input to the classifier (for Grad‑CAM).
    """
    # Apply the mask to the crop: set background pixels to zero
    arr = np.array(crop_pil).copy()
    arr[mask_np == 0] = 0
    masked = Image.fromarray(arr)
    x_in = cls_transform(masked).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = cls_model(x_in)
    pred_idx = int(logits.argmax(dim=1).item())
    label = IDX_TO_CLASS[pred_idx]
    return masked, label, pred_idx, x_in


# ----------------------------------------------------------------------------
# Pipeline composition
# ----------------------------------------------------------------------------

def process_image(
    pil_img: Image.Image,
    det_model: torchvision.models.detection.faster_rcnn.FasterRCNN,
    seg_model: nn.Module,
    seg_transform: T.Compose,
    cls_model: nn.Module,
    cls_transform: T.Compose,
    process_all: bool = False,
) -> Tuple[Optional[Image.Image], List[Tuple[str, Image.Image]], str]:
    """
    Run the full pipeline on ``pil_img``.  Returns a tuple

      ``(boxed_image, gallery, labels_str)``

    where ``boxed_image`` is the original image with boxes drawn (or
    ``None`` if no humans were detected), ``gallery`` is a list of
    ``(title, image)`` pairs containing the masked crop and the
    corresponding Grad‑CAM visualisation, and ``labels_str`` is a
    comma–separated string of predicted labels.

    If ``process_all`` is ``False`` the pipeline stops after the first
    successful detection; if ``True`` it processes every detected
    person.
    """
    try:
        boxes = detect_person_boxes(pil_img, det_model, threshold=0.5, max_boxes=5)
    except RuntimeError as e:
        # If no humans are detected, return the original image and message
        return pil_img, [], str(e)
    if boxes.numel() == 0:
        return pil_img, [], "No humans detected."

    out_images: List[Tuple[str, Image.Image]] = []
    labels: List[str] = []

    # Set up Grad‑CAM to inspect the last convolutional layer of the classifier
    target_layers = [cls_model.layer4[-1]]
    with GradCAM(model=cls_model, target_layers=target_layers) as cam:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.tolist())
            crop = pil_img.crop((x1, y1, x2, y2))
            mask = segment_crop(crop, seg_model, seg_transform, threshold=0.5)
            # If the mask is empty, skip or note
            if mask.sum() == 0:
                out_images.append(("No dress detected", crop))
                labels.append("No dress detected")
                if not process_all:
                    break
                continue
            masked, label, pred_idx, x_in = classify_masked_crop(crop, mask, cls_model, cls_transform)
            # Generate Grad‑CAM heatmap
            grayscale_cam = cam(
                input_tensor=x_in,
                targets=[ClassifierOutputTarget(pred_idx)],
            )[0]
            rgb = np.array(masked.resize((224, 224))).astype(np.float32) / 255.0
            heatmap = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
            heat_pil = Image.fromarray(heatmap)
            # Append masked image and Grad‑CAM
            out_images.append((f"Masked: {label}", masked))
            out_images.append((f"GradCAM: {label}", heat_pil))
            labels.append(label)
            if not process_all:
                break
    # Draw boxes on the original image for display
    boxed = draw_boxes(pil_img, boxes)
    labels_str = ", ".join(labels)
    return boxed, out_images, labels_str


# ----------------------------------------------------------------------------
# Gradio interface definition
# ----------------------------------------------------------------------------

# Preload models at module import time so that the first call to the
# interface is responsive.  If the weight files are missing the Space
# will raise an exception at startup, which is preferable to a silent
# failure later on.
DET_MODEL = load_detector()
SEG_MODEL, SEG_MEAN, SEG_STD = load_segmentation_model()
SEG_TRANSFORM = get_segmentation_transform(SEG_MEAN, SEG_STD)
CLS_MODEL = load_classifier(num_classes=10)
CLS_TRANSFORM = get_classification_transform()


def infer(image: Image.Image, process_all: bool) -> Tuple[Image.Image, List[Tuple[str, Image.Image]], str]:
    """
    Wrapper function for Gradio.  Accepts a PIL image and a boolean
    flag indicating whether to process all detections.  Returns the
    boxed image, gallery list and labels string as required by
    ``gr.Interface``.
    """
    if image is None:
        return image, [], "No image provided."
    return process_image(
        image,
        det_model=DET_MODEL,
        seg_model=SEG_MODEL,
        seg_transform=SEG_TRANSFORM,
        cls_model=CLS_MODEL,
        cls_transform=CLS_TRANSFORM,
        process_all=process_all,
    )


def create_demo() -> gr.Blocks:
    """
    Create the Gradio UI.  This function is separated from the global
    scope so that it can be called when the module is imported and
    reused if needed.
    """
    with gr.Blocks() as demo:
        gr.Markdown("# Dress Detection, Segmentation, Classification and Grad‑CAM")
        gr.Markdown(
            "Upload a photo containing people wearing dresses.  The pipeline "
            "will detect up to five people, segment the dress, classify it "
            "into one of ten categories and display a Grad‑CAM heatmap."\
        )
        with gr.Row():
            image_input = gr.Image(type="pil", label="Input Image")
            process_all_chk = gr.Checkbox(
                label="Process all detections", value=False
            )
        run_btn = gr.Button("Run")
        with gr.Row():
            boxed_output = gr.Image(label="Image with Bounding Boxes")
            gallery_output = gr.Gallery(
                label="Masked Crop & Grad‑CAM", columns=2, preview=True
            )
        labels_output = gr.Textbox(label="Predicted Labels", lines=1)
        # Connect button to function
        run_btn.click(
            fn=infer,
            inputs=[image_input, process_all_chk],
            outputs=[boxed_output, gallery_output, labels_output],
        )
    return demo


# Create and launch the demo when executed as a script.  When deployed
# on Hugging Face Spaces the ``__main__`` block will be executed.
if __name__ == "__main__":
    demo = create_demo()
    demo.launch()