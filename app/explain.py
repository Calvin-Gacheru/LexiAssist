
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def gradcam_overlay(model, x, rgb_img, target_class=1):
    # x: (1, 3, H, W) normalised tensor; rgb_img: float array (H, W, 3) in [0, 1]
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(target_class)])[0]
    return show_cam_on_image(rgb_img, heat, use_rgb=True)
