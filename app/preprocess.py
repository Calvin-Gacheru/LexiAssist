
import numpy as np
from PIL import Image
from torchvision import transforms as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def border_fill(img):
    arr = np.asarray(img.convert("L"))
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    return int(np.median(border))

class PadToSquare:
    # pad to a square canvas using the background colour, so the aspect ratio is kept
    def __call__(self, img):
        img = img.convert("L")
        fill = border_fill(img)
        w, h = img.size
        s = max(w, h)
        canvas = Image.new("L", (s, s), fill)
        canvas.paste(img, ((s - w) // 2, (s - h) // 2))
        return canvas

class RandomAffineBG:
    def __init__(self, degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)):
        self.degrees, self.translate, self.scale = degrees, translate, scale
    def __call__(self, img):
        fill = border_fill(img)
        return T.RandomAffine(self.degrees, self.translate, self.scale, fill=fill)(img)

def build_transform(img_size=224, train=False):
    steps = [PadToSquare(), T.Resize((img_size, img_size))]
    if train:
        steps += [RandomAffineBG(), T.ColorJitter(brightness=0.2, contrast=0.2)]
    steps += [T.Grayscale(num_output_channels=3), T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(steps)

def display_image(img, img_size=224):
    # float RGB array in [0, 1], used for the Grad-CAM overlay
    im = T.Resize((img_size, img_size))(PadToSquare()(img)).convert("RGB")
    return np.asarray(im).astype(np.float32) / 255.0
