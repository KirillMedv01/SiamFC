import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.tensorboard.summary import mesh


def crop_and_resize(img, center, size, out_size, pad_value=0):
    cx, cy = center
    np_img = np.array(img)
    H, W = np_img.shape[:2]
    pad_color = np_img.reshape(-1, np_img.shape[-1]).mean(axis=0)

    x1 = int(round(cx - size / 2))
    y1 = int(round(cy - size / 2))
    x2 = x1 + int(round(size))
    y2 = y1 + int(round(size))

    left_pad = max(0, -x1)
    top_pad = max(0, -y1)
    right_pad = max(0, -(W - x2))
    bottom_pad = max(0, -(H - y2))

    if any([left_pad, top_pad, right_pad, bottom_pad]):
        np_img = np.pad(
            np_img,
            ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
            mode='constant', constant_values=0)
        if top_pad:
            np_img[:top_pad, :] = pad_color
        if bottom_pad:
            np_img[-bottom_pad:, :] = pad_color
        if left_pad:
            np_img[:, :left_pad] = pad_color
        if right_pad:
            np_img[:, -right_pad:] = pad_color
        x1 += left_pad
        y1 += top_pad
        x2 += right_pad
        y2 += bottom_pad

    patch = np_img[y1:y2, x1:x2]
    patch = Image.fromarray(patch)
    patch = patch.resize((out_size, out_size), Image.BILINEAR)
    return patch


class SiamFCTransforms:
    def __init__(self, exemplar_size=127, instance_size=255, context=0.5, max_translate=16):
        self.exemplar_size = exemplar_size
        self.instance_size = instance_size
        self.context = context
        self.max_translate = max_translate

    def _canonical_size(self, w, h):
        context_amount = self.context * (w + h)
        return np.sqrt((w + context_amount) * (h + context_amount))

    def exemplar(self, img, box):
        x, y, w, h = box  # top-left x, y, width, height
        cx, cy = x + w / 2, y + h / 2
        z_size = self._canonical_size(w, h)
        return crop_and_resize(img, (cx, cy), z_size, self.exemplar_size)

    def search(self, img, box):
        x, y, w, h = box
        cx, cy = x + w / 2, y + h / 2
        z_size = self._canonical_size(w, h)
        x_size = z_size * (self.instance_size / self.exemplar_size)

        scale = x_size / self.instance_size
        dx = random.uniform(-self.max_translate, self.max_translate) * scale
        dy = random.uniform(-self.max_translate, self.max_translate) * scale

        patch = crop_and_resize(img, (cx + dx, cy + dy), x_size, self.instance_size)
        offset = (-dx / scale, -dy / scale)
        return patch, offset

def pil_to_tensor(img):
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img

class Got10KValDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transforms = transform or SiamFCTransforms()
        self.samples = []

        sequence_dirs = sorted(
            path for path in self.root.iterdir()
            if path.is_dir()
        )

        for sequence_dir in sequence_dirs:
            annotation_path = sequence_dir / 'groundtruth.txt'
            boxes = self._read_grondtruth(annotation_path)

            path_images = sorted(
                path for path in sequence_dir.iterdir()
                if path.suffix == '.jpg'
            )

            for i in range(1, len(boxes)):
                self.samples.append(
                    {
                        'template': path_images[0],
                        'template_box': boxes[0],
                        'search': path_images[i],
                        'search_box': boxes[i],
                    }
                )

    def _read_grondtruth(self, annotation_path):
        boxes = []
        with open(annotation_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line.strip()
                boxes.append(list(map(float, line.split(','))))

        return boxes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        template = Image.open(sample['template']).convert('RGB')
        search = Image.open(sample['search']).convert('RGB')

        template = self.transforms.exemplar(template, sample['template_box'])
        search, offset = self.transforms.search(search, sample['search_box'])

        return {
            'template': pil_to_tensor(template),
            'search': pil_to_tensor(search),
            'offset': torch.tensor(offset, dtype=torch.float32),
        }

def create_labels(out_shape, offsets, radius=16, total_stride=8):
    h, w = out_shape
    n = offsets.size(0)

    xs = torch.arange(w, dtype=torch.float32) - (w-1)/2
    ys = torch.arange(h, dtype=torch.float32) - (h-1)/2
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')

    cx = (offsets[:,0]/total_stride).view(n,1,1)
    cy = (offsets[:,1]/total_stride).view(n,1,1)

    dist = torch.sqrt((xx.unsqueeze(0) - cx) ** 2 + (yy.unsqueeze(0) - cy) ** 2) * total_stride
    labels = (dist <= radius).float()

    pos_num = labels.sum(dim=(1, 2), keepdim=True).clamp(min=1)
    neg_num = (h * w - pos_num).clamp(min=1)
    weights = torch.where(labels == 1, 0.5 / pos_num, 0.5 / neg_num)
    return labels, weights