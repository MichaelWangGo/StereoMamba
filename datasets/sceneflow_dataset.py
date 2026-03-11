import os
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, pfm_imread


class SceneFlowDatset(Dataset):
    def __init__(self, datapath, list_filename, training, eval_pad_to=32):
        self.datapath = datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        self.eval_pad_to = eval_pad_to

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 640, 320

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}
        else:
            w, h = left_img.size
            crop_w, crop_h = 960, 540

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity = disparity[h - crop_h:h, w - crop_w: w]

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            top_pad = 0
            right_pad = 0


            # Pad to nearest multiple (ceil to multiple), default 32
            if self.eval_pad_to is not None and self.eval_pad_to > 1:
                h_tensor, w_tensor = left_img.shape[-2:]
                top_pad = (self.eval_pad_to - (h_tensor % self.eval_pad_to)) % self.eval_pad_to
                right_pad = (self.eval_pad_to - (w_tensor % self.eval_pad_to)) % self.eval_pad_to

            if top_pad > 0 or right_pad > 0:
                left_img = F.pad(left_img, (0, right_pad, 0, top_pad))
                right_img = F.pad(right_img, (0, right_pad, 0, top_pad))
                disparity = np.pad(
                    disparity,
                    ((0, top_pad), (0, right_pad)),
                    mode="constant",
                    constant_values=0,
                ).astype(np.float32, copy=False)

            return {
                "left": left_img,
                "right": right_img,
                "disparity": disparity,
                "top_pad": top_pad,
                "right_pad": right_pad
            }