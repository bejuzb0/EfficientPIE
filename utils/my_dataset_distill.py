"""
my_dataset_distill.py

Phase 2 of offline VLM Knowledge Distillation.
Dual input dataset loading both the pedestrian crop and the offline VLM embeddings.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class JAADDatasetDistill(Dataset):

    def __init__(self, images_seq: list, data_opts: dict, transform=None, step=None):
        self.images_seq = images_seq
        self.data_opts = data_opts
        self.transform = transform
        self.step = step

    def __len__(self):
        length = len(self.images_seq['images'])
        return length

    def __getitem__(self, index):
        each_seq_imgs = self.images_seq['images'][index]
        each_seq_bboxes = self.images_seq['bboxes'][index]
        each_seq_labels = self.images_seq['output'][index]

        reverse_step = 1
        last_img = each_seq_imgs[self.data_opts['max_size_observe'] - reverse_step]
        last_bbox = each_seq_bboxes[self.data_opts['max_size_observe'] - reverse_step]
        last_label = each_seq_labels[self.data_opts['max_size_observe'] - reverse_step]

        # 1. Load standard crop
        img = Image.open(last_img)
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(last_img))
            
        x1, y1, x2, y2 = last_bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        half_size = 150  # 150
        new_x1 = int(center_x - half_size)
        new_y1 = int(center_y - half_size)
        new_x2 = int(center_x + half_size)
        new_y2 = int(center_y + half_size)
        
        img_width, img_height = img.size
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(img_width, new_x2)
        new_y2 = min(img_height, new_y2)
        
        crop_box = [new_x1, new_y1, new_x2, new_y2]
        img = img.crop(crop_box)

        if self.transform is not None:
            img = self.transform(img)
            
        # 2. Load Label
        label = torch.as_tensor(last_label)
        
        # 3. Load VLM tensor
        # Map: .../images/video_0001/00021.png -> .../vlm_features/video_0001/00021_vlm.pt
        vlm_feat_path = last_img.replace(os.sep + 'images' + os.sep, os.sep + 'vlm_features' + os.sep)
        vlm_feat_path = vlm_feat_path.rsplit('.', 1)[0] + '_vlm.pt'
        
        if not os.path.exists(vlm_feat_path):
            # Because python scripts might run from different dirs, fallback matching just video_xxxx/yyyy_vlm.pt
            # if direct replace fails due to path semantics
            path_parts = os.path.normpath(last_img).split(os.sep)
            # Find the root that contains 'images'
            try:
                img_idx = path_parts.index('images')
                path_parts[img_idx] = 'vlm_features'
                vlm_feat_path = os.sep.join(path_parts)
                vlm_feat_path = vlm_feat_path.rsplit('.', 1)[0] + '_vlm.pt'
            except ValueError:
                pass
                
            if not os.path.exists(vlm_feat_path):
                raise FileNotFoundError(f"Offline VLM tensor missing for {last_img}: expected at {vlm_feat_path}")
        
        vlm_feat = torch.load(vlm_feat_path, map_location='cpu', weights_only=True)
        # Squeeze if it got saved with batch dim of 1 (depends on extract_vlm_features.py output)
        vlm_feat = vlm_feat.squeeze(0) 
        
        return img, label, vlm_feat

    @staticmethod
    def collate_fn(batch):
        imgs, labels, vlms = tuple(zip(*batch))

        imgs = torch.stack(imgs, dim=0)
        
        labels = torch.stack(labels)
        labels = torch.squeeze(labels, dim=1)
        
        vlms = torch.stack(vlms, dim=0)

        return imgs, labels, vlms
