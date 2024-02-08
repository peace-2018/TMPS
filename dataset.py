import os
from PIL import Image
import torchvision.transforms as transforms_vision
import torch
import torch.utils.data
import torch.nn.functional as torch_F
class Dataset(torch.utils.data.Dataset):
    def __init__(self, size_h, size_w, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):

        self.size_h = size_h
        self.size_w = size_w
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_open = Image.open(os.path.join(self.img_dir, img_id + self.img_ext)).convert('RGB')
        mask_open = Image.open(os.path.join(self.mask_dir, img_id + self.mask_ext)).convert('L')
        tensor_img = transforms_vision.ToTensor()(img_open)
        tensor_mask = transforms_vision.ToTensor()(mask_open)
        tensor_img_size = tensor_img.size()[0]
        tensor_mask_size = tensor_mask.size()[0]
        if self.transform is not None:
            tensor_img_mask_cat = torch.cat((tensor_img, tensor_mask), dim=0)

            tensor_img_mask_cat_2 = self.transform(tensor_img_mask_cat)

            img, mask = torch.split(tensor_img_mask_cat_2,
                                    split_size_or_sections=[tensor_img_size, tensor_mask_size],
                                    dim=0)
            img_resize = img.unsqueeze(0)
            mask_resize = mask.unsqueeze(0)
            img_resize = torch_F.interpolate(img_resize, size=(self.size_h, self.size_w), mode='bilinear', align_corners=False)
            mask_resize = torch_F.interpolate(mask_resize, size=(self.size_h, self.size_w), mode='bilinear', align_corners=False)
            mask_resize_2 = mask_resize.squeeze(0)
            img_resize_2 = img_resize.squeeze(0)
            img_resize_nomalize = transforms_vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                img_resize_2)
        else:
            img = tensor_img.detach().clone()
            mask = tensor_mask.detach().clone()
            img_resize = img.unsqueeze(0)
            mask_resize = mask.unsqueeze(0)
            img_resize = torch_F.interpolate(img_resize, size=(self.size_h, self.size_w), mode='bilinear',
                                             align_corners=False)
            mask_resize = torch_F.interpolate(mask_resize, size=(self.size_h, self.size_w), mode='bilinear',
                                              align_corners=False)
            mask_resize_2 = mask_resize.squeeze(0)
            img_resize_2 = img_resize.squeeze(0)
            img_resize_nomalize = transforms_vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
                img_resize_2)

        return img_resize_nomalize, mask_resize_2, {'img_id': img_id}
