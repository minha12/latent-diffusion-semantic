import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset


class SegmentationBase(Dataset):
    def __init__(self,
                 data_csv, data_root, segmentation_root,
                 size=None, random_crop=False, interpolation="bicubic",
                 n_labels=5, shift_segmentation=False,
                 force_resize=True,
                 ):
        self.n_labels = n_labels
        self.shift_segmentation = shift_segmentation
        self.data_csv = data_csv
        self.data_root = data_root
        self.segmentation_root = segmentation_root
        self.force_resize = force_resize
        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
            "segmentation_path_": [os.path.join(self.segmentation_root, l.replace(".jpg", ".png"))
                                   for l in self.image_paths]
        }

        size = None if size is not None and size<=0 else size
        self.size = size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                interpolation=self.interpolation)
            self.segmentation_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                        interpolation=cv2.INTER_NEAREST)
            
            # Direct resize transformations (used when both dimensions > target size)
            self.image_resizer = albumentations.Resize(height=self.size, width=self.size,
                                                      interpolation=self.interpolation)
            self.segmentation_resizer = albumentations.Resize(height=self.size, width=self.size,
                                                             interpolation=cv2.INTER_NEAREST)
            
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            
            # We'll decide on the preprocessor in __getitem__ based on image dimensions

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        
        segmentation = Image.open(example["segmentation_path_"])
        assert segmentation.mode == "L", segmentation.mode
        segmentation = np.array(segmentation).astype(np.uint8)
        if self.shift_segmentation:
            # used to support segmentations containing unlabeled==255 label
            segmentation = segmentation+1
        
        if self.size is not None:
            # Check if we should force resize or use original crop-based approach
            if self.force_resize and image.shape[0] > self.size and image.shape[1] > self.size:
                # Both dimensions > target size, directly resize to target size
                processed = {"image": self.image_resizer(image=image)["image"],
                            "mask": self.segmentation_resizer(image=segmentation)["image"]}
            else:
                # Original approach: resize smallest dimension then crop
                image = self.image_rescaler(image=image)["image"]
                segmentation = self.segmentation_rescaler(image=segmentation)["image"]
                processed = self.cropper(image=image, mask=segmentation)
        else:
            processed = {"image": image, "mask": segmentation}
            
        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        segmentation = processed["mask"]
        onehot = np.eye(self.n_labels)[segmentation]
        example["segmentation"] = onehot
        return example


class Examples(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv="data/sflckr_examples.txt",
                         data_root="data/sflckr_images",
                         segmentation_root="data/sflckr_segmentations",
                         size=size, random_crop=random_crop, interpolation=interpolation)


class DrskSeg(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv=os.path.expanduser("~/datasets/drsk/image_names.txt"),
                         data_root=os.path.expanduser("~/datasets/drsk/images"),
                         segmentation_root=os.path.expanduser("~/datasets/drsk/plain-segmentation"),
                         size=size, random_crop=random_crop, interpolation=interpolation)


class DrskSegTrain(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv=os.path.expanduser("~/datasets/drsk/image_names_train.txt"),
                         data_root=os.path.expanduser("~/datasets/drsk/images"),
                         segmentation_root=os.path.expanduser("~/datasets/drsk/plain-segmentation"),
                         size=size, random_crop=random_crop, interpolation=interpolation)


class DrskSegEval(SegmentationBase):
    def __init__(self, size=None, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv=os.path.expanduser("~/datasets/drsk/image_names_eval.txt"),
                         data_root=os.path.expanduser("~/datasets/drsk/images"),
                         segmentation_root=os.path.expanduser("~/datasets/drsk/plain-segmentation"),
                         size=size, random_crop=random_crop, interpolation=interpolation)