import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import albumentations as A
from colorama import Fore
from matplotlib import pyplot as plt
from utils.boxes import rescale_bboxes, stacker
from utils.setup import get_classes
from utils.logger import get_logger
from utils.rich_handlers import DataLoaderHandler
import sys


# Matches successful transform path: float32 CHW after ImageNet normalization (not raw [0, 1]).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class DETRData(Dataset):
    def __init__(self, path, train=True):
        super().__init__()
        self.path = path
        self.labels_path = os.path.join(self.path, "labels")
        self.images_path = os.path.join(self.path, "images")
        self.label_files = os.listdir(self.labels_path)
        self.labels = list(filter(lambda x: x.endswith(".txt"), self.label_files))
        self.train = train

        # Initialize logger
        self.logger = get_logger("data_loader")
        self.data_handler = DataLoaderHandler()

        # Log dataset initialization
        dataset_info = {
            "Dataset Path": self.path,
            "Mode": "Training" if train else "Testing",
            "Total Samples": len(self.labels),
            "Images Path": self.images_path,
            "Labels Path": self.labels_path,
        }
        self.data_handler.log_dataset_stats(dataset_info)

        # Log transforms information
        transform_list = [
            "Resize to 500x500",
            "Random Crop 224x224 (training only)",
            "Final Resize to 224x224",
            "Horizontal Flip p=0.5 (training only)",
            "Color Jitter (training only)",
            "Normalize (ImageNet stats)",
            "Convert to Tensor",
        ]
        self.data_handler.log_transform_info(transform_list)

    def _deterministic_fallback(self, image: np.ndarray, bboxes: np.ndarray, labels: np.ndarray):
        """
        When random augment pipeline fails repeatedly, apply the same Resize/Normalize/ToTensor
        stack so the image is always a torch.Tensor (C,H,W) float32, aligned with training stats.
        """
        t_with_bbox = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                A.ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )
        try:
            bl = bboxes.tolist() if hasattr(bboxes, "tolist") else list(bboxes)
            cl = [int(x) for x in (labels.tolist() if hasattr(labels, "tolist") else labels)]
            return t_with_bbox(image=image, bboxes=bl, class_labels=cl)
        except Exception:
            pass

        t_img = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                A.ToTensorV2(),
            ]
        )
        out = t_img(image=image)
        return {"image": out["image"], "bboxes": [], "class_labels": []}

    def _numpy_hwc_to_model_tensor(self, image: np.ndarray) -> torch.Tensor:
        """HWC uint8/float RGB -> CHW float32 with ImageNet normalization (no bbox path)."""
        x = torch.as_tensor(np.ascontiguousarray(image))
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {tuple(x.shape)}")
        x = x.permute(2, 0, 1).float()
        if x.max() > 1.0:
            x = x / 255.0
        mean = torch.tensor(_IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=x.dtype, device=x.device).view(3, 1, 1)
        return (x - mean) / std

    def safe_transform(self, image, bboxes, labels, max_attempts=50):
        self.transform = A.Compose(
            [
                A.Resize(500, 500),
                *([A.RandomCrop(width=224, height=224, p=0.33)] if self.train else []),
                A.Resize(224, 224),
                *([A.HorizontalFlip(p=0.5)] if self.train else []),
                *(
                    [A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5, p=0.5)]
                    if self.train
                    else []
                ),
                A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
                A.ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

        for attempt in range(max_attempts):
            try:
                transformed = self.transform(image=image, bboxes=bboxes, class_labels=labels)
                if len(transformed["bboxes"]) > 0:
                    return transformed
            except Exception:
                continue

        return self._deterministic_fallback(image, bboxes, labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        self.label_path = os.path.join(self.labels_path, self.labels[idx])
        self.image_name = self.labels[idx].split(".")[0]
        self.image_path = os.path.join(self.images_path, f"{self.image_name}.jpg")

        img = Image.open(self.image_path)
        with open(self.label_path, "r") as f:
            annotations = f.readlines()
        class_labels = []
        bounding_boxes = []
        for annotation in annotations:
            annotation = annotation.split("\n")[:-1][0].split(" ")
            class_labels.append(annotation[0])
            bounding_boxes.append(annotation[1:])
        class_labels = np.array(class_labels).astype(int)
        bounding_boxes = np.array(bounding_boxes).astype(float)

        augmented = self.safe_transform(image=np.array(img), bboxes=bounding_boxes, labels=class_labels)
        augmented_img_tensor = augmented["image"]

        if not isinstance(augmented_img_tensor, torch.Tensor):
            augmented_img_tensor = self._numpy_hwc_to_model_tensor(augmented_img_tensor)

        augmented_img_tensor = augmented_img_tensor.detach()
        if augmented_img_tensor.dtype != torch.float32:
            augmented_img_tensor = augmented_img_tensor.float()

        assert isinstance(augmented_img_tensor, torch.Tensor)
        assert augmented_img_tensor.ndim == 3

        bbs = augmented["bboxes"]
        cls = augmented["class_labels"]
        if len(bbs) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            augmented_bounding_boxes = np.array(bbs, dtype=np.float32)
            augmented_classes = np.array(cls, dtype=np.int64)
            labels = torch.tensor(augmented_classes, dtype=torch.long)
            boxes = torch.tensor(augmented_bounding_boxes, dtype=torch.float32)

        return augmented_img_tensor, {"labels": labels, "boxes": boxes}


if __name__ == "__main__":
    dataset = DETRData("data/test", train=False)
    dataloader = DataLoader(dataset, collate_fn=stacker, batch_size=4, drop_last=True)

    X, y = next(iter(dataloader))
    print(Fore.LIGHTCYAN_EX + str(y) + Fore.RESET)
    CLASSES = get_classes()
    fig, ax = plt.subplots(2, 2)
    axs = ax.flatten()
    for idx, (img, annotations, ax) in enumerate(zip(X, y, axs)):
        ax.imshow(img.permute(1, 2, 0))
        box_classes = annotations["labels"]
        boxes = rescale_bboxes(annotations["boxes"], (224, 224))
        for box_class, bbox in zip(box_classes, boxes):
            if box_class != 3:
                xmin, ymin, xmax, ymax = bbox.detach().numpy()
                print(xmin, ymin, xmax, ymax)
                ax.add_patch(
                    plt.Rectangle(
                        (xmin, ymin),
                        xmax - xmin,
                        ymax - ymin,
                        fill=False,
                        color=(0.000, 0.447, 0.741),
                        linewidth=3,
                    )
                )
                text = f"{CLASSES[box_class]}"
                ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))

    fig.tight_layout()
    plt.show()
