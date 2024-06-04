from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models.resnet import ResNet18_Weights


class ImageData:
    def __init__(self, img_dir: str):  # noqa: ANN101, ANN204
        self.d = img_dir

    def load_images(self) -> list[Image.Image]:  # noqa: ANN101
        return [
            Image.open(Path(self.d) / f)
            for f in os.listdir(self.d)
            if f.endswith((".jpg", ".png"))
        ]


class ImgProcess:
    def __init__(self, size: int):  # noqa: D107, ANN204, ANN101
        self.s = size

    def resize_and_gray(self, img_list: list[Image.Image]) -> list:  # noqa: ANN101
        p_images = []
        for img in img_list:
            t = transforms.Compose(
                [
                    transforms.Resize((self.s, self.s)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[
                            0.229,
                            0.224,
                            0.225,
                        ],
                    ),
                ],
            )
            p_images.append(t(img))
        return p_images


class Predictor:
    def __init__(self):
        self.mdl = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.mdl.eval()

    def predict_img(self, proc_imgs: list) -> list:
        res = []
        for img_tensor in proc_imgs:
            predictions = self.mdl(img_tensor.unsqueeze(0))
            res.append(torch.argmax(predictions, dim=1).item())
        return res


if __name__ == "__main__":
    loader = ImageData("images/")
    images = loader.load_images()

    processor = ImgProcess(256)
    processed_images = processor.resize_and_gray(images)

    pred = Predictor()
    results = pred.predict_img(processed_images)
    print(results)  # noqa: T201
