import torchvision
from PIL import Image
import cv2
import numpy as np
"""
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img
"""
trans1 = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ]
)

img_path = './img105.jpg'
img = Image.open(img_path)
print(img.size)                # w, h
img_PIL_tensor = trans1(img)
print(img_PIL_tensor.size())   # c, h, w

img = cv2.imread(img_path)
print(img.shape)               # h, w, c
img_cv2_tensor = trans1(img)
print(img_cv2_tensor.size())   # c, h, w

img = np.zeros([100, 200, 3])    # h, w, c
print(img.shape)
img_np_tensor = trans1(img)
print(img_np_tensor.size())    # c, h, w   torch.Size([3, 100, 200])
