import os

import cv2
import numpy as np

import tfrecord


data_path = "./data/train"
mask_path = "./data/train_mask"

cnt = 0
size_image = 512
stride = 100
train_writer = tfrecord.TFRecordWriter("train.tfrecord")
valid_writer = tfrecord.TFRecordWriter("valid.tfrecord")

for img_name in os.listdir(data_path):
    img = cv2.imread(os.path.join(data_path, img_name))
    label = cv2.imread(os.path.join(mask_path, img_name.replace("jpg", "png")), 0)
    assert img is not None, "broken img: "+img_name
    for x in np.arange(0, img.shape[0] - size_image + 1, stride):
        for y in np.arange(0, img.shape[1] - size_image + 1, stride):
            img_part = img[int(x): int(x + size_image),
                           int(y): int(y + size_image)]
            img_part = img_part.transpose(2, 0, 1)
            label_part = label[int(x): int(x + size_image),
                               int(y): int(y + size_image)]
            if np.mean(label_part) < 30:
                continue

            if cnt < 8000:
                train_writer.write({
                    "inputs": (img_part.tobytes(), "byte"),
                    "labels": (label_part.tobytes(), "byte"),
                    "img_size": (size_image, "int"),
                })
                cnt += 1
            else:
                valid_writer.write({
                    "inputs": (img_part.tobytes(), "byte"),
                    "labels": (label_part.tobytes(), "byte"),
                    "img_size": (size_image, "int"),
                })

train_writer.close()
valid_writer.close()
print(cnt)
