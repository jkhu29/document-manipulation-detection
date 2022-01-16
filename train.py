import random
import warnings

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import dataloader
from tfrecord.torch.dataset import TFRecordDataset

from torchvision import transforms

from tqdm import tqdm

import segmentation.metrics as metrics
from segmentation.models import ResNeXTDAHead

import utils
from configs import seg as config


opt = config.get_options()

# device init
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.benchmark = True

# seed init
manual_seed = opt.seed
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# dataset init, train file need .tfrecord
description = {
    "inputs": "byte",
    "labels": "byte",
    "img_size": "int",
}
train_dataset = TFRecordDataset("train.tfrecord", None, description, shuffle_queue_size=1024)
train_dataloader = dataloader.DataLoader(
    dataset=train_dataset,
    batch_size=opt.batch_size,
    num_workers=opt.workers,
    pin_memory=True,
    drop_last=True
)
length = 4000

valid_dataset = TFRecordDataset("valid.tfrecord", None, description)
valid_dataloader = dataloader.DataLoader(
    dataset=valid_dataset,
    batch_size=1,
    num_workers=0,
    pin_memory=True,
)

# models init
model = ResNeXTDAHead(img_size=256, pretrain=True).to(device)

# criterion init
criterion = nn.BCELoss().to(device)

# prep
prep = transforms.Compose(
    [
        transforms.Lambda(lambda x: x.mul_(1 / 255)),
        transforms.Normalize(
            mean=[0.40760392, 0.4595686, 0.48501961],
            std=[0.225, 0.224, 0.229]
        ),
        # WARNING(hujiakui): Lambda --> inplace ops, can't backward
    ]
)

# optim and scheduler init
model_optimizer = optim.Adam(model.parameters(), lr=opt.lr, eps=1e-8, weight_decay=1)
model_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=opt.niter)

# train model
print("-----------------train-----------------")
for epoch in range(opt.niter):
    model.train()
    epoch_losses = utils.AverageMeter()

    with tqdm(total=(length - length % opt.batch_size)) as t:
        t.set_description('epoch: {}/{}'.format(epoch + 1, opt.niter))

        for record in train_dataloader:
            img_size = record["img_size"][0]
            inputs = record["inputs"].reshape(
                opt.batch_size,
                3,
                img_size,
                img_size,
            ).float().to(device)
            labels = record["labels"].reshape(
                opt.batch_size,
                1,
                img_size,
                img_size,
            ).float().to(device) / 255
            inputs = prep(inputs)

            out = model(inputs)

            model_optimizer.zero_grad()
            loss = criterion(out, labels)
            loss.backward()

            model_optimizer.step()
            epoch_losses.update(loss.item(), opt.batch_size)

            t.set_postfix(
                loss=epoch_losses.avg,
            )
            t.update(opt.batch_size)

    model_scheduler.step()
    
    model.eval()
    epoch_val_losses = utils.AverageMeter()
    epoch_val_pa = utils.AverageMeter()
    epoch_val_mpa = utils.AverageMeter()
    epoch_val_miou = utils.AverageMeter()
    cnt = 0
    with torch.no_grad():
        for record in valid_dataloader:
            img_size = record["img_size"][0]
            image = record["inputs"].reshape(
                1,
                3,
                img_size,
                img_size,
            ).float().to(device)
            image = prep(image)

            label = record["labels"].reshape(
                1,
                1,
                img_size,
                img_size,
            ).float().to(device) / 255

            pred = model(image)

            if cnt == 0:
                img_input = image.squeeze().cpu().numpy().transpose(1, 2, 0)
                img_label = label.squeeze().cpu().numpy()

                img_pred = pred.squeeze().cpu().numpy()
                img_pred_mask = np.zeros_like(img_pred)
                img_pred_mask[np.where(img_pred > 0.5)] = 1

                cv2.imwrite("demo.jpg", img_input * 255)
                cv2.imwrite("demo_pred.jpg", img_pred * 255)
                cv2.imwrite("demo_pred_mask_epoch_%d.jpg" % (epoch), img_pred_mask * 255)
                cv2.imwrite("demo_mask.jpg", img_label * 255)
                cv2.waitKey()
                cv2.destroyAllWindows()

                cnt += 1
                break

            val_loss = criterion(pred, label)

            # metric init
            metric = metrics.SegmentationMetric(2)
            hist = metric.add_batch(pred.int().squeeze(), label.int().squeeze())
            pa = metric.pixel_accuracy()
            mpa = metric.mean_pixel_accuracy()
            mIoU = metric.mean_intersection_over_union()

            epoch_val_losses.update(val_loss.item(), 1)
            epoch_val_pa.update(pa, 1)
            epoch_val_mpa.update(mpa, 1)
            epoch_val_miou.update(mIoU, 1)

    log = "epoch_%d-valid_loss_%.6f-pa_%.6f-mpa_%.6f-mIoU_%.6f" % (
        epoch, epoch_val_losses.avg, epoch_val_pa.avg, epoch_val_mpa.avg, epoch_val_miou.avg
    )
    torch.save(model.state_dict(), "./checkpoints/seg/" + log + ".pth")
    print(log)
