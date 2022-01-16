# image manipulation detection

task: --> [tianchi](https://tianchi.aliyun.com/competition/entrance/531812/forum)

## function

1. image segmentation
2. salient object detection

### seg

- use ResNeXT as encoder;
- use UNet framework;
- use DAHead as decode_head;

see [seg model](./segmentation/README.md)

mIoU: 

## usage

use `make_dataset.py` to make the `.tfrecord` files

```shell
python -W ignore train.py --batch_size $batch_size --niter $niter --lr $lr
```

## todo

- [ ] refine the u2net
- [ ] add: [EGNet](https://github.com/JXingZhao/EGNet/)
- [ ] data aug
- [ ] model fusion
