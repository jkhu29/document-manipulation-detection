# image manipulation detection

task: --> [tianchi](https://tianchi.aliyun.com/competition/entrance/531812/forum)

## function

1. image segmentation
2. salient object detection

### image segmentation

- use ResNeXT | Res2Net | SeResNeXT as encoder;
- use UNet | R2UNet framework;
- use DAHead | PSPHead as decode_head;

see [seg model](./segmentation/README.md)

### salient object detection

- use U2Net | EGNet | CSF+Res2Net;

see [det model](./detection/README.md)

## usage

use `make_dataset.py` to make the `.tfrecord` files

```shell
python -W ignore train.py --batch_size $batch_size --niter $niter --lr $lr
```

## todo

- [ ] data aug
- [ ] model fusion
