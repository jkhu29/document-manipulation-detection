import torch


def torch_nanmean(x):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum()
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum()
    return value / num


class SegmentationMetric(object):
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes)).to("cuda")

    def pixel_accuracy(self):
        acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def class_pixel_accuracy(self):
        acc = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return acc

    def mean_pixel_accuracy(self):
        class_acc = self.class_pixel_accuracy()
        acc = torch_nanmean(class_acc)
        return acc

    def intersection_over_union(self):
        intersection = torch.diag(self.confusion_matrix)
        union = torch.sum(self.confusion_matrix, axis=1) + \
                torch.sum(self.confusion_matrix, axis=0) - \
                torch.diag(self.confusion_matrix)
        IoU = intersection / union
        return IoU

    def mean_intersection_over_union(self):
        mIoU = torch_nanmean(self.intersection_over_union())
        return mIoU

    def gen_confusion_matrix(self, img_predict, img_label):
        mask = (img_label >= 0) & (img_label < self.num_classes)
        label = self.num_classes * img_label[mask] + img_predict[mask]
        count = torch.bincount(label, minlength=self.num_classes ** 2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def frequency_weighted_intersection_over_union(self):
        freq = torch.sum(self.confusion_matrix, axis=1) / torch.sum(self.confusion_matrix)
        iu = torch.diag(self.confusion_matrix) / \
            (
                torch.sum(self.confusion_matrix, axis=1) + \
                torch.sum(self.confusion_matrix, axis=0) - \
                torch.diag(self.confusion_matrix)
            )
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def add_batch(self, img_predict, img_label):
        assert img_predict.shape == img_label.shape
        self.confusion_matrix += self.gen_confusion_matrix(img_predict, img_label)
        return self.confusion_matrix

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes))
