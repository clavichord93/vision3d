import numpy as np


def _safe_divide(overall_sum, num_record):
    if num_record == 0:
        return 0
    else:
        return overall_sum / num_record


class AverageMeter(object):
    def __init__(self):
        self.overall_sum = 0
        self.num_record = 0

    def add_results(self, results):
        if not isinstance(results, list):
            results = [results]
        self.num_record += len(results)
        self.overall_sum += np.sum(results)

    def average(self):
        return _safe_divide(self.overall_sum, self.num_record)


class AccuracyMeter(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.num_correct = 0
        self.num_record = 0
        self.num_correct_per_class = [0] * num_class
        self.num_record_per_class = [0] * num_class

    def add_results(self, preds, labels):
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)
        results = np.equal(preds, labels)
        self.num_correct += np.sum(results)
        self.num_record += results.size
        for i in range(self.num_class):
            results_per_class = results[labels == i]
            self.num_correct_per_class[i] += np.sum(results_per_class)
            self.num_record_per_class[i] += results_per_class.size

    def accuracy(self):
        return _safe_divide(self.num_correct, self.num_record)

    def mean_accuracy(self):
        return np.mean([self.accuracy_per_class(i) for i in range(self.num_class)])

    def accuracy_per_class(self, class_id):
        return _safe_divide(self.num_correct_per_class[class_id], self.num_record_per_class[class_id])


class PartMeanIoUMeter(object):
    r"""
    Mean IoU (Intersect over Union) metric for Part Segmentation task.
    """
    def __init__(self, num_class, num_part, class_id_to_part_ids):
        self.num_class = num_class
        self.num_part = num_part
        self.class_id_to_part_ids = class_id_to_part_ids
        self.ious = []
        self.ious_per_class = [[] for _ in range(num_class)]

    def add_results(self, preds, labels, class_ids):
        batch_size = preds.shape[0]
        for i in range(batch_size):
            self._add_result(preds[i], labels[i], class_ids[i])

    def _add_result(self, preds, labels, class_id):
        ious = []
        part_ids = self.class_id_to_part_ids[class_id]
        for part_id in part_ids:
            labels_per_part = np.equal(labels, part_id)
            preds_per_part = np.equal(preds, part_id)
            intersect_per_part = np.sum(np.logical_and(labels_per_part, preds_per_part))
            union_per_part = np.sum(np.logical_or(labels_per_part, preds_per_part))
            if union_per_part > 0:
                iou = intersect_per_part / union_per_part
            else:
                iou = 1.
            ious.append(iou)
        iou = np.mean(ious)
        self.ious.append(iou)
        self.ious_per_class[class_id].append(iou)

    def instance_miou(self):
        return np.mean(self.ious)

    def class_miou(self):
        mean_iou_per_class = [self.instance_miou_per_class(i) for i in range(self.num_class)]
        return np.mean(mean_iou_per_class)

    def instance_miou_per_class(self, class_id):
        return np.mean(self.ious_per_class[class_id])


class MeanIoUMeter(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.intersect_per_class = [0] * num_class
        self.union_per_class = [0] * num_class

    def add_results(self, preds, labels):
        for class_id in range(self.num_class):
            preds_per_class = np.equal(preds, class_id)
            labels_per_class = np.equal(labels, class_id)
            intersect = np.count_nonzero(np.logical_and(preds_per_class, labels_per_class))
            union = np.count_nonzero(np.logical_or(preds_per_class, labels_per_class))
            self.intersect_per_class[class_id] += intersect
            self.union_per_class[class_id] += union

    def class_miou(self):
        return np.mean([self.class_miou_per_class(i) for i in range(self.num_class)])

    def class_miou_per_class(self, class_id):
        return _safe_divide(self.intersect_per_class[class_id], self.union_per_class[class_id])
