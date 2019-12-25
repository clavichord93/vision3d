import numpy as np


class AccuracyRecorderV1(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.num_correct = 0
        self.num_record = 0
        self.num_correct_per_class = [0] * num_class
        self.num_record_per_class = [0] * num_class

    def add_record(self, pred, label):
        result = pred == label
        self.num_correct += result
        self.num_record += 1
        self.num_correct_per_class[label] += result
        self.num_record_per_class[label] += 1

    def get_overall_accuracy(self):
        return self.num_correct / self.num_record

    def get_accuracy(self, class_id):
        return self.num_correct_per_class[class_id] / self.num_record_per_class[class_id]


class AccuracyRecorderV2(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.num_correct = 0
        self.num_record = 0
        self.num_correct_per_class = np.zeros(num_class, dtype=np.float)
        self.num_record_per_class = np.zeros(num_class, dtype=np.float)

    def add_records(self, preds, labels):
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
        return self.num_correct / self.num_record

    def accuracy_per_class(self, class_id):
        return self.num_correct_per_class[class_id] / self.num_record_per_class[class_id]


class PartMeanIoURecorder(object):
    r"""
    Mean IoU (Intersect over Union) metric for Part Segmentation task.
    """
    def __init__(self, num_class, num_part, class_id_to_part_ids):
        self.num_class = num_class
        self.num_part = num_part
        self.class_id_to_part_ids = class_id_to_part_ids
        self.ious = []
        self.ious_per_class = [[] for _ in range(num_class)]

    def add_records(self, preds, labels, class_id):
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
                iou = 1
            ious.append(iou)
        iou = np.mean(ious)
        self.ious.append(iou)
        self.ious_per_class[class_id].append(iou)

    def mean_iou_over_instance(self):
        return np.mean(self.ious)

    def mean_iou_over_class(self):
        mean_iou_per_class = [np.mean(self.ious_per_class[i]) for i in range(self.num_class)]
        return np.mean(mean_iou_per_class)

    def mean_iou_per_class(self, class_id):
        return np.mean(self.ious_per_class[class_id])