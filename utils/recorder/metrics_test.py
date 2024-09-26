# recorder.py
import numpy as np

class TestMetricsCalculator:
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def update(self, pred_mask, gt_mask):
        # Convert masks to binary (0 or 1)
        pred_binary = (pred_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)

        # Calculate true positives, false positives, and false negatives
        self.true_positives += np.sum(pred_binary & gt_binary)
        self.false_positives += np.sum(pred_binary & ~gt_binary)
        self.false_negatives += np.sum(~pred_binary & gt_binary)

    def precision(self):
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    def recall(self):
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    def dice_coefficient(self):
        if self.true_positives + self.false_positives + self.false_negatives == 0:
            return 0.0
        return 2 * self.true_positives / (2 * self.true_positives + self.false_positives + self.false_negatives)

    def get_metrics(self):
        return {
            "Precision": self.precision(),
            "Recall": self.recall(),
            "Dice Coefficient": self.dice_coefficient(),
        }
