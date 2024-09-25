import torch

class MetricsCalculator:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def __str__(self):
        """
        Returns a string representation of the true positives, false positives,
        false negatives, and true negatives.
        """
        return (f"TP: {self.tp}, FP: {self.fp}, "
                f"FN: {self.fn}, TN: {self.tn}")

    def reset(self):
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives
        self.tn = 0  # True Negatives

    def update(self, outputs, targets):
        """
        Updates the confusion matrix counts based on the outputs and targets.
        
        Args:
            outputs (torch.Tensor): Predicted probabilities (after sigmoid if logits).
            targets (torch.Tensor): Ground truth masks.
        """
        assert outputs.shape == targets.shape

        outputs = outputs.detach()
        targets = targets.detach()

        if outputs.dim() == 4 and outputs.size(1) == 1:
            outputs = outputs.squeeze(1)
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        # print(f"Output max: {outputs.max()}, Output min: {outputs.min()}")

        preds = (outputs >= self.threshold).long()
        targets = targets.long()

        # print(f"Plaques: {(targets == 1).sum().item()}, Background: {(targets == 0).sum().item()}")

        tp = ((preds == 1) & (targets == 1)).sum().item()
        fp = ((preds == 1) & (targets == 0)).sum().item()
        fn = ((preds == 0) & (targets == 1)).sum().item()
        tn = ((preds == 0) & (targets == 0)).sum().item()

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

    def compute_metrics(self):
        """
        Computes the metrics based on the accumulated counts.

        Returns:
            dict: A dictionary containing precision, recall, accuracy, specificity, dice, and IoU.
        """
        eps = 1e-7  # Avoid division by zero

        print(self)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        specificity = self.tn / (self.tn + self.fp + eps)
        dice = (2 * self.tp) / (2 * self.tp + self.fp + self.fn + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)

        metrics = {
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'specificity': specificity,
            'dice': dice,
            'iou': iou
        }

        return metrics
