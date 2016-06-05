import numpy as np
from sklearn.metrics import f1_score

y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 1, 1, 1, 0, 0, 1, 0])


class ClassificationMetrics(object):
    """
    Compute some (bnary) classification metrics. By default, the positive class
    is labeled with 1, the negative one is 0.
    """

    def __init__(self, true, pred, positive=1, negative=0):
        assert pred.shape == true.shape
        self.pred = pred
        self.true = true
        self.positive = positive
        self.negative = negative

    @property
    def TP(self):
        return ((self.pred == self.positive) * (self.true == self.positive)).sum()

    @property
    def FP(self):
        return ((self.pred == self.positive) * (self.true == self.negative)).sum()

    @property
    def FN(self):
        return ((self.pred == self.negative) * (self.true == self.positive)).sum()

    @property
    def recall(self):
        return self.TP / (self.TP + self.FN)

    @property
    def precision(self):
        return self.TP / (self.TP + self.FP)

    @property
    def f1_score(self):
        return 2 * (self.recall * self.precision) / (self.recall + self.precision)


clf_metrics = ClassificationMetrics(y_true, y_pred)
print("My own implementation of the F1 score gives: ", clf_metrics.f1_score)
print("Sklearn implementation of the F1 score gives: ", f1_score(y_true, y_pred))
assert f1_score(y_true, y_pred) == clf_metrics.f1_score
