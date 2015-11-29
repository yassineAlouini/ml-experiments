"""
Confusion matrix = Prediction vs Reference for a binary classification task
False positive rate (FPR) = FP/ P => should be close to 0
Accuracy or True positive rate (TPR) = TP/ P should be close to 1
FP + TP = PP (Predicted positives)
xaxis => FPR
yaxis => TPR
For a random, unbiased generator => TP = FP
For a great example, check this scikit-learn tutorial:
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import roc_curve, auc

# Generate some random data

np.random.seed(314151692)
nsamples = 1000
simulated_data = {
    'predicted': np.random.random(nsamples),
    'reference': np.random.random(nsamples)
}
random_data_frame = pd.DataFrame(data=simulated_data)

random_data_frame.plot()
