"""
Confusion matrix = Prediction vs Reference for a binary classification task
False positive rate (FPR) = FP/ P => should be close to 0
Sensitivity, hit rate, recall or True positive rate (TPR) = TP/ P should be close to 1
FP + TP = PP (Predicted positives)
xaxis => FPR
yaxis => TPR
For a random, unbiased generator => TP = FP
For a great example, check this scikit-learn tutorial:
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
ROC: Receiver operating characteristic
AUC: Area under curve. auc(fpr, tpr)
"""

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import auc, roc_curve
from sklearn.tree import DecisionTreeClassifier

# Generate some random data

np.random.seed(314151692)
N_SAMPLES = 1000
N_FEATURES = 2
TEST_SIZE = 0.2
simulated_data = {
    'target': np.random.random_integers(0, 1, N_SAMPLES),
    'features': np.random.rand(N_SAMPLES, N_FEATURES)
}

features = simulated_data['features']
target = simulated_data['target']
train_features, test_features, train_target, test_target = train_test_split(
    features, target, test_size=TEST_SIZE,
    random_state=314151692)

# Make a simple classifier


classifier = DecisionTreeClassifier()
classifier.fit(train_features, train_target)
predicted_target_probabilites = classifier.predict_proba(test_features)[:, 1]


# Construct the ROC curve and plot it
tpr, fpr, threshold = roc_curve(test_target, predicted_target_probabilites)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.plot(tpr, fpr, label='Decision tree')
plt.show()
