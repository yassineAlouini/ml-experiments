import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

np.random.seed(314151692)
y_pred = np.random.random_integers(0, 1, 1000)
y_true = np.random.random_integers(0, 1, 1000)


data = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(1)
sns.heatmap(data, ax=ax)
ax.set_ylabel('True label')
ax.set_xlabel('Predicted label')

fig.show()
