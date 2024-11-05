import sys
import numpy as np
import matplotlib.pyplot as plt                                                                                                                                                                                                                      
from sklearn.metrics import ConfusionMatrixDisplay
 
 
if __name__ == '__main__':
    file_name = sys.argv[1]
    conf_matrix = np.load(open(file_name, 'rb'))
    labels = np.arange(1, 11)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
 
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45)
 
    plt.show()
