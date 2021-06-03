import numpy as np

Classes1=['Benign','Malignant']
np.save('classes.npy',Classes1)
Classes = np.load('classes.npy')
print(Classes1)
