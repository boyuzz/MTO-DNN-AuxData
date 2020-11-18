import numpy as np
from skimage import io


# x1 = io.imread('../../data/RSSCN7/train/fResident/f049.jpg')/255
# x2 = io.imread('../../data/RSSCN7/train/fResident/f076.jpg')/255
x1 = io.imread('../../data/RSSCN7/train/bField/b213.jpg')/255
x2 = io.imread('../../data/RSSCN7/train/bField/b214.jpg')/255
a1 = np.random.rand()
a2 = 1-a1
x1_shape = x1.shape
x2_shape = x2.shape
w = min(x1_shape[0], x2_shape[0])
h = min(x1_shape[1], x2_shape[1])
w1_s = np.random.randint(x1_shape[0]-w) if x1_shape[0]>w else 0
h1_s = np.random.randint(x1_shape[1]-h) if x1_shape[1]>h else 0
w2_s = np.random.randint(x2_shape[0]-w) if x2_shape[0]>w else 0
h2_s = np.random.randint(x2_shape[1]-h) if x2_shape[1]>h else 0

x = x1[w1_s:w1_s+w, h1_s:h1_s+h, :]*a1+x2[w2_s:w2_s+w, h2_s:h2_s+h, :]*a2
io.imsave('b.jpg', x)
print(a1, a2)