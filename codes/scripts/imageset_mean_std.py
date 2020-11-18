import glob
import numpy as np
from skimage import io

# path = '../../data/ISIC/ISIC2018_Task3_Training_Input'
# path = '../../data/OxfordFlowers\jpg'
path = '../../data/StanfordDogs/images/Images/n02085620-Chihuahua'

img_list = glob.glob(path+'/*.jpg')


imgs = [io.imread(file).transpose([2,1,0]).reshape(3, -1)/255. for file in img_list]
imgs = np.concatenate(imgs, 1)
# for file in img_list:
# 	img = io.imread(file)/255.
# 	img_mean += img.mean(0).mean(0)
# 	img_std += img.std(0).std(0)

# img_mean /= len(img_list)
# img_std /= len(img_list)
img_mean = imgs.mean(1)
img_std = imgs.std(1)
print('mean and std are: {}, {}'.format(img_mean, img_std))
