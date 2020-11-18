import os
import numpy as np
from skimage import io

image_path = '../../../experiments/inria_segmentation/task0/pred' # path of cropped images
save_path = '../../../experiments/inria_segmentation/task0/pred_2500'     # path of destination folder
image_parts=[]
count = 0

if not os.path.exists(save_path):
	os.mkdir(save_path)

for (dirpath, dirnames, filenames) in os.walk(image_path):
	break

for file in filenames:
	image_parts.append(file)

n_images = len(image_parts)//4

result = np.zeros((5000, 5000)).astype(np.int)

for index, file in enumerate(image_parts):
	path = os.path.expanduser(file)
	img = io.imread(os.path.join(image_path, path))
	# img = img.resize((500, 500), Image.ANTIALIAS)
	y = index %4 // 2 * 2500
	x = index%4 % 2 * 2500
	w, h = img.shape
	print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
	result[x:x+w, y:y+h] = img
	if (index+1) % 4 == 0:
		name = file.split('_')[0]
		result[result>=127] = 255
		result[result<=127] = 0
		io.imsave(os.path.join(save_path, name)+'.png', result)