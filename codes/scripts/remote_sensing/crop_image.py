import cv2
import os
import lmdb
import numpy as np
from sys import exit
# from dataloader import util

crop_size = 1024
total = 180
# stride = crop_size//4
stride = 512

image_path = '../../../data/INRIA/images_val/'  # path of training images
crop_image_path = '../../../data/INRIA/images_crop_big_val/'    # path to save cropped images
gt_path = '../../../data/INRIA/gt_val/'     # path of ground truth
crop_gt_path = '../../../data/INRIA/gt_crop_big_val/'   # path to save cropped ground truth
# lmdb_save_path = 'D:/data/AerialImageDataset/INRIA/INRIA_train.lmdb'
if not os.path.exists(crop_image_path):
	os.mkdir(crop_image_path)

if not os.path.exists(crop_gt_path):
	os.mkdir(crop_gt_path)

image_list = []
for (dirpath, dirnames, filenames) in os.walk(image_path):
	image_list.extend(filenames)
	break
image_list = sorted(image_list)

gt_list = []
for (dirpath, dirnames, filenames) in os.walk(gt_path):
	gt_list.extend(filenames)
	break
gt_list = sorted(gt_list)

assert len(image_list) == len(gt_list)

def save_file():
	for i in range(len(image_list)):
		assert image_list[i] == gt_list[i]
		img_color = cv2.imread(image_path + image_list[i])
		img_gt = cv2.imread(gt_path + gt_list[i], cv2.IMREAD_GRAYSCALE)

		for w in range((img_gt.shape[0] - crop_size) // stride + 1):
			for h in range((img_gt.shape[1] - crop_size) // stride + 1):
				data = img_color[h * stride:h * stride + crop_size, w * stride:w * stride + crop_size]
				im_name = image_list[i].split('.')[0] + '_' + str(w) + '_' + str(h) + '.png'
				cv2.imwrite(os.path.join(crop_image_path, im_name), data)

				label = img_gt[h * stride:h * stride + crop_size, w * stride:w * stride + crop_size]
				gt_name = gt_list[i].split('.')[0] + '_' + str(w) + '_' + str(h) + '.png'
				cv2.imwrite(os.path.join(crop_gt_path, gt_name), label)

def save_lmdb():
	count = 0
	scales = [1.0]
	initial_size = 1073741824*16 # 1073741824 = 1G
	env = lmdb.open(lmdb_save_path, map_size=initial_size)    # 7GB
	with env.begin(write=True) as txn:  # txn is a Transaction object
		for i in range(len(image_list)):
			img_color = cv2.imread(image_path+image_list[i])
			img_gt = cv2.imread(gt_path+gt_list[i], cv2.IMREAD_GRAYSCALE)
			for s in scales:
				width = int(img_gt.shape[0] * s)
				height = int(img_gt.shape[1]  * s)

				img = cv2.resize(img_color, (width, height), interpolation=cv2.INTER_CUBIC)
				im_gt = cv2.resize(img_gt, (width, height), interpolation=cv2.INTER_CUBIC)

				for w in range((im_gt.shape[0]-crop_size)//stride+1):
					for h in range((im_gt.shape[1]-crop_size)//stride+1):
						data = img[h*stride:h*stride+crop_size, w*stride:w*stride+crop_size]
						im_name = image_list[i].split('.')[0]+'_'+str(w)+'_'+str(h)+'.png'
						# cv2.imwrite(crop_image_path+name[0]+'_'+str(w)+'_'+str(h)+'.png', img2)

						label = im_gt[h*stride:h*stride+crop_size, w*stride:w*stride+crop_size]
						# gt_name = gt_list[i].split('.')
						# cv2.imwrite(crop_gt_path + name[0] + '_' + str(w) + '_' + str(h) + '.png', im_gt2)

						key = im_name.encode('ascii')
						if data.ndim == 2:
							H, W = data.shape
							C = 1
						else:
							H, W, C = data.shape
						meta_key = (im_name + '.meta').encode('ascii')
						meta = '{:d}, {:d}, {:d}'.format(H, W, C)
						label_key = (im_name + '.label').encode('ascii')

						# The encode is only essential in Python 3
						try:
							txn.put(key, np.ascontiguousarray(data))
							txn.put(meta_key, meta.encode('ascii'))
							txn.put(label_key, np.ascontiguousarray(label))
							count += 1
						except lmdb.MapFullError:
							import warnings

							warnings.warn('Needs {}x more space.'.format(len(image_list) / i))
							exit()
	print('{} images in total'.format(count))


save_file()
