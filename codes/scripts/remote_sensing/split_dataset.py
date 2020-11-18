import os
from os.path import join
import shutil
import numpy as np

patch_per_img = 1
crop_image_path = '../../../data/INRIA/images_crop/'    # path to save cropped images
val_crop_image_path = '../../../data/INRIA/images_crop_val/'    # path to save cropped images
crop_gt_path = '../../../data/INRIA/gt_crop/'   # path to save cropped ground truth
val_crop_gt_path = '../../../data/INRIA/gt_crop_val/'   # path to save cropped ground truth


if not os.path.exists(val_crop_image_path):
	os.mkdir(val_crop_image_path)

if not os.path.exists(val_crop_gt_path):
	os.mkdir(val_crop_gt_path)

image_list = []
for (dirpath, dirnames, filenames) in os.walk(crop_image_path):
	image_list.extend(filenames)
	break
image_list = np.array(sorted(image_list))

gt_list = []
for (dirpath, dirnames, filenames) in os.walk(crop_gt_path):
	gt_list.extend(filenames)
	break
gt_list = np.array(sorted(gt_list))

assert len(image_list) == len(gt_list)

def average_pick():
	for i in range(0, len(gt_list)//patch_per_img):
		rand_pick = np.random.permutation(range(patch_per_img))[:patch_per_img//10]
		sub_img_list = image_list[rand_pick+i*patch_per_img]
		sub_gt_list = gt_list[rand_pick+i*patch_per_img]
		# pass
		list(map(lambda x: shutil.move(join(crop_image_path, x), join(val_crop_image_path, x)), sub_img_list))
		list(map(lambda x: shutil.move(join(crop_gt_path, x), join(val_crop_gt_path, x)), sub_gt_list))

def random_pick():
	rand_pick = np.random.permutation(range(len(gt_list)))[:len(gt_list)//10]
	sub_img_list = image_list[rand_pick]
	sub_gt_list = gt_list[rand_pick]
	# pass
	list(map(lambda x: shutil.move(join(crop_image_path, x), join(val_crop_image_path, x)), sub_img_list))
	list(map(lambda x: shutil.move(join(crop_gt_path, x), join(val_crop_gt_path, x)), sub_gt_list))

random_pick()