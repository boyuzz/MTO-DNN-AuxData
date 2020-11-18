import os
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='D:/data/AerialImageDataset/INRIA/images_val')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='D:/data/AerialImageDataset/INRIA/pred')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='D:/data/AerialImageDataset/INRIA/val_mask')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)', action='store_true')
args = parser.parse_args()

for arg in vars(args):
	print('[%s] = ' % arg, getattr(args, arg))

# splits = os.listdir(args.fold_A)

# for sp in splits:
img_fold_A = args.fold_A
img_fold_B = args.fold_B
img_list = os.listdir(img_fold_A)
if args.use_AB:
	img_list = [img_path for img_path in img_list if '_A.' in img_path]

num_imgs = min(args.num_imgs, len(img_list))
print('split, use %d/%d images' % ( num_imgs, len(img_list)))
img_fold_AB = os.path.join(args.fold_AB)
if not os.path.isdir(img_fold_AB):
	os.makedirs(img_fold_AB)
print('split, number of images = %d' % (num_imgs))
for n in range(num_imgs):
	name_A = img_list[n]
	path_A = os.path.join(img_fold_A, name_A)
	if args.use_AB:
		name_B = name_A.replace('_A.', '_B.')
	else:
		fname = name_A.split('.')[0]
		name_B = fname+'.png'
	path_B = os.path.join(img_fold_B, name_B)
	if os.path.isfile(path_A) and os.path.isfile(path_B):
		name_AB = name_A
		if args.use_AB:
			name_AB = name_AB.replace('_A.', '.')  # remove _A
		path_AB = os.path.join(img_fold_AB, name_AB)
		im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
		im_B = cv2.imread(path_B, 0) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
		# mask = cv2.inRange(im_B, np.array([1,1,1]), np.array([255, 255, 255]))
		# cv2.imshow('Mask', mask)
		im_B = np.expand_dims(im_B, 2)
		B = G = np.zeros(im_B.shape).astype(im_B.dtype)
		im_B = np.concatenate((B, G, im_B), axis=2)
		# for i in range(im_B.shape[0]):
		# 	for j in range(im_B.shape[1]):
		# 		if (im_B[i, j] == [255, 255, 255]).all():  # 0代表黑色的点
		# 			im_B[i, j] = [0,0,255]
		# im_AB = np.concatenate([im_A, im_B], 1)
		im_A = im_A.astype(np.uint16)
		im_A += (im_B*0.3).astype(np.uint16)
		im_A[im_A>255] = 255
		im_A = im_A.astype(np.uint8)
		cv2.imwrite(path_AB, im_A)
