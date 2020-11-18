#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import os
import glob
from cv2 import imread, imwrite
from skimage import io
from utils.segmentation.metrics import runningScore

image_path = "/home/piri/programs/denseCRF/images" #验证集RGB图片路径
anno_path = "/home/piri/programs/denseCRF/anno"    #验证集分割后图片路径
out_path='/home/piri/programs/denseCRF/crf_1/'     #经过crf处理后的图片的保存路径
gt_path='/home/piri/programs/denseCRF/gt/'         #验证集的groud truth路径

#可调节的参数
use_2d=True											#True对应Using 2D specialized functions，False对应Using tamen  2D functions
gt_prob=0.9											#当前分割结果的可信度。gt_prob越小，crf的结果变化越大
Gaussian_sxy=(3,3)
Gaussian_compat=3
Gaussian_kernel=dcrf.DIAG_KERNEL					#可选：CONST_KERNEL，DIAG_KERNEL (the default)，FULL_KERNEL
Gaussion_norm=dcrf.NORMALIZE_SYMMETRIC,			#可选：NO_NORMALIZATION，NORMALIZE_BEFORE，NORMALIZE_AFTER，NORMALIZE_SYMMETRIC (the default)
Bilateral_sxy=(20,20)
Bilateral_srgb=(13, 13, 13)
Bilateral_compat=3
Bilateral_kernel=dcrf.DIAG_KERNEL
Bilateral_norm=dcrf.NORMALIZE_SYMMETRIC
Bilateral_chidim=2
infer_times=5
random_times=5

image_list=glob.glob(os.path.join(image_path,"*.png"))
anno_list=glob.glob(os.path.join(anno_path,"*.png"))

image_list.sort()
anno_list.sort()

nameB=[]
for paths in anno_list:
	name,_=os.path.splitext(os.path.basename(paths))
	nameB.append(name)

def zerone(mat,thresh):
	mat[mat<thresh]=0
	mat[mat>=thresh]=1
	return mat

for j in range(len(image_list)):

	fn_im=image_list[j]
	fn_anno=anno_list[j]
	fn_output=out_path+nameB[j]+'.png'

	##################################
	### Read images and annotation ###
	##################################
	img = imread(fn_im)
	anno = imread(fn_anno)

	anno_rgb = zerone(imread(fn_anno), 127).astype(np.uint32)
	anno_rgb = anno_rgb[:, :, 0]

	n_labels = len(set(anno_rgb.flat))
	#print(n_labels, " labels", set(anno_rgb.flat))

	###########################
	### Setup the CRF model ###
	###########################
	#use_2d = True
	# use_2d = True
	if use_2d:
		#print("Using 2D specialized functions")

		# Example using the DenseCRF2D code
		d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

		# get unary potentials (neg log probability)
		U = unary_from_labels(anno_rgb, n_labels, gt_prob=gt_prob, zero_unsure=False)
		d.setUnaryEnergy(U)

		# This adds the color-independent term, features are the locations only.
		d.addPairwiseGaussian(sxy=Gaussian_sxy, compat=Gaussian_compat, kernel=Gaussian_kernel,
		                      normalization=Gaussion_norm)

		# This adds the color-dependent term, i.e. features are (x,y,r,g,b).
		d.addPairwiseBilateral(sxy=Bilateral_sxy, srgb=Bilateral_srgb, rgbim=img,
		                       compat=Bilateral_compat,
		                       kernel=Bilateral_kernel,
		                       normalization=Bilateral_kernel)
	else:
		print("Using tamen  2D functions")

		# Example using the DenseCRF class and the util functions
		d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

		# get unary potentials (neg log probability)
		U = unary_from_labels(anno_rgb, n_labels, gt_prob=gt_prob, zero_unsure=False)
		d.setUnaryEnergy(U)

		# This creates the color-independent features and then add them to the CRF
		feats = create_pairwise_gaussian(sdims=Gaussian_sxy, shape=img.shape[:2])
		d.addPairwiseEnergy(feats, compat=Gaussian_compat,
		                    kernel=Gaussian_kernel,
		                    normalization=Gaussion_norm)

		# This creates the color-dependent features and then add them to the CRF
		feats = create_pairwise_bilateral(sdims=Bilateral_sxy, schan=Bilateral_srgb,
		                                  img=img, chdim=Bilateral_chidim)
		d.addPairwiseEnergy(feats, compat=Bilateral_compat,
		                    kernel=Bilateral_kernel,
		                    normalization=Bilateral_norm)
	####################################
	### Do inference and compute MAP ###
	####################################

	# Run five inference steps.
	Q = d.inference(infer_times)

	# Find out the most probable class for each pixel.
	MAP = np.argmax(Q, axis=0)
	# print 'before',MAP.shape
	# Convert the MAP (labels) back to the corresponding colors and save the image.
	# Note that there is no "unknown" here anymore, no matter what we had at first.
	# MAP = colorize[MAP,:]
	# print 'after:',MAP.shape
	MAP[MAP == 1] = 255
	imwrite(fn_output, MAP.reshape(img.shape[:2]))

	# Just randomly manually run inference iterations
	Q, tmp1, tmp2 = d.startInference()
	for i in range(random_times):
		#print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
		d.stepInference(Q, tmp1, tmp2)

	print('the'+' ' +str(j+1)+'th'+' picture is being generated~')

print('========crf process has been completed!!!=========')

image_paths_A=glob.glob(os.path.join(out_path,"*.png"))
image_paths_B=glob.glob(os.path.join(gt_path,"*.png"))
running_metrics = runningScore(2)

for im_path, gt_path in zip(image_paths_A, image_paths_B):
	img = io.imread(im_path)
	gt = io.imread(gt_path)
	running_metrics.update(gt, img)

score, cls_iou = running_metrics.get_scores()
print(score, cls_iou)

# f=open(out_path+'result.txt','w')
# f.write('====config:====='+'\n')
# f.write('use_2d='+str(use_2d)+'\n')
# f.write('gt_prob='+str(gt_prob)+'\n')
# f.write('Gaussian_sxy='+str(Gaussian_sxy)+'\n')
# f.write('Gaussian_compat='+str(Gaussian_compat)+'\n')
# f.write('Gaussian_kernel='+str(Gaussian_kernel)+'\n')
# f.write('Gaussion_norm='+str(Gaussion_norm)+'\n')
# f.write('Bilateral_sxy='+str(Bilateral_sxy)+'\n')
# f.write('Bilateral_srgb='+str(Bilateral_srgb)+'\n')
# f.write('Bilateral_compat='+str(Bilateral_compat)+'\n')
# f.write('Bilateral_kernel='+str(Bilateral_kernel)+'\n')
# f.write('Bilateral_norm='+str(Bilateral_norm)+'\n')
# f.write('Bilateral_chidim='+str(Bilateral_chidim)+'\n')
# f.write('infer_times='+str(infer_times)+'\n')
# f.write('random_times='+str(random_times)+'\n')
# f.write('====result:====='+'\n')
# f.write('IoU:'+str(iou)+'\n')
# f.write(('accuracy:'+str(acc)))
# f.close()

