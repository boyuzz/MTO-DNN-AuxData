# -- coding: utf-8 --
"""
-------------------------------------------------
   File Name：     full_loader
   Description :
   Author :       zhangboyu
   date：          25/9/19
-------------------------------------------------
   Change Activity:
                   25/9/19:
-------------------------------------------------
"""
__author__ = 'zhangboyu'

import numpy as np
import os
from PIL import Image
from collections import Counter
from random import shuffle
import torch.utils.data as data
import csv

class FullSet(data.Dataset):
	def __init__(self, opt, train='train', preload=False, transform=None, target_transform=None, resample=0):
		self.root = os.path.expanduser(opt['dataroot'])
		self.data_dicts = []
		self.mode = train
		self.preload = preload
		self.transform = transform
		self.target_transform = target_transform
		self.resample = resample

		# opt['run'] here is the same as seed. 5-fold CV here.
		csv_path = os.path.join(self.root, '{}_fold{}.csv'.format(train, opt['run']))

		data_label_dict = self.parse_csv(csv_path)
		self.samples = list(
			map(lambda x: (os.path.join(self.root, x[0]), x[1]), data_label_dict))

		if self.preload:
			for img_path, target in self.samples:
				temp = Image.open(img_path)
				keep = temp.copy()
				# img = io.imread(img_path)
				# img = Image.fromarray(img)
				self.data_dicts.append((keep, target))
				temp.close()

			if resample and self.mode == 'train':
				X, y = zip(*self.data_dicts)
				X = list(X)
				y = list(y)

				for i, data in enumerate(X):
					X[i] = np.array(data)
				X = np.array(X)
				y = np.array(y)
				X, y = self.mixup_oversampling(X, y, resample)

				# re_index = np.random.randint(0, len(self.samples), int(len(self.samples)*resample))
				# shuffle(self.data_dicts)
				# self.data_dicts = self.data_dicts[:len(self.samples)]
				# self.samples = list(map(lambda x: self.samples[x], re_index))
				self.data_dicts = list(zip(X, y))
				for i in range(len(self.data_dicts)):
					bundle = list(self.data_dicts[i])
					bundle[0] = Image.fromarray(bundle[0].astype(np.uint8))
					self.data_dicts[i] = tuple(bundle)
		else:
			self.data_dicts = self.samples
			if resample and self.mode == 'train':
				label_list = np.array(list(zip(*self.samples))[1])
				real_select = np.random.permutation(len(label_list))[:int(len(label_list) // resample)]
				self.data_dicts = [p for i, p in enumerate(self.data_dicts) if i in real_select]

				label_counts = Counter(label_list)
				keys = label_counts.keys()
				for k in keys:
					indexes = np.where(label_list == k)[0]
					remain = int(len(indexes) * (resample - 1) / resample)
					for i in range(remain):
						picked = np.random.choice(indexes, 2, replace=False)
						path1 = self.samples[picked[0]][0]
						path2 = self.samples[picked[1]][0]
						self.data_dicts.append(((path1, path2), k))
				shuffle(self.data_dicts)

	def parse_csv(self, path):
		label_dict = []
		with open(path, 'r') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',')
			for row in spamreader:
				if '\\' in row[0]:
					row[0] = row[0].replace('\\', '/')
				label_dict.append(row)
		return label_dict

	def mixup(self, x1, x2):
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

		return x1[w1_s:w1_s+w, h1_s:h1_s+h, :]*a1+x2[w2_s:w2_s+w, h2_s:h2_s+h, :]*a2

	def mixup_oversampling(self, X, y, ratio):
		label_counts = Counter(y)
		keys = label_counts.keys()
		X_res = []
		y_res = []
		for k in keys:
			indexes = np.where(y==k)[0]
			remain = int(len(indexes)*(ratio - 1)/ratio)
			for i in range(remain):
				picked = np.random.choice(indexes, 2, replace=False)
				x1 = X[picked[0]]
				x2 = X[picked[1]]
				X_res.append(self.mixup(x1, x2))
				y_res.append(k)
		X_res = np.array(X_res)
		y_res = np.array(y_res)
		real_select = np.random.permutation(len(y))[:int(len(y)//ratio)]   #
		# fake_select = np.random.permutation(len(y_res))[:int(len(y_res)*(1-1/ratio))]   #

		X_res = np.concatenate((X[real_select], X_res), axis=0)
		y_res = np.concatenate((y[real_select], y_res), axis=0)
		return X_res, y_res

	def __getitem__(self, item):
		img, target = self.data_dicts[item]
		target = int(target)

		if not self.preload:
			if isinstance(img, str):
				img = Image.open(img)
			elif isinstance(img, tuple):
				img1 = Image.open(img[0])
				img2 = Image.open(img[1])
				img = Image.fromarray(self.mixup(np.asarray(img1), np.asarray(img2)).astype(np.uint8))

			# if self.resample and np.random.rand() > 1/self.resample:
			# 	ran_idx = np.random.choice(np.where(self.label_list == str(target))[0])
			# 	assert str(target) == self.data_dicts[ran_idx][1]
			# 	img2_path = self.data_dicts[ran_idx][0]
			# 	try:
			# 		img2 = Image.open(img2_path)
			# 	except OSError:
			# 		print(img2_path)
			# 	img = Image.fromarray(self.mixup(np.asarray(img), np.asarray(img2)).astype(np.uint8))

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.data_dicts)


def AID(opt, train='train', transform=None, target_transform=None):
	dataset = FullSet(
		opt, train, preload=False,
		transform=transform, target_transform=target_transform, resample=opt['resample'])
	return dataset

def UCMerced(opt, train='train', transform=None, target_transform=None):
	dataset = FullSet(
		opt, train,
		transform=transform, target_transform=target_transform, resample=opt['resample'])
	return dataset


def WHU19(opt, train='train', transform=None, target_transform=None):
	dataset = FullSet(
		opt, train,
		transform=transform, target_transform=target_transform, resample=opt['resample'])
	return dataset


def RSSCN7(opt, train='train', transform=None, target_transform=None):
	dataset = FullSet(
		opt, train, preload=True,
		transform=transform, target_transform=target_transform, resample=opt['resample'])
	return dataset