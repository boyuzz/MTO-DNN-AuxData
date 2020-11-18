# -- coding: utf-8 --
"""
-------------------------------------------------
   File Name：     dataset
   Description :
   Author :       zhangboyu
   date：          20/11/18
-------------------------------------------------
   Change Activity:
				   20/11/18:
-------------------------------------------------
"""
__author__ = 'zhangboyu'

from torch.utils import data
import torch
import numpy as np
import os
from torch.utils.data import DataLoader


class UCIDataset(data.Dataset):
	def __init__(self, opt, train=True):
		super(UCIDataset, self).__init__()
		self.data = []
		path = os.path.join(opt['dataroot'], 'data.dat')
		type_path = os.path.join(opt['dataroot'], 'type')
		self.is_norm = opt['is_norm']

		# self.label = []
		with open(path, 'r') as f:
			article = f.readlines()
			for line in article:
				dataline = line.split()
				self.data.append(np.array([float(value) for value in dataline]))
				# self.label.append(dataline[-1])

		if type_path is not None:
			with open(type_path, 'r') as f:
				line = f.readlines()[0]
				dataline = line.split(',')
				self.data_type = np.array([int(value) for value in dataline])

		np_data = np.array(self.data)

		if self.is_norm:
			self.d_min = np.min(np_data, axis=0)
			self.d_max = np.max(np_data, axis=0)
			self.data = (np_data-self.d_min)/(self.d_max-self.d_min)

		self.data = torch.from_numpy(self.data.astype(np.float32))
		if opt['resample']:
			re_index = np.random.randint(0, len(self.data), len(self.data))
			self.data = list(map(lambda x: self.data[x], re_index))

	def __getitem__(self, index):
		return self.data[index]  #, self.label[index]

	def __len__(self):
		return len(self.data)

	def get_data_dim(self):
		assert len(self.data) > 0
		return len(self.data[0])

	def reverse_norm(self, sample):
		if self.is_norm:
			sample = np.around(sample * (self.d_max-self.d_min) + self.d_min, 2)
		return sample

	def round_data(self, sample):
		for i in range(len(sample)):
			sample[i] = np.round(sample[i], self.data_type[i])
		return sample


class UCILoader:
	def __init__(self, config=None):
		super(UCILoader, self).__init__()
		self.config = config
		self.train_set = UCIDataset(self.config)

	def get_train_data(self):
		train_loader = DataLoader(dataset=self.train_set, num_workers=self.config['threads'],
								  batch_size=self.config['batch_size'], shuffle=True)
		return train_loader

	def get_data_dim(self):
		return self.train_set.get_data_dim()
