import pandas as pd
import os
import torch
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class KDDDataset(data.Dataset):
	def __init__(self, opt, train=True):
		super(KDDDataset, self).__init__()
		path = os.path.join(opt['dataroot'], 'kddcup.data_10_percent_corrected')
		type_path = os.path.join(opt['dataroot'], 'kddcup.names')
		self.is_norm = opt['is_norm']

		title = []
		with open(type_path, 'r') as fp:
			files = fp.readlines()
			for line in files:
				if ':' in line:
					title.append(line.split(':')[0])
		title.append('label')

		self.data = pd.read_csv(path, names=title)
		number = LabelEncoder()
		# 检测每一列属性
		for name, t in self.data.dtypes.items():
			if not np.issubdtype(t, np.number):
				self.data[name] = number.fit_transform(self.data[name].astype('str'))

		self.data = self.data.astype(float)

		# print(self.data.dtypes)
		if self.is_norm:
			self.label_min = self.data['label'].min()
			self.label_max = self.data['label'].max()
			x = self.data.values
			self.min_max_scaler = MinMaxScaler()
			x_scaled = self.min_max_scaler.fit_transform(x)
			self.data = pd.DataFrame(x_scaled)

	def __getitem__(self, index):
		return torch.tensor(np.array(self.data.loc[index]), dtype=torch.float32)  #, self.label[index]

	def __len__(self):
		return len(self.data)

	def get_data_dim(self):
		return len(self.data.columns)

	def reverse_norm(self, labels):
		if self.is_norm:
			labels = (labels*(self.label_max-self.label_min) + self.label_min).astype(int)
		return labels


class KDDLoader:
	def __init__(self, config=None):
		super(KDDLoader, self).__init__()
		self.config = config
		self.train_set = KDDDataset(self.config)

	def get_train_data(self):
		train_loader = DataLoader(dataset=self.train_set, num_workers=self.config['threads'],
								  batch_size=self.config['batch_size'], shuffle=True)
		return train_loader

	def get_data_dim(self):
		return self.train_set.get_data_dim()