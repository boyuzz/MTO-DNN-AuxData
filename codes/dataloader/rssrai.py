from __future__ import print_function
from PIL import Image
import os
import numpy as np
from torchvision import transforms

import torch.utils.data as data
from dataloader import util


class RSSRAI(data.Dataset):
	"""

	Args:
		root (string): Root directory of dataset where directory
		train (bool, optional): If True, creates dataset from training set, otherwise
			creates from test set.
		transform (callable, optional): A function/transform that  takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		target_transform (callable, optional): A function/transform that takes in the
			target and transforms it.
		download (bool, optional): Currently not available. [If true, downloads the dataset from the internet and
			puts it in root directory. If dataset is already downloaded, it is not
			downloaded again.]

	"""

	# data_filename = "102flowers.tgz"
	id_filename = "ClsName2id.txt"

	# data_md5 = '52808999861908f626f3c1f4e79d11fa'
	# label_md5 = 'e0620be6f572b9609742df49c70aed4d'
	# datasplit_md5 = 'a5357ecc9cb78c4bef273ce3793fc85c'

	def __del__(self):
		if self.create_lmdb:
			self.env.close()

	def __init__(self, opt, train='train',
				 transform=None, target_transform=None,):
		self.root = os.path.expanduser(opt['dataroot'])
		self.transform = transform
		self.target_transform = target_transform
		self.train = train  # training set or test set
		self.create_lmdb = opt['lmdb']

		# if train == 'val':
		# 	self.transform.transforms.insert(0, transforms.FiveCrop)

		# im_list = glob.glob(os.path.join(self.root, 'jpg')+'/*.jpg')
		id_path = os.path.join(self.root, self.id_filename)
		data_path = os.path.join(self.root, self.train)

		label_dict = {}
		with open(id_path, 'r', encoding='utf-8-sig') as f:
			im_list = f.readlines()
			for line in im_list:
				record = line.split(':')
				label_dict[record[1]] = int(record[-1])

		self.data_label_dict = []

		basename = os.path.basename(data_path) + '.lmdb'
		lmdb_save_path = os.path.join(self.root, basename)

		if (self.create_lmdb and not os.path.exists(lmdb_save_path)) or not self.create_lmdb:
		# now load the image paths
			data_folders = os.listdir(data_path)
			self.sample_weight = []     # weighted samples for imbalanced dataset
			# if self.train != 'test':
			for folder in data_folders:
				# file = os.path.join(self.root, self.base_folder, f)
				path = os.path.join(data_path, folder)
				data_images = os.listdir(path)
				self.sample_weight.extend([1./len(data_images)]*len(data_images))
				data_images = [os.path.join(path, p) for p in data_images]
				label_id = label_dict[folder]
				labels = [label_id-1] * len(data_images)
				self.data_label_dict.extend(list(zip(data_images, labels)))

			if self.create_lmdb:
				util.create_lmdb(self.data_label_dict, lmdb_save_path, multiplier=100)
		else:
			self.env, self.data_label_dict = util._get_paths_from_lmdb(lmdb_save_path)
			# img = Image.open(os.path.join(self.root, fpath))
			# self.train_data.append(img)
			# self.train_labels.append(int(label_mat['labels'][0][im_id-1])-1)
		# else: # for testing set
		# 	# file = os.path.join(self.root, self.base_folder, f)
		# 	data_images = [os.path.join(data_path, p) for p in data_folders]
		# 	label_id = -1
		# 	labels = [label_id-1] * len(data_images)
		# 	self.data_label_dict.extend(list(zip(data_images, labels)))

		# if self.create_lmdb:
		# 	basename = os.path.basename(data_path) + '.lmdb'
		# 	lmdb_save_path = os.path.join(self.root, basename)
		# 	if not os.path.exists(lmdb_save_path):
		# 		util.create_lmdb(self.data_label_dict, lmdb_save_path, multiplier=100)
		#
		# 	self.env, self.data_label_dict = util._get_paths_from_lmdb(lmdb_save_path)

		# if opt['resample']:
		# 	re_index = np.random.randint(0, len(self.data_label_dict), len(self.data_label_dict))
		# 	self.data_label_dict = list(map(lambda x: self.data_label_dict[x], re_index))

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img_path, target = self.data_label_dict[index]

		if not self.create_lmdb:
			img = Image.open(img_path)
		else:
			img = util._read_lmdb_img(self.env, img_path)

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.data_label_dict)

	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		tmp = 'train' if self.train is True else 'test'
		fmt_str += '    Split: {}\n'.format(tmp)
		fmt_str += '    Root Location: {}\n'.format(self.root)
		tmp = '    Transforms (if any): '
		fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
		tmp = '    Target Transforms (if any): '
		fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
		return fmt_str
