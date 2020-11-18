from __future__ import print_function
from PIL import Image
from scipy import io
import os
import numpy as np

import torch.utils.data as data
from dataloader import util
from torchvision.datasets.utils import download_url, check_integrity


class RSSCN7(data.Dataset):
	"""`RSSCN7 <https://www.dropbox.com/s/j80iv1a0mvhonsa/RSSCN7.zip?dl=0>`_ Dataset.

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
	# data_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
	# label_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
	# datasplit_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

	# data_filename = "102flowers.tgz"
	id_filename = "id_files.txt"
	label_filename = "labels.mat"
	tr_filename = "tr_idx.mat"
	ts_filename = "ts_idx.mat"

	# data_md5 = '52808999861908f626f3c1f4e79d11fa'
	# label_md5 = 'e0620be6f572b9609742df49c70aed4d'
	# datasplit_md5 = 'a5357ecc9cb78c4bef273ce3793fc85c'

	def __del__(self):
		if self.create_lmdb:
			self.env.close()

	def __init__(self, opt, train=True,
				 transform=None, target_transform=None,):
		self.root = os.path.expanduser(opt['dataroot'])
		self.transform = transform
		self.target_transform = target_transform
		self.train = train  # training set or test set
		self.create_lmdb = opt['lmdb']

		# if download:
		# 	self.download()

		# if not self._check_integrity():
		# 	raise RuntimeError('Dataset not found or corrupted.' +
		# 					   ' You can use download=True to download it')

		# im_list = glob.glob(os.path.join(self.root, 'jpg')+'/*.jpg')
		id_path = os.path.join(self.root, self.id_filename)
		label_path = os.path.join(self.root, self.label_filename)
		tr_path = os.path.join(self.root, self.tr_filename)
		ts_path = os.path.join(self.root, self.ts_filename)

		label_mat = io.loadmat(label_path)
		tr_mat = io.loadmat(tr_path)
		ts_mat = io.loadmat(ts_path)
		with open(id_path, 'r') as f:
			im_list = f.readlines()

		self.data_label_dict = []

		# now load the picked numpy arrays
		if self.train:
			for im_id in tr_mat['tr_idx'][0]:
				# file = os.path.join(self.root, self.base_folder, f)
				fpath = im_list[im_id-1].strip('\n').strip('../')
				self.data_label_dict.append((os.path.join(self.root, fpath), int(label_mat['labels'][0][im_id - 1]) - 1))
				# img = Image.open(os.path.join(self.root, fpath))
				# self.train_data.append(img)
				# self.train_labels.append(int(label_mat['labels'][0][im_id-1])-1)
			if self.create_lmdb:
				basename = os.path.basename(self.root)+'_train.lmdb'
				lmdb_save_path = os.path.join(self.root, basename)
				if not os.path.exists(lmdb_save_path):
					util.create_lmdb(self.data_label_dict, lmdb_save_path)

				self.env, self.data_label_dict = util._get_paths_from_lmdb(lmdb_save_path)
			if opt['resample']:
				re_index = np.random.randint(0, len(self.data_label_dict), len(self.data_label_dict))
				self.data_label_dict = list(map(lambda x: self.data_label_dict[x], re_index))
		else:
			for im_id in ts_mat['ts_idx'][0]:
				# file = os.path.join(self.root, self.base_folder, f)
				fpath = im_list[im_id - 1].strip('\n').strip('../')
				self.data_label_dict.append((os.path.join(self.root, fpath), int(label_mat['labels'][0][im_id - 1]) - 1))
				# img = Image.open(os.path.join(self.root, fpath))
				# self.test_data.append(img)
				# self.test_labels.append(int(label_mat['labels'][0][im_id-1])-1)
			if self.create_lmdb:
				basename = os.path.basename(self.root)+'_val.lmdb'
				lmdb_save_path = os.path.join(self.root, basename)
				if not os.path.exists(lmdb_save_path):
					util.create_lmdb(self.data_label_dict, lmdb_save_path)

				self.env, self.data_label_dict = util._get_paths_from_lmdb(lmdb_save_path)

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
			# img, target = self.test_data[index], self.test_labels[index]

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.data_label_dict)

	# def _check_integrity(self):
	# 	root = self.root
	# 	check_list = [(self.data_filename, self.data_md5), (self.datasplit_filename, self.datasplit_md5),
	# 				  (self.label_filename, self.label_md5)]
	#
	# 	for file, md5 in check_list:
	# 		fpath = os.path.join(root, file)
	# 		if not check_integrity(fpath, md5):
	# 			return False
	#
	# 	return True
	#
	# def download(self):
	# 	import tarfile
	#
	# 	if self._check_integrity():
	# 		print('Files already downloaded and verified')
	# 		return
	#
	# 	root = self.root
	# 	download_url(self.data_url, root, self.data_filename, self.data_md5)
	# 	download_url(self.label_url, root, self.label_filename, self.label_md5)
	# 	download_url(self.datasplit_url, root, self.datasplit_filename, self.datasplit_md5)
	#
	# 	# extract file
	# 	cwd = os.getcwd()
	# 	tar = tarfile.open(os.path.join(root, self.data_filename), "r:gz")
	# 	os.chdir(root)
	# 	tar.extractall()
	# 	tar.close()
	# 	os.chdir(cwd)

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


# class OxfordFlower_test(OxfordFlower):
# 	"""`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
#
# 	This is a subclass of the `OxfordFlower` Dataset.
# 	"""
# 	base_folder = 'cifar-100-python'
# 	url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
# 	filename = "cifar-100-python.tar.gz"
# 	tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
# 	train_list = [
# 		['train', '16019d7e3df5f24257cddd939b257f8d'],
# 	]
#
# 	test_list = [
# 		['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
# 	]
