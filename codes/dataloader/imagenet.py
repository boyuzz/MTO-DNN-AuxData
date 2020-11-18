from torchvision import datasets
import numpy as np
import os
from dataloader import util
from PIL import Image
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, DatasetFolder


class ImageFolder(DatasetFolder):
	"""A generic data loader where the images are arranged in this way: ::

		root/dog/xxx.png
		root/dog/xxy.png
		root/dog/xxz.png

		root/cat/123.png
		root/cat/nsdf3.png
		root/cat/asd932_.png

	Args:
		root (string): Root directory path.
		transform (callable, optional): A function/transform that  takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		target_transform (callable, optional): A function/transform that takes in the
			target and transforms it.
		loader (callable, optional): A function to load an image given its path.

	 Attributes:
		classes (list): List of the class names.
		class_to_idx (dict): Dict with items (class_name, class_index).
		samples (list): List of (image path, class_index) tuples
	"""
	def __init__(self, root, transform=None, target_transform=None,
				 loader=default_loader, resample=0, create_lmdb=False, lmdb_mul=1.):
		super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
										  transform=transform,
										  target_transform=target_transform)
		self.create_lmdb = create_lmdb

		if self.create_lmdb:
			basename = os.path.basename(self.root) + '.lmdb'
			lmdb_save_path = os.path.join(os.path.dirname(self.root), basename)
			if not os.path.exists(lmdb_save_path):
				util.create_lmdb(self.samples, lmdb_save_path, multiplier=lmdb_mul)

			self.env, self.samples = util._get_paths_from_lmdb(lmdb_save_path)

		# if resample:
		# 	# warnings.warn("warning! resample method based on image folder is different to others!")
		# 	re_index = np.random.randint(0, len(self.samples), int(len(self.samples)*resample))
		# 	# print(re_index[:20])
		# 	# self.samples = list(np.random.permutation(self.samples)[:len(self.samples)//2])
		# 	self.samples = list(map(lambda x: self.samples[x], re_index))

	def __getitem__(self, item):
		img_path, target = self.samples[item]
		if not self.create_lmdb:
			img = Image.open(img_path)
		# img = img_path
		else:
			img = util._read_lmdb_img(self.env, img_path)

		target = int(target)

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target


def ImageNet(opt, train=True, transform=None, target_transform=None):
	'''
	Load ImageNet dataset
	:param opt: datasets options
	:param train: (bool) if load training set
	:param transform: data augmentation
	:return: torch.datasets
	'''
	if train:
		dataset = ImageFolder(
			os.path.join(opt['dataroot'], 'train'),
			transform=transform, target_transform=target_transform, resample=opt['resample'],
			create_lmdb=opt['lmdb'], lmdb_mul=10)
	else:
		dataset = ImageFolder(
			os.path.join(opt['dataroot'], 'val'),
			transform=transform, target_transform=target_transform, resample=opt['resample'],
			create_lmdb=opt['lmdb'], lmdb_mul=10)

	return dataset


def AID(opt, train='train', transform=None, target_transform=None):
	# if train:
	dataset = ImageFolder(
		os.path.join(opt['dataroot'], train),
		transform=transform, target_transform=target_transform, resample=opt['resample'],
		create_lmdb=opt['lmdb'], lmdb_mul=12)
	# else:
	# 	dataset = ImageFolder(
	# 		os.path.join(opt['dataroot'], train), train,
	# 		transform=transform, target_transform=target_transform, resample=opt['resample'],
	# 		create_lmdb=opt['lmdb'], lmdb_mul=10)

	return dataset

def UCMerced(opt, train='train', transform=None, target_transform=None):
	# if train:
	dataset = ImageFolder(
		os.path.join(opt['dataroot'], train),
		transform=transform, target_transform=target_transform, resample=opt['resample'],
		create_lmdb=opt['lmdb'], lmdb_mul=1)
	# else:
	# 	dataset = ImageFolder(
	# 		os.path.join(opt['dataroot'], train), train,
	# 		transform=transform, target_transform=target_transform, resample=opt['resample'],
	# 		create_lmdb=opt['lmdb'], lmdb_mul=10)

	return dataset


def WHU19(opt, train='train', transform=None, target_transform=None):
	# if train:
	dataset = ImageFolder(
		os.path.join(opt['dataroot'], train),
		transform=transform, target_transform=target_transform, resample=opt['resample'],
		create_lmdb=opt['lmdb'], lmdb_mul=1)
	# else:
	# 	dataset = ImageFolder(
	# 		os.path.join(opt['dataroot'], 'val'), train,
	# 		transform=transform, target_transform=target_transform, resample=opt['resample'],
	# 		create_lmdb=opt['lmdb'], lmdb_mul=1)

	return dataset


def RSSCN7(opt, train='train', transform=None, target_transform=None):
	# if train:
	dataset = ImageFolder(
		os.path.join(opt['dataroot'], train),
		transform=transform, target_transform=target_transform, resample=opt['resample'],
		create_lmdb=opt['lmdb'], lmdb_mul=1.5)
	# else:
	# 	dataset = ImageFolder(
	# 		os.path.join(opt['dataroot'], 'val'), train,
	# 		transform=transform, target_transform=target_transform, resample=opt['resample'],
	# 		create_lmdb=opt['lmdb'], lmdb_mul=1)

	return dataset
