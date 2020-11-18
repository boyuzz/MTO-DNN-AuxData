import os
import numpy as np
from PIL import Image

from torch.utils import data

from dataloader import util
import torchvision.transforms.functional as TF
from torchvision import transforms


class INRIA(data.Dataset):
	colors = [  # [  0,   0,   0],
		0,
		255
	]

	label_colours = dict(zip(range(2), colors))
	n_classes = 2

	# mean_rgb = {
	#     "pascal": [103.939, 116.779, 123.68],
	#     "cityscapes": [0.0, 0.0, 0.0],
	# }  # pascal mean for PSPNet and ICNet pre-trained model
	np.random.rand()
	def __init__(self,
		opt,
		train=True,
		transform=False,
		target_transform=None,
		):
		"""__init__

		:param root:
		:param split:
		:param is_transform:
		:param img_size:
		:param augmentations
		"""
		self.root = os.path.expanduser(opt['dataroot'])
		self.transform = transform
		self.target_transform = target_transform
		self.train = train  # training set or test set
		self.create_lmdb = opt['lmdb']

		self.valid_classes = [0, 1]
		self.class_names = ["unlabelled", "building"]

		self.class_map = dict(zip(self.valid_classes, range(2)))
		# self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

		if self.train:
			self.images_base = os.path.join(self.root, "images")
			self.annotations_base = os.path.join(self.root, "gt")
			lmdb_base = '_val.lmdb'
		else:
			self.images_base = os.path.join(self.root, "images_val")
			self.annotations_base = os.path.join(self.root, "gt_val")
			lmdb_base = '_val.lmdb'

		image_list = []
		for (dirpath, dirnames, filenames) in os.walk(self.images_base):
			image_list.extend(filenames)
			break
		image_list = sorted(image_list)

		gt_list = []
		for (dirpath, dirnames, filenames) in os.walk(self.annotations_base):
			gt_list.extend(filenames)
			break
		gt_list = sorted(gt_list)

		self.data_label_dict = list(zip(image_list, gt_list))
		if self.create_lmdb:
			basename = os.path.basename(self.root) + lmdb_base
			lmdb_save_path = os.path.join(self.root, basename)
			if not os.path.exists(lmdb_save_path):
				util.create_lmdb(self.data_label_dict, lmdb_save_path, is_seg=True)
			self.env, self.data_label_dict = util._get_paths_from_lmdb(lmdb_save_path, is_seg=True)

		if self.train and opt['resample']:
			re_index = np.random.randint(0, len(self.data_label_dict), len(self.data_label_dict))
			self.data_label_dict = list(map(lambda x: self.data_label_dict[x], re_index))

		# self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

	def __len__(self):
		"""__len__"""
		return len(self.data_label_dict)

	def __getitem__(self, index):
		"""__getitem__

		:param index:
		"""
		img_path, target_path = self.data_label_dict[index]
		assert img_path == target_path or target_path == -1, "{} is not the same as {}".format(img_path, target_path)

		if not self.create_lmdb:
			img = Image.open(os.path.join(self.images_base, img_path))
			target = Image.open(os.path.join(self.annotations_base, target_path))
		else:
			img, target = util._read_lmdb_img(self.env, img_path, is_seg=True)
		# img.save('test.png')
		# target.save('label.png')
		target = self.encode_segmap(np.asarray(target, dtype=np.uint8))

		img, target = self.pair_transform(img, target)

		return img, target

	def pair_transform(self, img, target):
		# Random horizontal flipping

		if self.train and self.transform:
			if np.random.random() > 0.5:
				img = TF.hflip(img)
				target = TF.hflip(target)

			# Random vertical flipping
			if np.random.random() > 0.5:
				img = TF.vflip(img)
				target = TF.vflip(target)

			# Random Resized Crop
			i, j, h, w = transforms.RandomResizedCrop.get_params(
				img, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
			img = TF.resized_crop(img, i, j, h, w, (768, 768), Image.BICUBIC)
			target = TF.resized_crop(target, i, j, h, w, (768, 768), Image.BICUBIC)

			# img = TF.center_crop(img, (256, 256))
			# target = TF.center_crop(target, (256, 256))

			# Gamma transformation
			# if np.random.random() > 0.5:
			# 	gamma = 0.25 + 1.75 * np.random.random()
			# 	img = TF.adjust_gamma(img, gamma)

			# Hue transformation
			if np.random.random() > 0.5:
				hue_factor = np.random.random() - 0.5
				img = TF.adjust_hue(img, hue_factor)
		else:
			img = TF.center_crop(img, (1440, 1440))
			target = TF.center_crop(target, (1440, 1440))

		img = TF.to_tensor(img)
		img = TF.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
		# img = TF.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		target = TF.to_tensor(target).float()

		# im_dem = TF.to_pil_image(img)
		# target_dem = TF.to_pil_image(target)
		# im_dem.save('img.png')
		# target_dem.save('gt.png')
		# target = target.unsqueeze(0)
		return img, target

	@classmethod
	def decode_segmap(self, temp):
		r = temp.copy()
		for l in range(0, self.n_classes):
			r[temp == l] = self.label_colours[l]
		# r = r/255.
		return r.squeeze()

	@classmethod
	def encode_segmap(self, mask):
		# Put all void classes to zero
		label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int8)
		for ii, label in enumerate(self.colors):
			label_mask[mask==label] = ii
		label_mask = Image.fromarray(label_mask)
		return label_mask


if __name__ == "__main__":
	import matplotlib.pyplot as plt

	# augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])

	local_path = "/datasets01/cityscapes/112817/"
	dst = cityscapesLoader(local_path, transform=None)
	bs = 4
	trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
	for i, data_samples in enumerate(trainloader):
		imgs, labels = data_samples
		import pdb

		pdb.set_trace()
		imgs = imgs.numpy()[:, ::-1, :, :]
		imgs = np.transpose(imgs, [0, 2, 3, 1])
		f, axarr = plt.subplots(bs, 2)
		for j in range(bs):
			axarr[j][0].imshow(imgs[j])
			axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
		plt.show()
		a = input()
		if a == "ex":
			break
		else:
			plt.close()
