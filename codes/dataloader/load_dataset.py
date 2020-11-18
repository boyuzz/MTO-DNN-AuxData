from torchvision import datasets
from torchvision import transforms
import dataloader


# import horovod.torch as hvd
# import torch
def torchvisionDataset(opt, train=False):
	'''
	Load the existing datasets from torchvision
	'''
	transform_op = [transforms.ToTensor()]
	if 'mnist' in opt['name'].lower():
		transform_op.append(transforms.Normalize((0.1307,), (0.3081,)))
		if train:
			transform_op.insert(0, transforms.RandomCrop(28, padding=4))

	if 'cifar' in opt['name'].lower():
		transform_op.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
		if train:
			transform_op.insert(0, transforms.RandomHorizontalFlip())
			transform_op.insert(0, transforms.RandomCrop(32, padding=4))

	transform = transforms.Compose(transform_op)
	train_dataset = getattr(datasets, opt['name'])(opt['dataroot'], train=train, download=opt['download'],
	                                               transform=transform)
	# if hvd.rank() == 0:
	# 	train_dataset = getattr(datasets, opt['name'])(opt['dataroot'], train=opt['is_train'], download=opt['download'],
	# 												   transform=transform)
	# hvd.broadcast(torch.zeros(1), root_rank=0, name="Barrier")
	# if hvd.rank() != 0:
	# 	train_dataset = getattr(datasets, opt['name'])(opt['dataroot'], train=opt['is_train'], download=False,
	# 	                                               transform=transform)
	# hvd.allreduce([], name="Barrier")

	return train_dataset


def fileDataset(opt, split='train'):
	transform_op = [transforms.ToTensor()]
	# transform_op = [transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))]

	if 'mnist' in opt['name'].lower():
		transform_op.append(transforms.Normalize((0.1307,), (0.3081,)))
		if split == 'train':
			transform_op.insert(0, transforms.RandomCrop(28, padding=4))
	elif 'cifar' in opt['name'].lower():
		transform_op.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
		if split == 'train':
			transform_op.insert(0, transforms.RandomHorizontalFlip())
			transform_op.insert(0, transforms.RandomCrop(32, padding=4))
	elif 'oxfordflowers' in opt['name'].lower():
		transform_op.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
		transform_op.insert(0, transforms.CenterCrop(224))
		transform_op.insert(0, transforms.Resize(256))
		if split == 'train':
			transform_op.insert(0, transforms.RandomHorizontalFlip())
			transform_op.insert(0, transforms.RandomVerticalFlip())
	elif opt['name'].lower() in ['rsscn7', 'aid', 'whu19', 'ucmerced', 'rssrai', 'isic', 'imagenet', 'oxfordpets', 'stanfordcar', 'fgvcaircraft', 'imagenetflowers', 'stanforddogs']:
	# if 'rsscn7' in opt['name'].lower() or 'isic' in opt['name'].lower() or 'imagenet' == opt['name'].lower():
		transform_op.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
		# transform_op.append(transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(crop) for crop in crops])))
		if split == 'train':
			# TODO: remove flip augmentation!!! Why??
			transform_op.insert(0, transforms.RandomHorizontalFlip())
			transform_op.insert(0, transforms.RandomResizedCrop(224))
		else:
			transform_op.insert(0, transforms.CenterCrop(224))
			# transforms.FiveCrop
			# transform_op.insert(0, transforms.FiveCrop(224))
			transform_op.insert(0, transforms.Resize(256))
	else:
		transform_op.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
		transform_op.insert(0, transforms.Resize(256))

	if 'imagenet32' in opt['name'].lower():
		transform_op.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
		if split == 'train':
			transform_op.insert(0, transforms.RandomHorizontalFlip())
			transform_op.insert(0, transforms.RandomCrop(32, padding=4))

	# transform_op = [transforms.ToTensor()]
	if 'imagenet' in opt['name'].lower() and 'imagenet32' not in opt['name'].lower():
		transform = transforms.Compose(transform_op)
		train_dataset = getattr(dataloader, 'ImageNet')(opt, train=split, transform=transform)
	elif opt['name'].lower() in ['inria']:
		train_dataset = getattr(dataloader, opt['name'])(opt, train=split, transform=True)
	else:
		transform = transforms.Compose(transform_op)
		train_dataset = getattr(dataloader, opt['name'])(opt, train=split, transform=transform)

	return train_dataset


def imgenDataset(opt, split='train'):
	transform_op = [transforms.Resize(64),
	                transforms.RandomHorizontalFlip(),
	                transforms.RandomResizedCrop(64),
	                transforms.ToTensor(),
	                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
	transform = transforms.Compose(transform_op)
	train_dataset = getattr(dataloader, 'ImgReader')(opt, transform=transform)
	return train_dataset
# def numericDataset(opt, train=False):
#
# 	return train_dataset