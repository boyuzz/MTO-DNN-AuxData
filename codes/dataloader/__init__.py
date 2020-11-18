'''create dataset and dataloader'''
from .oxfordflowers import OxfordFlowers
from .imagenet import ImageNet, AID, WHU19, RSSCN7, UCMerced
# from .full_loader import AID, WHU19,RSSCN7, UCMerced
from .imagenet32 import ImageNet32
# from .rsscn7 import RSSCN7
from .isic import ISIC
from .cifar import CIFAR10, CIFAR100
from .mnist import MNIST, FashionMNIST, EMNIST
from .oxfordpets import OxfordPets
from .stanfordcar import StanfordCar
from .fgvcaircraft import FgvcAircraft
from .stanforddogs import StanfordDogs
from .inria import INRIA
from .rssrai import RSSRAI
from .imgReader import ImgReader

import logging
import torch.utils.data
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler


def create_dataloader(dataset, dataset_opt):
	'''create dataloader '''
	if 'imbalance' in dataset_opt.keys() and dataset_opt['imbalance'] == True:
	# if dataset_opt['resample'] == True:
		samples_weight = np.zeros(len(dataset))
		samples_weight[:int(0.8*len(dataset))] = 1
		samples_weight = np.random.permutation(samples_weight)
		return torch.utils.data.DataLoader(
			dataset,
			sampler=WeightedRandomSampler(samples_weight, len(samples_weight)),
			batch_size=dataset_opt['batch_size'],
			# shuffle=dataset_opt['use_shuffle'],
			num_workers=dataset_opt['n_workers'],
			drop_last=False,
			pin_memory=True)
	else:
		return torch.utils.data.DataLoader(
			dataset,
			batch_size=dataset_opt['batch_size'],
			shuffle=dataset_opt['use_shuffle'],
			num_workers=dataset_opt['n_workers'],
			drop_last=False,
			pin_memory=True)


def create_dataset(dataset_opt, split='train'):
	'''create dataset'''
	mode = dataset_opt['mode']
	if mode == 'file':
		from dataloader.load_dataset import fileDataset as D
	elif mode == 'numeric':
		from dataloader.uciloader import UCIDataset as D
	elif mode == 'table':
		from dataloader.kddloader import KDDDataset as D
	elif mode == 'imgen':
		from dataloader.load_dataset import imgenDataset as D
	else:
		raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
	dataset = D(dataset_opt, split)
	logger = logging.getLogger('base')
	logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
														   dataset_opt['name']))
	return dataset
