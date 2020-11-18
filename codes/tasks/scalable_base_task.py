import os
import torch
import torch.nn as nn
import numpy as np
from dataloader import create_dataloader, create_dataset
import math
# import random
# from utils import util
from prefetch_generator import BackgroundGenerator, background
from options import options


class BaseTask:
	def __init__(self, opt, logger):
		self.opt = opt
		self.is_train = opt['is_train']
		self.schedulers = []
		self.optimizers = []
		self.training_step = 0
		self.start_epoch = 0
		self.task_id = opt['task_id']
		self.device_per_task = opt['device_per_task']
		self.logger = logger

		self.seed = self.opt['manual_seed']
		# if self.seed is None or self.task_id != 0:
		# 	# self.seed = random.randint(1, 10000)
		# 	self.seed = self.task_id*self.task_id
		# self.logger.info('Random seed: {}'.format(self.seed))
		# util.set_random_seed(self.seed)
		self.create_dataset()
		if self.is_train:

			self.corr_list = np.zeros(opt['ntasks'] - 1)
			self.temperature = 1

		# self.temperature = 1/np.log(self.num_tasks-1)
	def resume_training(self):
		if self.opt['resume'] and self.opt['path']['resume_state']:  # resuming training
			resume_state = torch.load(self.opt['path']['resume_state'])
		else:
			resume_state = None

		if resume_state:
			self.logger.info('Resuming training from epoch for task {}: {}, iter: {}.'.format(
				self.task_id, resume_state['epoch'], resume_state['iter']))
			options.check_resume(self.opt)  # check resume options
			start_epoch = resume_state['epoch']
			current_step = resume_state['iter']
			self._resume_training(resume_state)  # handle optimizers and schedulers
		else:
			start_epoch = 0
			current_step = 0
		# training
		self.logger.info('Task {} start training from epoch: {:d}, iter: {:d}'.format(self.task_id, start_epoch,
		                                                                         current_step))

	def create_dataset(self):
		# create train and val dataloader
		for phase, dataset_opt in self.opt['datasets'].items():
			if phase == 'val':
				val_set = create_dataset(dataset_opt, split=phase)
				self.val_loader = create_dataloader(val_set, dataset_opt)
				self.logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
			elif phase == 'train':
				train_set = create_dataset(dataset_opt, split=phase)

				# if 'no_split' in dataset_opt.keys() and dataset_opt['no_split'] == True:
				# 	train_set = full_set
				# else:
				# 	train_size = int(0.8 * len(full_set))
				# 	val_size = len(full_set) - train_size
				# 	train_set, val_set = torch.utils.data.random_split(full_set, [train_size, val_size])
				# 	# val_set = create_dataset(dataset_opt, split='val')
				# 	self.val_mto_loader = create_dataloader(val_set, dataset_opt)
				# 	self.logger.info('Number of val_mto images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))

				train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
				self.logger.info('Number of train images: {:,d}, iters per epoch: {:,d}'.format(
					len(train_set), train_size))
				total_iters = int(self.opt['niter'])
				total_epochs = int(math.ceil(total_iters / train_size))
				self.logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
					total_epochs, total_iters))
				self.train_loader = create_dataloader(train_set, dataset_opt)
				# self.train_iter = iter(BackgroundGenerator(self.cycle(self.train_loader)))
				self.train_iter = iter(self._cycle(self.train_loader))
			elif phase == 'test':
				test_set = create_dataset(dataset_opt, split=phase)
				self.test_loader = create_dataloader(test_set, dataset_opt)
				self.logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
			# else:
			# 	raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
		assert self.train_loader is not None
		# assert self.val_loader is not None
		self.total_epochs = total_epochs
		self.total_iters = total_iters

	def validation(self, **kwargs):
		raise NotImplementedError

	@staticmethod
	# @background(max_prefetch=8)
	def _cycle(iteration):
		while True:
			for batch in iteration:
				yield batch

	def get_current_visuals(self):
		pass

	def get_current_losses(self):
		pass

	def print_network(self):
		pass

	def save(self, label):
		pass

	def load(self):
		pass

	def update_learning_rate(self, step=None):
		for scheduler in self.schedulers:
			scheduler.step(step)

	def get_current_learning_rate(self):
		return self.schedulers[0].get_lr()[-1]

	def get_network_description(self, network):
		'''Get the string and total parameters of the network'''
		if isinstance(network, nn.DataParallel):
			network = network.module
		s = str(network)
		n = sum(map(lambda x: x.numel(), network.parameters()))
		return s, n

	def save_network(self, network, iter_step, prefix=None):
		if prefix is not None:
			save_filename = '{}_{}.pth'.format(prefix, iter_step)
		else:
			save_filename = '{}.pth'.format(iter_step)
		save_path = os.path.join(self.opt['path']['models'], save_filename)
		if isinstance(network, nn.DataParallel):
			network = network.module
		state_dict = network.state_dict()
		for key, param in state_dict.items():
			state_dict[key] = param.cpu()
		torch.save(state_dict, save_path)

	def load_network(self, load_path, network, strict=True):
		if isinstance(network, nn.DataParallel):
			network = network.module
		network.load_state_dict(torch.load(load_path), strict=strict)

	def save_training_state(self, epoch, iter_step):
		'''Saves training state during training, which will be used for resuming 'seed': self.seed,'''
		state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
		for s in self.schedulers:
			state['schedulers'].append(s.state_dict())
		for o in self.optimizers:
			state['optimizers'].append(o.state_dict())
		save_filename = '{}.state'.format(iter_step)
		save_path = os.path.join(self.opt['path']['training_state'], save_filename)
		torch.save(state, save_path)

	def _resume_training(self, resume_state):
		'''Resume the optimizers and schedulers for training'''
		resume_optimizers = resume_state['optimizers']
		resume_schedulers = resume_state['schedulers']
		assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
		assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
		for i, o in enumerate(resume_optimizers):
			self.optimizers[i].load_state_dict(o)
		for i, s in enumerate(resume_schedulers):
			self.schedulers[i].load_state_dict(s)
		self.training_step = resume_state['iter']
		self.start_epoch = resume_state['epoch']
		# self.seed = resume_state['seed']
		# self.logger.info('Resume from seed {}'.format(self.seed))
		# util.set_random_seed(self.seed)
