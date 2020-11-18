import os
import torch
import torch.nn as nn
import numpy as np
from dataloader import create_dataloader, create_dataset
import math
from mpi4py import MPI
import logging
# import random
# from utils import util


class BaseTask:
	def __init__(self, opt, comm, device):
		self.opt = opt
		self.comm = comm
		self.device = device
		self.is_train = opt['is_train']
		self.schedulers = []
		self.optimizers = []
		self.training_step = 0
		self.start_epoch = 0
		self.task_id = opt['task_id']
		# self.logger = logger
		self.logger = logging.getLogger(str(self.task_id))

		self.seed = self.opt['manual_seed']
		# if self.seed is None or self.task_id != 0:
		# 	# self.seed = random.randint(1, 10000)
		# 	self.seed = self.task_id*self.task_id
		# self.logger.info('Random seed: {}'.format(self.seed))
		# util.set_random_seed(self.seed)
		self.init_comm()
		self.create_dataset()

	def init_comm(self):
		self.world_size = self.comm.Get_size()
		device_per_node = torch.cuda.device_count()
		self.all_explore_id = [i for i in list(range(self.world_size)) if i % device_per_node != 0]
		self.all_exploit_id = [i for i in list(range(self.world_size)) if i % device_per_node == 0]

		self.node_id = self.task_id // device_per_node
		self.node_exploit_rank = self.task_id - self.task_id % device_per_node
		self.node_explorer_ranks = [self.node_exploit_rank + i for i in range(1, device_per_node)]
		self.num_nodes = self.world_size // device_per_node

		group = self.comm.Get_group()
		exploit_group = MPI.Group.Incl(group, self.all_exploit_id)
		self.exploit_comm = self.comm.Create(exploit_group)

		self.corr_list = np.ones(self.num_nodes)

	def create_dataset(self):
		# create train and val dataloader
		for phase, dataset_opt in self.opt['datasets'].items():
			if phase == 'train':
				train_set = create_dataset(dataset_opt, split=phase)
				train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
				self.logger.info('Number of train images: {:,d}, iters per epoch: {:,d}'.format(
					len(train_set), train_size))
				total_iters = int(self.opt['niter'])
				total_epochs = int(math.ceil(total_iters / train_size))
				self.logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
					total_epochs, total_iters))
				self.train_loader = create_dataloader(train_set, dataset_opt)
			elif phase == 'val':
				val_set = create_dataset(dataset_opt, split=phase)
				self.val_loader = create_dataloader(val_set, dataset_opt)
				self.logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
			elif phase == 'mix':
				mix_set = create_dataset(dataset_opt, split=phase)
				self.mix_loader = create_dataloader(mix_set, dataset_opt)
				self.logger.info('Number of mix images in [{:s}]: {:d}'.format(dataset_opt['name'], len(mix_set)))
			else:
				raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
		assert self.train_loader is not None
		# assert self.val_loader is not None
		self.total_epochs = total_epochs
		self.total_iters = total_iters

	def validation(self):
		raise NotImplementedError

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

	def update_learning_rate(self):
		for scheduler in self.schedulers:
			scheduler.step()

	def get_current_learning_rate(self):
		return self.schedulers[0].get_lr()[0]

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

	def resume_training(self, resume_state):
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
