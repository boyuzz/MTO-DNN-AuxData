import os
import torch
import torch.nn as nn
import numpy as np
from dataloader import create_dataloader, create_dataset
from dataloader.util import cv_split, train_val_split
import math
from mpi4py import MPI
import logging
from options import options
import copy
# import random
# from utils import util


class ResampleDataset(torch.utils.data.Dataset):
	def __init__(self, dataset):
		self.dataset = dataset
		length = len(dataset)
		self.index = np.random.randint(0, length, length)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, item):
		img, target = self.dataset[self.index[item]]
		return img, target

class BaseTask:
	def __init__(self, opt, comm, device, create=True):
		self.opt = opt
		self.comm = comm
		self.device = device
		self.is_train = opt['is_train']
		self.schedulers = []
		self.optimizers = []
		self.training_step = 0
		self.start_epoch = 0
		self.task_id = opt['task_id']
		self.device_per_task = opt['device_per_task']
		self.logger = logging.getLogger(str(self.task_id))

		self.seed = self.opt['manual_seed']
		# if self.seed is None or self.task_id != 0:
		# 	# self.seed = random.randint(1, 10000)
		# 	self.seed = self.task_id*self.task_id
		# self.logger.info('Random seed: {}'.format(self.seed))
		# util.set_random_seed(self.seed)
		self.init_comm()
		if create:
			self.create_dataset()

		# if opt['resume'] and opt['path']['resume_state']:  # resuming training
		# 	resume_state = torch.load(opt['path']['resume_state'])
		# else:
		# 	resume_state = None
		#
		# if resume_state:
		# 	self.logger.info('Resuming training from epoch for task {}: {}, iter: {}.'.format(
		# 		self.task_id, resume_state['epoch'], resume_state['iter']))
		# 	option.check_resume(opt)  # check resume options
		# 	start_epoch = resume_state['epoch']
		# 	current_step = resume_state['iter']
		# 	self.resume_training(resume_state)  # handle optimizers and schedulers
		# else:
		# 	start_epoch = 0
		# 	current_step = 0
		# # training
		# self.logger.info('Task {} start training from epoch: {:d}, iter: {:d}'.format(self.task_id, start_epoch,
		#                                                                          current_step))

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

	def init_comm(self):
		self.world_size = self.comm.Get_size()
		self.rank = self.comm.Get_rank()

		self.all_explore_id = [i for i in list(range(self.world_size)) if i % self.device_per_task != 0]
		self.all_exploit_id = [i for i in list(range(self.world_size)) if i % self.device_per_task == 0]

		self.task_solver_rank = self.rank - self.rank % self.device_per_task
		self.task_explorer_ranks = [self.task_solver_rank + i for i in range(1, self.device_per_task)]
		self.num_tasks = self.world_size // self.device_per_task
		if self.num_tasks == 1:
			self.logger.warning("Only 1 task is running, all the transferring methods will be switched off!")

		group = self.comm.Get_group()
		exploit_group = MPI.Group.Incl(group, self.all_exploit_id)
		self.exploit_comm = self.comm.Create(exploit_group)

		task_ranks = copy.deepcopy(self.task_explorer_ranks)
		task_ranks.append(self.task_solver_rank)
		task_group = MPI.Group.Incl(group, task_ranks)
		self.task_comm = self.comm.Create(task_group)

		self.corr_list = np.zeros(self.num_tasks - 1)
		self.temperature = 1
		# self.temperature = 1/np.log(self.num_tasks-1)

	def create_dataset(self):
		# create train and val dataloader
		for phase, dataset_opt in self.opt['datasets'].items():
			if phase == 'train':
				if self.opt['varyOnCV']:
					train_set = create_dataset(dataset_opt, split=phase)
					nfold = self.num_tasks
					# nfold = 5
					folds = cv_split(train_set, nfold, self.comm)
					self.folds_loaders = [create_dataloader(f, dataset_opt) for f in folds]
					self.train_set = folds.pop(dataset_opt['fold']-1)
					self.logger.info("split into {} folds, currently in fold {}".format(nfold, dataset_opt['fold']))
					# self.val_set = val_fold
					if self.opt['varyOnSample']:
						self.train_set = ResampleDataset(self.train_set)
				else:
					self.train_set = create_dataset(dataset_opt, split=phase)
					# self.opt['varyOnSample'] = True
					if self.opt['varyOnSample']:
						self.train_set = ResampleDataset(self.train_set)

					# self.opt['create_val'] = True
					if self.opt['create_val']:
						# task0 for 0.1, else for random in [0, 0.3]
						ratio = 0.1
						# if self.task_id == 0:
						# 	ratio = 0.1
						# else:
						# 	ratio = np.random.choice([0.1,0.2,0.3])
						self.train_set, val_set = train_val_split(self.train_set, ratio, comm=None)	#self.comm
						# val_folds = self.comm.allgather(val_set)
						# self.logger.info([vf[0] for vf in val_folds])	# test if val_folds in all ranks are the same
						# self.folds_loaders = [create_dataloader(f, dataset_opt) for f in val_folds]
						self.val_loader = create_dataloader(val_set, dataset_opt)
						self.logger.info(
							'rank {}, Number of val images in [{:s}]: {:d}'.format(self.rank, dataset_opt['name'],
							                                                       len(val_set)))

				# self.opt['varyOnSample'] = True
				self.train_loader = create_dataloader(self.train_set, dataset_opt)
				self.train_iter = iter(self._cycle(self.train_loader))
				train_size = int(math.ceil(len(self.train_set) / dataset_opt['batch_size']))
				self.logger.info('rank {}, Number of train images: {:,d}, iters per epoch: {:,d}'.format(self.rank,
					len(self.train_set), train_size))
				total_iters = int(self.opt['niter'])
				total_epochs = int(math.ceil(total_iters / train_size))
				self.logger.info('rank {}, Total epochs needed: {:d} for iters {:,d}'.format(self.rank,
					total_epochs, total_iters))
				self.total_epochs = total_epochs
				self.total_iters = total_iters

			elif phase == 'val' and not self.opt['varyOnCV'] and not self.opt['create_val']:
				val_set = create_dataset(dataset_opt, split=phase)
				self.val_loader = create_dataloader(val_set, dataset_opt)
				self.logger.info('rank {}, Number of val images in [{:s}]: {:d}'.format(self.rank, dataset_opt['name'], len(val_set)))
			elif phase == 'test':
				test_set = create_dataset(dataset_opt, split=phase)
				self.test_loader = create_dataloader(test_set, dataset_opt)
				self.logger.info('rank {}, Number of test images in [{:s}]: {:d}'.format(self.rank, dataset_opt['name'], len(test_set)))
			else:
				raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
		assert self.train_loader is not None
		# assert self.val_loader is not None

	@staticmethod
	# @background(max_prefetch=8)
	def _cycle(iteration):
		while True:
			for batch in iteration:
				yield batch

	def _validation(self):
		raise NotImplementedError

	def get_current_visuals(self):
		pass

	def _is_solver(self):
		return True if self.rank == self.task_solver_rank else False

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
			if scheduler is not None:
				scheduler.step()

	def get_current_learning_rate(self):
		# return self.schedulers[0].get_lr()[-1]
		return self.optimizers[0].param_groups[0]['lr']

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
