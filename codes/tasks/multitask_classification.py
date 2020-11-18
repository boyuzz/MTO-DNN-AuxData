from collections import OrderedDict
from sklearn import metrics
import numpy as np
from scipy import special
from mpi4py import MPI
import copy

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import warnings
from prefetch_generator import BackgroundGenerator

import model.cifar as cifar
import model.mnist as mnist
import model.imagenet as imagenet
from model.networks import init_weights

from .scalable_base_task import BaseTask


class ClassificationTask(BaseTask):
	def __init__(self, opt, **kwargs):
		super(ClassificationTask, self).__init__(opt, kwargs['logger'])
		train_opt = opt['train']
		self.ran_cl = opt['rancl']
		self.num_classes = opt['datasets']['train']['num_classes']
		self.kd_transfer = opt['kd_transfer']
		self.att_transfer = opt['att_transfer']
		self.fsp_transfer = opt['fsp_transfer']
		self.w_transfer = opt['w_transfer']
		self.ws_transfer = opt['ws_transfer']
		self.tb_logger = kwargs['tb_logger']
		self.device_id = kwargs['device_id']
		self.replace_classifier = opt['varyOnData']

		# -----prepare for transfer-------------
		if self.kd_transfer:
			# set seeds of all tasks the same to ensure the dataloader is in the same order
			torch.manual_seed(0)

		if self.fsp_transfer or self.att_transfer:
			self.activation = OrderedDict()

		# -----define network and load pretrained tasks-----
		data_name, model_name = opt['network'].split('_')
		self.model_name = model_name
		if data_name.lower() == 'mnist':
			self.network = getattr(mnist, model_name)(num_classes=self.num_classes).cuda(self.device_id)
		elif data_name.lower() == 'cifar':
			self.network = getattr(cifar, model_name)(num_classes=self.num_classes).cuda(self.device_id)
			# if self.fsp_transfer and 'resnet' in model_name.lower():
			# 	# self.network.layer1[0].conv1.register_forward_hook(self.get_activation('b1_in'))
			# 	# self.network.layer1[-1].register_forward_hook(self.get_activation('b1_out'))
			# 	# self.network.layer2[0].conv1.register_forward_hook(self.get_activation('b2_in'))
			# 	# self.network.layer2[-1].register_forward_hook(self.get_activation('b2_out'))
			# 	# self.network.layer3[0].conv1.register_forward_hook(self.get_activation('b3_in'))
			# 	# self.network.layer3[-1].register_forward_hook(self.get_activation('b3_out'))
			# 	self.network.layer4[0].conv1.register_forward_hook(self.get_activation('b4_in'))
			# 	self.network.layer4[-1].register_forward_hook(self.get_activation('b4_out'))

			if self.att_transfer and 'resnet' in model_name.lower():
				self.network.layer1[-1].register_forward_hook(self.get_activation('b1_out'))
				self.network.layer2[-1].register_forward_hook(self.get_activation('b2_out'))
				self.network.layer3[-1].register_forward_hook(self.get_activation('b3_out'))
				self.network.layer4[-1].register_forward_hook(self.get_activation('b4_out'))
		elif data_name.lower() == 'imagenet':
			if opt['imagenet_pretrained']:
				self.network = getattr(imagenet, model_name)(pretrained=True)
				if opt['train_lastlayer']:
					for param in self.network.parameters():
						param.requires_grad = False

				if 'resnet' in self.model_name:
					self.network.fc = nn.Linear(self.network.fc.in_features, self.num_classes)
				elif 'vgg' in self.model_name:
					self.network.classifier[-1] = nn.Linear(4096, self.num_classes)

				self.network = self.network.cuda(self.device_id)
			else:
				self.network = getattr(imagenet, model_name)(num_classes=self.num_classes).cuda(self.device_id)

			if self.att_transfer and 'resnet' in model_name.lower():
				self.network.layer1[-1].register_forward_hook(self.get_activation('b1_out'))
				self.network.layer2[-1].register_forward_hook(self.get_activation('b2_out'))
				self.network.layer3[-1].register_forward_hook(self.get_activation('b3_out'))
				self.network.layer4[-1].register_forward_hook(self.get_activation('b4_out'))
		else:
			raise NotImplementedError('Network [{:s}, {:s}] is not defined.'.format(data_name, model_name))

		# make starts the same
		# if USE_HVD:
		# 	hvd.broadcast_parameters(self.network.state_dict(), root_rank=0)
		# test if different task has the same initialization under same seed
		# for name, param in self.network.named_parameters():
		# 	print(param[0])

		# load pretrained model if exists
		self.load()
			# print network
			# self.print_network()

		# for name, param in self.network.named_parameters():
		# 	a = param.clone().cpu().data.numpy()
		# 	print(self.task_id, 'scale', name, a.max(), a.min())

		# -----define loss function------
		self.one_hot = False
		self.prob_est = False
		loss_type = train_opt['loss']
		if loss_type == 'l1':
			self.loss_func = nn.L1Loss().cuda(self.device_id)
			self.one_hot = True
		elif loss_type == 'l2':
			self.loss_func = nn.MSELoss().cuda(self.device_id)
			self.one_hot = True
		elif loss_type == 'l1_pro':
			self.loss_func = nn.L1Loss().cuda(self.device_id)
			self.prob_est = True
			self.one_hot = True
		elif loss_type == 'l2_pro':
			self.loss_func = nn.MSELoss().cuda(self.device_id)
			self.prob_est = True
			self.one_hot = True
		elif loss_type == 'cross_entropy':
			self.loss_func = nn.CrossEntropyLoss().cuda(self.device_id)
			# self.loss_func = nn.NLLLoss().cuda(self.device_id)
			# self.prob_est = True
		elif loss_type == 'marginloss':
			self.loss_func = nn.MultiMarginLoss().cuda(self.device_id)
		else:
			raise NotImplementedError(
				'Loss type [{:s}] is not recognized. Please specifiy it from following options:'.format(loss_type))

		if self.is_train:
			self.network.train()

			# -----define optimizers-----
			optim_type = train_opt['optim']

			# if opt['imagenet_pretrained']:
			# 	import warnings
			# 	warnings.warn("Since using pretrained model, the lr will be reset as 1e-6 for 1st layer, 1e-4 for middle layer,"
			# 	              "and lr for last layer.")
			param_groups = []
			# for name, param in self.network.named_parameters():
			if 'resnet' in model_name:
				param_groups.append(
					{'params': [param for name, param in self.network.named_parameters() if 'fc' not in name]})
				param_groups.append({'params': self.network.fc.parameters()})
			elif 'vgg' in model_name:
				# TODO: classifier lr different group
				param_groups.append(
					{'params': [param for name, param in self.network.named_parameters() if 'fc' not in name]})
				param_groups.append({'params': self.network.fc.parameters()})
				# if 'classifier.6' not in name:
				# 	param_groups += [{'params': param}]
				# else:
				# 	param_groups += [{'params': param}]

			# 	self.optimizer = getattr(optim, optim_type)(param_groups, **opt['train']['optimizer_param'])
			# else:

			self.optimizer = getattr(optim, optim_type)(param_groups, **opt['train']['optimizer_param'])
			self.optimizers.append(self.optimizer)
			self.init_lr = opt['train']['optimizer_param']['lr']

			# -----define schedulers-----
			if train_opt['lr_scheme'] == 'MultiStepLR':
				for optimizer in self.optimizers:
					# self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
					# 	train_opt['lr_steps'], train_opt['lr_gamma']))
					lambda1 = lambda step: train_opt['lr_gamma'] ** sum([step > mst for mst in train_opt['lr_steps']])
					self.schedulers.append(lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda1]))
			elif train_opt['lr_scheme'] == 'CycleLR':
				from utils.cycleLR import CyclicLR
				for optimizer in self.optimizers:
					self.schedulers.append(CyclicLR(optimizer, base_lr=train_opt['lr_param'][0],
					                                max_lr=train_opt['lr_param'][1]))
			else:
				raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

			# # -----register gradient clipping-----
			# for param in self.network.parameters():
			# 	param.register_hook(lambda grad: torch.clamp(grad, -0.2, 0.2))
			# -----define log_dict-----
			self.log_dict = OrderedDict()
			self.received_count = 0

	def get_activation(self, name):
		def hook(model, input, output):
			self.activation[name] = output
		return hook

	def _convert_int_onehot(self, labels):
		# One hot encoding buffer that you create out of the loop and just keep reusing
		labels = labels.view(-1, 1)
		y_onehot = torch.zeros(labels.size(0), self.num_classes).cuda(self.device_id)

		# In your for loop
		y_onehot.scatter_(1, labels, 1)
		return y_onehot

	def step(self, rank, log_freq, save_freq, is_solver=False):
		(data, target) = next(self.train_iter)

		self.training_step += 1
		# update learning rate
		self.update_learning_rate()
		# t = time.time()
		data, target = data.cuda(self.device_id), target.cuda(self.device_id)
		# print(target)
		if self.one_hot:
			target = self._convert_int_onehot(target)

		self.optimizer.zero_grad()
		logits = self.network(data)

		loss = 0
		if self.prob_est:
			logits = F.softmax(logits, dim=1)
		# self.logger.info(logits[0])
		loss += self.loss_func(logits, target)

		loss.backward()
		self.optimizer.step()
		if self.loss_func:
			self.log_dict['Training_loss'] = loss.item()

		# log
		if self.training_step % log_freq == 0:  # and self.rank == self.task_exploit_rank
			logs = self.get_current_log()

			message = '<Task {}, rank{} iter:{:3d} lr:{}, task:{}> '.format(
				self.task_id, rank, self.training_step, self.get_current_learning_rate(), self.task_id)

			for k, v in logs.items():
				message += '{:s}: {:.4e} '.format(k, v)
				# tensorboard self.logger
				if is_solver and self.opt['use_tb_logger']:
					self.tb_logger.add_scalar(k, v, self.training_step)

			if is_solver and self.opt['use_tb_logger']:
				for i, lr in enumerate(self.get_current_learning_rate()):
					self.tb_logger.add_scalar('lr_{}'.format(i), lr, self.training_step)

			# if tb_logger is not None:
			# 	for name, param in self.network.named_parameters():
			# 		if param.grad is not None:
			# 			tb_logger.add_histogram(name, param.grad.clone().cpu().data.numpy(), self.training_step)
			self.logger.info(message)

			# save tasks and training states
			if self.training_step % save_freq == 0 and is_solver:
				self.logger.info('Saving tasks and training states for task {}.'.format(self.task_id))
				self.save(self.training_step)
				self.save_training_state(0, self.training_step)

	def validation(self, verbose=False):
		self.network.eval()
		test_loss = 0.
		# test_accuracy = 0.
		full_preds = []
		full_labels = []

		with torch.no_grad():
			for data, target in self.val_loader:
				data, target = data.cuda(self.device_id), target.cuda(self.device_id)
				labels = target.data
				if self.one_hot:
					target = self._convert_int_onehot(target)

				logits = self.network(data)
				if self.prob_est:
					logits = F.softmax(logits, dim=1)
				# sum up batch loss
				test_loss += self.loss_func(logits, target).item()
				# get the index of the max log-probability
				pred = logits.data.max(1, keepdim=True)[1]
				full_preds.extend(pred.view(-1).cpu())
				full_labels.extend(labels.cpu())
				# test_accuracy += pred.eq(labels.view_as(pred)).cpu().float().sum()

		test_loss /= len(self.val_loader)
		# test_accuracy /= len(self.testing_set)
		test_accuracy = 100*metrics.accuracy_score(np.array(full_labels), np.array(full_preds))
		test_f1 = metrics.f1_score(np.array(full_labels), np.array(full_preds), average='macro')
		# test_roc_auc = metrics.roc_auc_score(np.array(full_labels), np.array(full_preds))
		# TODO: add mAP metrics
		if verbose:
			self.logger.info('# Task {}  # Validation # Loss: {:.4e}, Accuracy: {:.2f}%, F1 score: {:.4f}'.format(self.task_id, test_loss,
																						 test_accuracy, test_f1))
		self.network.train()
		return test_loss, test_accuracy, test_f1 #, test_roc_auc

	def get_current_learning_rate(self):
		return [self.optimizer.param_groups[i]['lr'] for i in range(len(self.optimizer.param_groups))]

	def get_current_log(self):
		return self.log_dict

	def print_network(self):
		s, n = self.get_network_description(self.network)
		if isinstance(self.network, nn.DataParallel):
			net_struc_str = '{} - {}'.format(self.network.__class__.__name__,
											 self.network.module.__class__.__name__)
		else:
			net_struc_str = '{}'.format(self.network.__class__.__name__)

		self.logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
		self.logger.info(s)

	def load(self):
		load_path_network = self.opt['path']['pretrain_model']
		if load_path_network is not None:
			self.logger.info('Loading pretrained model [{:s}] ...'.format(load_path_network))
			self.load_network(load_path_network, self.network)

	def save(self, iter_step):
		self.save_network(self.network, iter_step)
