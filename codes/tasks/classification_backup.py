import logging
from collections import OrderedDict
from sklearn import metrics
import numpy as np
import scipy.stats as stats
import copy

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

import model.cifar as cifar
import model.mnist as mnist
import model.imagenet as imagenet
from model.networks import init_weights

try:
	import horovod.torch as hvd
	USE_HVD = True
except ImportError:
	USE_HVD = False

# import threading
# import mpi4py
# mpi4py.rc.initialize = False

from .base_task import BaseTask


class ClassificationTask(BaseTask):
	def __init__(self, opt, logger, comm, device):
		super(ClassificationTask, self).__init__(opt, logger, comm, device)
		train_opt = opt['train']
		self.num_classes = opt['num_classes']
		self.kd_transfer = opt['kd_transfer']
		self.att_transfer = opt['att_transfer']
		self.fsp_transfer = opt['fsp_transfer']
		self.w_transfer = opt['w_transfer']
		self.ws_transfer = opt['ws_transfer']

		# -----prepare for transfer-------------
		if self.kd_transfer:
			# set seeds of all tasks the same to ensure the dataloader is in the same order
			torch.manual_seed(0)

		if self.fsp_transfer or self.att_transfer:
			self.activation = OrderedDict()

		# -----define network and load pretrained tasks-----
		data_name, model_name = opt['network'].split('_')
		if data_name.lower() == 'mnist':
			self.network = getattr(mnist, model_name)(num_classes=self.num_classes).cuda(self.device)
		elif data_name.lower() == 'cifar':
			self.network = getattr(cifar, model_name)(num_classes=self.num_classes).cuda(self.device)
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

				if 'resnet' in model_name:
					self.network.fc = nn.Linear(self.network.fc.in_features, self.num_classes)
				elif 'vgg' in model_name:
					self.network.classifier[6] = nn.Linear(4096, self.num_classes)
				self.network = self.network.cuda(self.device)
			else:
				self.network = getattr(imagenet, model_name)(num_classes=self.num_classes).cuda(self.device)
			if self.att_transfer and 'resnet' in model_name.lower():
				self.network.layer1[-1].register_forward_hook(self.get_activation('b1_out'))
				self.network.layer2[-1].register_forward_hook(self.get_activation('b2_out'))
				self.network.layer3[-1].register_forward_hook(self.get_activation('b3_out'))
				self.network.layer4[-1].register_forward_hook(self.get_activation('b4_out'))
		else:
			raise NotImplementedError('Network [{:s}, {:s}] is not defined.'.format(data_name, model_name))

		init_weights(self.network)

		# make starts the same
		# if USE_HVD:
		# 	hvd.broadcast_parameters(self.network.state_dict(), root_rank=0)

		# test if different task has the same initialization under same seed
		# for name, param in self.network.named_parameters():
		# 	print(param[0])

		# load pretrained model if exists
		self.load()

		# -----define loss function------
		self.one_hot = False
		self.prob_est = False
		loss_type = train_opt['loss']
		if loss_type == 'l1':
			self.loss_func = nn.L1Loss().cuda(self.device)
			self.one_hot = True
		elif loss_type == 'l2':
			self.loss_func = nn.MSELoss().cuda(self.device)
			self.one_hot = True
		elif loss_type == 'l1_pro':
			self.loss_func = nn.L1Loss().cuda(self.device)
			self.prob_est = True
			self.one_hot = True
		elif loss_type == 'l2_pro':
			self.loss_func = nn.MSELoss().cuda(self.device)
			self.prob_est = True
			self.one_hot = True
		elif loss_type == 'cross_entropy':
			self.loss_func = nn.CrossEntropyLoss().cuda(self.device)
			# self.loss_func = nn.NLLLoss().cuda(self.device)
			# self.prob_est = True
		elif loss_type == 'marginloss':
			self.loss_func = nn.MultiMarginLoss().cuda(self.device)
		else:
			raise NotImplementedError(
				'Loss type [{:s}] is not recognized. Please specifiy it from following options:'.format(loss_type))

		if self.is_train:
			self.network.train()

			self.logits_loss = nn.KLDivLoss(reduction='batchmean').cuda(self.device)
			self.norm_loss = nn.MSELoss().cuda(self.device)
			self.at_weight = train_opt['at_weight']
			self.kd_weight = train_opt['kd_weight']
			self.ws_weight = train_opt['ws_weight']

			# -----define optimizers-----
			optim_params = []
			for k, v in self.network.named_parameters():  # can optimize for a part of the model
				if v.requires_grad:
					optim_params.append(v)
				else:
					self.logger.warning('Params [{:s}] will not optimize.'.format(k))
			optim_type = train_opt['optim']
			self.optimizer = getattr(optim, optim_type)(self.network.parameters(), **opt['train']['optimizer_param'])
			self.optimizers.append(self.optimizer)

			# -----define schedulers-----
			if train_opt['lr_scheme'] == 'MultiStepLR':
				for optimizer in self.optimizers:
					self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
						train_opt['lr_steps'], train_opt['lr_gamma']))
			else:
				raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

			# # -----register gradient clipping-----
			# for param in self.network.parameters():
			# 	param.register_hook(lambda grad: torch.clamp(grad, -0.2, 0.2))

			# -----define log_dict-----
			self.log_dict = OrderedDict()
			self.log_dict['transfer_count'] = 0

		# print network
		# self.print_network()

	def get_activation(self, name):
		def hook(model, input, output):
			self.activation[name] = output
		return hook

	def _convert_int_onehot(self, labels):
		# One hot encoding buffer that you create out of the loop and just keep reusing
		labels = labels.view(-1, 1)
		y_onehot = torch.zeros(labels.size(0), self.num_classes).cuda(self.device)

		# In your for loop
		y_onehot.scatter_(1, labels, 1)
		return y_onehot

	def full_training(self, log_freq, val_freq, save_freq, tb_logger=None):
		for epoch in range(self.start_epoch, self.total_epochs):
			for batch_idx, (data, target) in enumerate(self.train_loader):
				self.training_step += 1
				if self.training_step > self.total_iters:
					break
				# update learning rate
				self.update_learning_rate()

				data, target = data.cuda(self.device), target.cuda(self.device)
				# print(target)
				if self.one_hot:
					target = self._convert_int_onehot(target)

				# weights = copy.deepcopy(self.network.state_dict())
				# for name in weights:
				# 	for i in range(weights[name].shape[0]):
				# 		weights[name][i] = weights[name][0].clone()
				# self.network.load_state_dict(weights)

				# apply weight transfer
				if self.w_transfer and USE_HVD and np.random.rand() < self.training_step/self.total_iters:
					self.weights_transfer()

				self.optimizer.zero_grad()

				logits = self.network(data)

				loss = 0

				# apply fsp_transfer
				if self.fsp_transfer and USE_HVD:
					fsp_loss = self.fsp_matrix_transfer()
					self.log_dict['fsp_loss'] = fsp_loss.item()
					loss += self.at_weight * fsp_loss

				# apply attention_transfer
				if self.att_transfer and USE_HVD:
					att_loss = self.attention_transfer()
					self.log_dict['att_loss'] = att_loss.item()
					loss += self.at_weight * att_loss

				# apply knowledge distillation
				if self.kd_transfer:
					kd_loss = self.knowledge_distillation_transfer(logits)
					self.log_dict['kd_loss'] = kd_loss.item()
					loss += self.kd_weight * kd_loss

				# apply weight statistical transfer
				if self.ws_transfer and USE_HVD:
					ws_loss = self.weights_statistic_transfer()
					self.log_dict['ws_loss'] = ws_loss.item()
					loss += self.ws_weight * ws_loss

				if self.prob_est:
					logits = F.softmax(logits, dim=1)
				loss += self.loss_func(logits, target)

				# TODO: trial for asynchronous communication
				# if hvd.poll(rec_knowledge_handle):
				# 	rec_knowledge = hvd.synchronize(rec_knowledge_handle)
				# 	kd_loss = 0
				# 	for i, other_logits in rec_knowledge:
				# 		if i != self.task_id:
				# 			kd_loss += self.logits_loss(other_logits, logits)
				# 	loss = self.loss_func(logits, target) + kd_loss/(rec_knowledge.size(0)-1)
				# 	transfer_count += 1
				# else:
				# 	loss = self.loss_func(logits, target)
				loss.backward()
				# clip gradient by L2 norm
				for param in self.network.parameters():
					torch.nn.utils.clip_grad_norm_(param, 1.0, 2)
				# for name, param in self.network.named_parameters():
				# 	if param.grad is not None:
				# 		a = param.grad.clone().cpu().data.numpy()
				# 		print(self.task_id, 'grad', name, a.max(), a.min())
				# print('\n')
				self.optimizer.step()

				# set log
				if self.loss_func:
					self.log_dict['Training_loss'] = loss.item()

				# log
				if self.training_step % log_freq == 0:
					logs = self.get_current_log()

					message = '<Task {}, epoch:{:3d}, iter:{:3d} lr:{:.3e}, task:{}> '.format(
						self.task_id, epoch, self.training_step, self.get_current_learning_rate(), self.task_id)

					for k, v in logs.items():
						message += '{:s}: {:.4e} '.format(k, v)
						# tensorboard self.logger
						if tb_logger is not None:
							tb_logger.add_scalar(k, v, self.training_step)

					if tb_logger is not None:
						tb_logger.add_scalar('lr', self.get_current_learning_rate(), self.training_step)

					# if tb_logger is not None:
					# 	for name, param in self.network.named_parameters():
					# 		tb_logger.add_histogram(name, param.clone().cpu().data.numpy(), self.training_step)
					self.logger.info(message)

				# validation
				if self.training_step % val_freq == 0 and self.val_loader is not None:
					# log
					logger_val = logging.getLogger('val')  # validation self.logger
					test_loss, test_accuracy, test_f1 = self.validation()
					step = 'last' if self.training_step == self.total_iters else self.training_step
					logger_val.info(
						'<Task {}, epoch:{:3d}, # Iter: {} Loss: {:.4e}, Accuracy: {:.2f}%, F1 score: {:.4f}>'.format(
							self.task_id, epoch, step, test_loss, test_accuracy*100, test_f1))
					self.network.train()
					# tensorboard self.logger
					if tb_logger is not None:
						tb_logger.add_scalar('Test_loss', test_loss, self.training_step)
						tb_logger.add_scalar('Accuracy', test_accuracy, self.training_step)
						tb_logger.add_scalar('F1', test_f1, self.training_step)

				# save tasks and training states
				if self.training_step % save_freq == 0:
					self.logger.info('Saving tasks and training states for task {}.'.format(self.task_id))
					self.save(self.training_step)
					self.save_training_state(epoch, self.training_step)

		self.logger.info("knowledge transferred {:d} times".format(self.log_dict['transfer_count']))

	def validation(self, verbose=False):
		self.network.eval()
		test_loss = 0.
		# test_accuracy = 0.
		full_preds = []
		full_labels = []

		with torch.no_grad():
			for data, target in self.val_loader:
				data, target = data.cuda(self.device), target.cuda(self.device)
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
		test_accuracy = metrics.accuracy_score(np.array(full_labels), np.array(full_preds))
		test_f1 = metrics.f1_score(np.array(full_labels), np.array(full_preds), average='macro')
		# test_roc_auc = metrics.roc_auc_score(np.array(full_labels), np.array(full_preds))
		# TODO: add mAP metrics
		if verbose:
			self.logger.info('# Task {}  # Validation # Loss: {:.4e}, Accuracy: {:.2f}%, F1 score: {:.4f}'.format(self.task_id, test_loss,
																						 test_accuracy*100, test_f1))

		return test_loss, test_accuracy, test_f1 #, test_roc_auc

	def knowledge_distillation_transfer(self, logits):
		'''
		To use this function, make sure the input dataloader to each task is exactly the same,
		otherwise this function is meaningless.
		:return:
		'''

		logits_data = logits.data.unsqueeze(0)
		rec_knowledge = self.comm.allgather(logits_data)

		kd_loss = 0
		# batchsize = int(rec_knowledge.size(0) // hvd.size())
		# self.logger.info("knowledge size {}, batchsize {}".format(rec_knowledge.size(), batchsize))
		# Note, the KDDivloss in pytorch says:
		# As with NLLLoss, the input given is expected to contain log-probabilities. However, unlike NLLLoss,
		# input is not restricted to a 2D Tensor. The targets are given as probabilities
		# (i.e. without taking the logarithm).
		# KLDivLoss: l_n = y_n*(log(y_n)-x_n)
		for i in range(0, self.comm.Get_size()):
			if i != self.task_id:
				# logits_from_other = rec_knowledge[i * batchsize:(i + 1) * batchsize, :].cuda(self.device)
				kd_loss += self.logits_loss(F.log_softmax(logits, dim=1), F.softmax(rec_knowledge[i].cuda(self.device), dim=1))
		kd_loss /= (self.comm.Get_size()-1)

		self.log_dict['transfer_count'] += 1
		return kd_loss

	def attention_transfer(self):
		def at(x):
			return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

		handles = []
		att_group = []
		for key in self.activation:
			at_out = at(self.activation[key])
			att_group.append(at_out)
			at_numpy = at_out.data.unsqueeze(0)
			handle = hvd.allgather_async(at_numpy, key)
			handles.append(handle)
			# self.norm_loss

		att_loss = 0
		for idx, handle in enumerate(handles):
			rec_att = hvd.synchronize(handle)
			# att_loss += self.norm_loss(att_group[idx], rec_att.mean(0).cuda(self.device))
			for i in range(0, hvd.size()):
				if i != self.task_id:
					att_loss += self.norm_loss(att_group[idx], rec_att[i].cuda(self.device))
		att_loss /= (hvd.size() - 1)
		self.log_dict['transfer_count'] += 1
		return att_loss

	def fsp_matrix_transfer(self):
		'''
		obtain the feature maps of bottlenecks (h*w*m), reshape it to (hw*m), then do matrix multiplication (m*n)
		allgather the mm, use L2 loss on it
		:return:
		'''
		handles = []
		matrix_group = []
		for key in self.activation:
			if 'in' in key:
				fm_in = self.activation[key]
			if 'out' in key:
				fm_out = self.activation[key]

				fm_in = fm_in.view(fm_in.shape[0], fm_in.shape[1], -1)
				fm_out = fm_out.view(fm_out.shape[0], fm_out.shape[1], -1)
				fm_out = torch.transpose(fm_out, 1, 2)
				fsp_matrix = torch.bmm(fm_in, fm_out)/fm_in.shape[-1]
				matrix_group.append(fsp_matrix)
				fsp_matrix = fsp_matrix.unsqueeze(0)
				handle = hvd.allgather_async(fsp_matrix, key)
				handles.append(handle)

		fsp_loss = 0
		for idx, handle in enumerate(handles):
			rec_fsp = hvd.synchronize(handle)
			for i in range(0, hvd.size()):
				if i != self.task_id:
					fsp_loss += self.norm_loss(matrix_group[idx], rec_fsp[i])
		fsp_loss /= (hvd.size()-1)
		self.log_dict['transfer_count'] += 1
		return fsp_loss

	def weights_statistic_transfer(self):
		mean_per_layers = []
		std_per_layers = []
		for name, param in self.network.named_parameters():
			if 'weight' in name:
				mean_per_layers.append(param.mean())
				std_per_layers.append(param.std())

		tensor_mean = torch.tensor(mean_per_layers).view([1,1,-1]).cuda(self.device)
		tensor_std = torch.tensor(std_per_layers).view([1,1,-1]).cuda(self.device)
		statistic = torch.cat([tensor_mean, tensor_std], dim=1)

		rec_statistic = hvd.allgather(statistic).cuda(self.device)
		ws_loss = 0
		for i in range(0, hvd.size()):
			if i != self.task_id:
				ws_loss += self.norm_loss(tensor_mean, rec_statistic[i][0])\
				           +self.norm_loss(tensor_std, rec_statistic[i][1])
		ws_loss /= (hvd.size()-1)

		self.log_dict['transfer_count'] += 1
		return ws_loss

	def weights_transfer(self):
		# transfer model weights
		weights = copy.deepcopy(self.network.state_dict())
		handles = []
		for name in weights:
			# TODO: need to consider bias
			if 'weight' in name:
				# print(self.task_id, 'send', name)
				handle = hvd.allgather_async(weights[name], name)
				handles.append(handle)

		hidx = 0
		for name, param in self.network.named_parameters():
			if 'weight' in name:
				# print(self.task_id, 'rec', name)
				rec_weights = hvd.synchronize(handles[hidx])
				hidx += 1
				# print(rec_weights.shape)

				n_num = param.shape[0]
				rec_weights = list(torch.split(rec_weights, n_num, 0))

				del rec_weights[self.task_id]

				# TODO weights cat in the first dim, 2*[64,3]--> [128,3]
				# logging.info(type(rec_weights), rec_weights.shape)
				# calculate IOM of each filter
				im_list = []
				for i in range(param.shape[0]):
					im_list.append(torch.sum(torch.abs(param[i])).data.cpu().numpy())
				im_list = np.array(im_list)
				# print('minimal weight sum is {} size {}'.format(im_list.min(), im_list.shape[0]))

				for i, im in enumerate(im_list):
					prob = 1 - stats.norm(0, 2).cdf(im)
					if np.random.rand() < prob:
						random_sender = np.random.randint(0, len(rec_weights))
						new_param = rec_weights[random_sender].clone()
						# random pic
						random_filter = np.random.randint(0, new_param.shape[0])
						# TODO give larger weights more chance
						weights[name][i] = new_param[random_filter]
						self.log_dict['transfer_count'] += 1
			# self.network.state_dict()[name].copy_(param.clone())
			# TODO: maybe modify the optimizer
		self.network.load_state_dict(weights)
		hvd.allreduce(torch.zeros(1), name='Barrier')

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
