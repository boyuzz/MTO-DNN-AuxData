from .base_task import BaseTask
from dataloader import create_dataloader, create_dataset
import math
from sklearn import metrics
import numpy as np

from collections import OrderedDict
import torch.nn as nn
import torch
from torch import optim
from torch.optim import lr_scheduler

import model.imagenet as imagenet
import copy


class ClassificationMTLTask(BaseTask):
	def __init__(self, opt, **kwargs):
		super(ClassificationMTLTask, self).__init__(opt, kwargs["comm"], kwargs["device"], False)
		# Hard coding!!!!!!
		self.resample = 2
		self.create_dataset()

		self.num_classes = opt['datasets']['train']['num_classes']
		train_opt = opt['train']

		self.best_metric = None
		self.best_weights = None
		self.wait = 0
		self.stop_training = False
		self.patience = opt['patience']

		# build MTL models: ResNet-18, AlexNet, VGG-16, ResNet-50, DenseNet-121
		# -----define network and load pretrained tasks-----
		data_name, model_name = opt['network'].split('-')
		self.model_name = model_name

		# self.network.layer4[-1].register_forward_hook(self.get_activation('b4_out'))
		if data_name.lower() == 'imagenet':
			if opt['imagenet_pretrained']:
				self.network = getattr(imagenet, model_name)(pretrained=True, branches=[self.num_classes,self.num_classes,self.num_classes,self.num_classes])

				self.network = self.network.cuda(self.device)
			else:
				self.network = getattr(imagenet, model_name)(num_classes=self.num_classes).cuda(self.device)

		else:
			raise NotImplementedError('Network [{:s}, {:s}] is not defined.'.format(data_name, model_name))

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
		elif loss_type == 'marginloss':
			self.loss_func = nn.MultiMarginLoss().cuda(self.device)
		else:
			raise NotImplementedError(
				'Loss type [{:s}] is not recognized. Please specifiy it from following options:'.format(loss_type))

		if self.is_train:
			self.network.train()
			# -----define optimizers-----
			optim_type = train_opt['optim']

			self.optimizer = getattr(optim, optim_type)(self.network.parameters(), **opt['train']['optimizer_param'])
			self.optimizers.append(self.optimizer)
			# self.lr = opt['train']['optimizer_param']['lr']

			# -----define schedulers-----
			for optimizer in self.optimizers:
				if train_opt['lr_scheme'] == 'MultiStepLR':
					scheduler = lr_scheduler.MultiStepLR(optimizer, **opt['train']['lr_scheme_param'])
				elif train_opt['lr_scheme'] == 'CycleLR':
					scheduler = lr_scheduler.CyclicLR(optimizer, **opt['train']['lr_scheme_param'])
				elif train_opt['lr_scheme'] == 'ReduceLROnPlateau':
					scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **opt['train']['lr_scheme_param'])
				elif train_opt['lr_scheme'] is None:
					scheduler = None
				else:
					raise NotImplementedError('{} is not implemented!'.format(train_opt['lr_scheme']))
				self.schedulers.append(scheduler)

			# # -----register gradient clipping-----
			# for param in self.network.parameters():
			# 	param.register_hook(lambda grad: torch.clamp(grad, -0.2, 0.2))

			# -----define log_dict-----
			self.log_dict = OrderedDict()
			self.transfer_count = 0

	def create_dataset(self):
		# create train and val dataloader
		for phase, dataset_opt in self.opt['datasets'].items():
			if phase == 'train':
				self.train_set = create_dataset(dataset_opt, split=phase)
				dataset_opt['resample'] = self.resample
				dataset_opt['batch_size'] = max(dataset_opt['batch_size']//8, 1)
				aux_set1 = create_dataset(dataset_opt, split=phase)
				aux_set2 = create_dataset(dataset_opt, split=phase)
				aux_set3 = create_dataset(dataset_opt, split=phase)
				train_size = int(math.ceil(len(self.train_set) / dataset_opt['batch_size']))
				self.logger.info('rank {}, Number of train images: {:,d}, iters per epoch: {:,d}'.format(self.rank,
				                                                                                         len(
					                                                                                         self.train_set),
				                                                                                         train_size))
				total_iters = int(self.opt['niter'])
				total_epochs = int(math.ceil(total_iters / train_size))
				self.logger.info('rank {}, Total epochs needed: {:d} for iters {:,d}'.format(self.rank,
				                                                                             total_epochs, total_iters))
				self.train_loader = create_dataloader(self.train_set, dataset_opt)
				aux_loader1 = create_dataloader(aux_set1, dataset_opt)
				aux_loader2 = create_dataloader(aux_set2, dataset_opt)
				aux_loader3 = create_dataloader(aux_set3, dataset_opt)
				self.train_iter = iter(self._cycle(self.train_loader))
				aux_iter1 = iter(self._cycle(aux_loader1))
				aux_iter2 = iter(self._cycle(aux_loader2))
				aux_iter3 = iter(self._cycle(aux_loader3))
				self.iters = [self.train_iter, aux_iter1, aux_iter2, aux_iter3]

			elif phase == 'val':
				val_set = create_dataset(dataset_opt, split=phase)
				self.val_loader = create_dataloader(val_set, dataset_opt)
				self.logger.info('rank {}, Number of val images in [{:s}]: {:d}'.format(self.rank, dataset_opt['name'],
				                                                                        len(val_set)))
			elif phase == 'test':
				test_set = create_dataset(dataset_opt, split=phase)
				self.test_loader = create_dataloader(test_set, dataset_opt)
				self.logger.info('rank {}, Number of test images in [{:s}]: {:d}'.format(self.rank, dataset_opt['name'],
				                                                                         len(test_set)))
			else:
				raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
		assert self.train_loader is not None
		# assert self.val_loader is not None
		self.total_epochs = total_epochs
		self.total_iters = total_iters

	def step(self):
		loss = 0
		self.optimizer.zero_grad()
		for i, iter in enumerate(self.iters):
			(data, target) = next(iter)
			data, target = data.cuda(self.device), target.cuda(self.device)
			# self.logger.info(target[:10])

			logits = self.network(data, i)

			loss += self.loss_func(logits, target)
		loss /= len(self.iters)
		# set log
		self.log_dict['Training_loss'] = loss.item()

		loss.backward()
		self.optimizer.step()
		self.update_learning_rate()

	def full_training(self, log_freq, val_freq, save_freq, tb_logger=None):
		while True:
			# if self._is_solver():
			epoch = self.training_step // len(self.train_loader)
			self.training_step += 1
			if self.training_step > self.total_iters:
				# [self.comm.send(-1, dest=i, tag=1024) for i in self.task_explorer_ranks]
				self.logger.info('Reach maximal steps')
				return

			if self.rank == 0 and self.stop_training:
				stop = True
			else:
				stop = False

			if stop:
				self.logger.info('Early stop!')
				return

			self.step()

			# log
			if self.training_step % log_freq == 0:  # and self.rank == self.task_exploit_rank
				logs = self.get_current_log()

				message = '<Task {}, rank {}, iter:{:3d} lr:{:.3e}, task:{}> '.format(
					self.task_id, self.rank, self.training_step, self.get_current_learning_rate(),
					self.task_id)

				for k, v in logs.items():
					message += '{:s}: {:.4e} '.format(k, v)
					if tb_logger is not None:
						tb_logger.add_scalar(k, v, self.training_step)

				if tb_logger is not None:
					tb_logger.add_scalar('lr', self.get_current_learning_rate(), self.training_step)

				self.logger.info(message)

			if self.training_step % save_freq == 0:
				self.logger.info('Saving tasks and training states for task {}.'.format(self.task_id))
				self.save(self.training_step)
				self.save_training_state(epoch, self.training_step)

			if self.training_step % val_freq == 0 and self.val_loader is not None:
				self.solver_update(tb_logger)

	def solver_update(self, tb_logger):
		# validation
		if self.val_loader is not None:
			test_loss, test_accuracy, l2_reg = self.validation(verbose=False, split=self.opt['val_split'])

			if self.best_metric is None or self.best_metric > test_loss:
				self.best_metric = test_loss
				self.best_weights = copy.deepcopy(self.network)
				self.wait = 0
			else:
				self.wait += 1
				if self.wait >= self.patience:
					self.stop_training = True

			step = 'last' if self.training_step == self.total_iters else self.training_step
			message = '<Task {}-rank {}, # Iter: {} Lowest loss: {:.4e}, Loss: {:.4e}, Accuracy: {:.2f}%, l2_reg: {:.4f}, Transfer Count: {}>'.format(
				self.task_id, self.rank, step, self.best_metric, test_loss, test_accuracy, l2_reg, self.transfer_count)
			self.logger.info(message)

			if self._is_solver():
				if self.opt['use_tb_logger']:
					tb_logger.add_scalar('Test_loss', test_loss, self.training_step)
					tb_logger.add_scalar('Accuracy', test_accuracy, self.training_step)
					tb_logger.add_scalar('L2', l2_reg, self.training_step)

	def validation(self, **kwargs):
		if 'best' in kwargs.keys() and kwargs['best'] and self.best_weights is not None:
			self.network.load_state_dict(self.best_weights.state_dict())

		self.network.eval()
		test_loss = 0.
		# test_accuracy = 0.
		full_preds = []
		full_labels = []
		full_logits = []

		if 'split' in kwargs.keys() and kwargs['split'] == 'test' and self.test_loader is not None:
			dataloader = self.test_loader
		elif 'split' in kwargs.keys() and kwargs['split'] == 'train' and self.train_loader is not None:
			dataloader = self.train_loader
		else:
			dataloader = self.val_loader

		with torch.no_grad():
			for data, target in dataloader:
				data, target = data.cuda(self.device), target.cuda(self.device)
				# bs, ncrops, c, h, w = data.size()
				labels = target.data

				logits = self.network(data)

				full_logits.append(logits)
				# sum up batch loss
				test_loss += self.loss_func(logits, target).item()
				# get the index of the max log-probability
				pred = logits.data.max(1, keepdim=True)[1]
				# full_preds.extend(pred.view(-1).cpu())
				full_preds.extend(pred.view(-1).cpu())
				full_labels.extend(labels.cpu())
		# test_accuracy += pred.eq(labels.view_as(pred)).cpu().float().sum()

		test_loss /= len(dataloader)
		# test_accuracy /= len(self.testing_set)
		test_accuracy = 100 * metrics.accuracy_score(np.array(full_labels), np.array(full_preds))
		test_f1 = metrics.f1_score(np.array(full_labels), np.array(full_preds), average='macro')
		l2_reg = torch.tensor(0.)
		for param in self.network.parameters():
			l2_reg += torch.norm(param)
		# test_roc_auc = metrics.roc_auc_score(np.array(full_labels), np.array(full_preds))
		if 'verbose' in kwargs.keys() and kwargs['verbose']:
			self.logger.info(
				'# Task {} # Validation loss: {:.2f}, Accuracy: {:.2f}%, F1 score: {:.4f}'.format(self.task_id,
				                                                                                  test_loss,
				                                                                                  test_accuracy,
				                                                                                  test_f1))
		if 'report' in kwargs.keys() and kwargs['report']:
			from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
			cm = confusion_matrix(full_labels, full_preds)
			report = classification_report(np.array(full_labels), np.array(full_preds), digits=4)

			self.logger.info("overall accuracy: {}, F1: {}.".format(test_accuracy, test_f1))
			self.logger.info("confusion matrix\n {}.".format(cm))
			self.logger.info("classification report\n {}.".format(report))
		self.network.train()
		return test_loss, test_accuracy, l2_reg

	def get_current_log(self):
		return self.log_dict

	def save(self, iter_step):
		if self.best_weights is not None:
			self.save_network(self.best_weights, iter_step)
