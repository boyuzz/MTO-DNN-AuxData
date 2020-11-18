from collections import OrderedDict
from sklearn import metrics
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

import model.cifar as cifar
import model.mnist as mnist
import model.imagenet as imagenet
import random
# from model.networks import init_weights

from .base_task import BaseTask


class Archive:
	def __init__(self, size=0):
		self.size = size
		self.vector = []

	def push(self, item):
		if len(self.vector) >= self.size > 0:
			self.vector.pop(0)
		self.vector.append(item)

	def get(self):
		return random.choice(self.vector)


class ClassificationTask(BaseTask):
	def __init__(self, opt, **kwargs):
		super(ClassificationTask, self).__init__(opt, kwargs["comm"], kwargs["device"])
		train_opt = opt['train']
		self.ran_cl = opt['rancl']
		self.num_classes = opt['datasets']['train']['num_classes']
		self.device = torch.device("cuda:{}".format(kwargs['device']) if torch.cuda.is_available() else "cpu")
		self.archive = Archive(2)

		# -----define network and load pretrained tasks-----
		data_name, model_name = opt['network'].split('-')
		self.model_name = model_name
		if data_name.lower() == 'mnist':
			self.network = getattr(mnist, model_name)(num_classes=self.num_classes).to(self.device)
		elif data_name.lower() == 'cifar':
			self.network = getattr(cifar, model_name)(num_classes=self.num_classes).to(self.device)
		elif data_name.lower() == 'imagenet':
			if opt['imagenet_pretrained']:
				self.network = getattr(imagenet, model_name)(pretrained=True)
				if opt['train_lastlayer']:
					for param in self.network.parameters():
						param.requires_grad = False

				if 'resnet' in model_name or 'inception' in model_name:
					self.network.fc = nn.Linear(self.network.fc.in_features, self.num_classes)
				elif 'vgg' in model_name or 'alex' in model_name:
					self.network.classifier[6] = nn.Linear(4096, self.num_classes)
				elif 'squeeze' in self.model_name:
					self.network.num_classes = self.num_classes
					self.network.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=1)
				elif 'dense' in self.model_name:
					num_features = self.network.classifier.in_features
					self.network.classifier = nn.Linear(num_features, self.num_classes)
				elif 'mobile' in self.model_name:
					num_features = self.network.classifier[-1].in_features
					self.network.classifier[-1] = nn.Linear(num_features, self.num_classes)

				self.network = self.network.to(self.device)
			else:
				self.network = getattr(imagenet, model_name)(num_classes=self.num_classes).to(self.device)
		else:
			raise NotImplementedError('Network [{:s}, {:s}] is not defined.'.format(data_name, model_name))

		# load pretrained model if exists
		if self._is_solver():
			self.load()
			# print network
			# self.print_network()

		# -----define loss function------
		self.one_hot = False
		self.prob_est = False
		loss_type = train_opt['loss']
		if loss_type == 'l1':
			self.loss_func = nn.L1Loss().to(self.device)
			self.one_hot = True
		elif loss_type == 'l2':
			self.loss_func = nn.MSELoss().to(self.device)
			self.one_hot = True
		elif loss_type == 'l1_pro':
			self.loss_func = nn.L1Loss().to(self.device)
			self.prob_est = True
			self.one_hot = True
		elif loss_type == 'l2_pro':
			self.loss_func = nn.MSELoss().to(self.device)
			self.prob_est = True
			self.one_hot = True
		elif loss_type == 'cross_entropy':
			self.loss_func = nn.CrossEntropyLoss().to(self.device)
		elif loss_type == 'marginloss':
			self.loss_func = nn.MultiMarginLoss().to(self.device)
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

	def set_requires_grad(self, is_grad):
		for k, p in self.network.named_parameters():
			if 'fc' not in k and 'classifier' not in k:
				p.requires_grad = is_grad

	def _convert_int_onehot(self, labels):
		# One hot encoding buffer that you create out of the loop and just keep reusing
		labels = labels.view(-1, 1)
		y_onehot = torch.zeros(labels.size(0), self.num_classes).to(self.device)

		# In your for loop
		y_onehot.scatter_(1, labels, 1)
		return y_onehot

	def step(self):
		(data, target) = next(self.train_iter)
		data, target = data.to(self.device), target.to(self.device)
		# self.logger.info(target[:10])
		if self.one_hot:
			target = self._convert_int_onehot(target)

		self.optimizer.zero_grad()

		logits = self.network(data)

		if self.prob_est:
			logits = F.softmax(logits, dim=1)
		loss = self.loss_func(logits, target)
		# set log
		self.log_dict['Training_loss'] = loss.item()

		loss.backward()
		self.optimizer.step()
		self.update_learning_rate()

	def full_training(self, log_freq, val_freq, save_freq, tb_logger=None):
		while True:
			epoch = self.training_step // len(self.train_loader)
			self.training_step += 1
			if self.training_step > self.total_iters:
				# [self.comm.send(-1, dest=i, tag=1024) for i in self.task_explorer_ranks]
				self.logger.info('Reach maximal steps')
				return

			self.step()

			# log
			if self.training_step % log_freq == 0:  # and self.rank == self.task_exploit_rank
				logs = self.get_current_log()

				message = '<Task {}, rank {}, iter:{:3d} lr:{:.3e}, task:{}> '.format(
					self.task_id, self.rank, self.training_step, self.get_current_learning_rate(), self.task_id)

				for k, v in logs.items():
					message += '{:s}: {:.4e} '.format(k, v)
					if tb_logger is not None:
						tb_logger.add_scalar(k, v, self.training_step)

				if tb_logger is not None:
					tb_logger.add_scalar('lr', self.get_current_learning_rate(), self.training_step)

				self.logger.info(message)

			# save tasks and training states
			if self.training_step % save_freq == 0:
				self.logger.info('Saving tasks and training states for task {}.'.format(self.task_id))
				self.save(self.training_step)
				self.save_training_state(epoch, self.training_step)

			# validation
			if self.training_step % val_freq == 0:
				self.solver_update()

	def solver_update(self):
		# validation
		all_hmean = self.cv_validation()
		results = [r['acc'] for r in all_hmean]
		idx = np.argmax(results)
		best_task_id = all_hmean[idx]['id']

		if self.task_id == best_task_id:
			best_model = {'acc': max(results), 'model': self.network.state_dict(),
						  'optim': self.optimizer.state_dict(), 'scheduler': self.schedulers}
		else:
			best_model = None

		best_model = self.comm.bcast(best_model, root=best_task_id)
		self.archive.push(best_model)

		select_model = self.archive.get()
		acc = select_model['acc']
		self.network.load_state_dict(select_model['model'])
		self.optimizer.load_state_dict(select_model['optim'])

		for i, scheduler in enumerate(select_model['scheduler']):
			if scheduler is not None:
				self.schedulers[i].load_state_dict(scheduler.state_dict())

		step = 'last' if self.training_step == self.total_iters else self.training_step
		message = '<Task {}-rank {}, # Iter: {} Accuracy: {:.4e},>'.format(self.task_id, self.rank, step, acc)
		self.logger.info(message)

	def cv_validation(self):
		from scipy import stats
		# self.network.load_state_dict(self.best_weights.state_dict())

		folds_accuracy = []

		self.network.eval()
		with torch.no_grad():
			for loader in self.folds_loaders:
				test_loss = 0.
				full_preds = []
				full_labels = []
				full_logits = []
				for data, target in loader:
					data, target = data.to(self.device), target.to(self.device)
					labels = target.data
					if self.one_hot:
						target = self._convert_int_onehot(target)

					logits = self.network(data)

					if self.prob_est:
						logits = F.softmax(logits, dim=1)
					full_logits.append(logits)

					test_loss += self.loss_func(logits, target).item()
					# get the index of the max log-probability
					pred = logits.data.max(1, keepdim=True)[1]
					full_preds.extend(pred.view(-1).cpu())
					full_labels.extend(labels.cpu())

				test_accuracy = 100 * metrics.accuracy_score(np.array(full_labels), np.array(full_preds))
				folds_accuracy.append(test_accuracy)

		hmean = stats.hmean(folds_accuracy)
		self.logger.info('# Task {} # Harmonic Accuracy: {:.2f}%'.format(self.task_id, hmean))
		result = {'id':self.task_id, 'acc':hmean}
		all_hmean = self.exploit_comm.allgather(result)

		return all_hmean

	def validation(self, **kwargs):
		# if 'best' in kwargs.keys() and kwargs['best'] and self.best_weights is not None:
		# 	self.network.load_state_dict(self.best_weights.state_dict())

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
				data, target = data.to(self.device), target.to(self.device)
				# bs, ncrops, c, h, w = data.size()
				labels = target.data
				if self.one_hot:
					target = self._convert_int_onehot(target)

				logits = self.network(data)
				# result = self.network(data.view(-1, c, h, w))
				# result_avg = result.view(bs, ncrops, -1).mean(1)
				if self.prob_est:
					logits = F.softmax(logits, dim=1)
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

		if 'ensemble_peer' in kwargs.keys() and kwargs['ensemble_peer'] and self._is_solver():
			full_logits = torch.cat(tuple(full_logits), 0)
			rec_logits = self.exploit_comm.allgather(full_logits)
			acc_array = np.zeros(len(rec_logits))
			for i, r in enumerate(rec_logits):
				ave_logits = (full_logits+r)/2
				ensemble_pred = ave_logits.data.max(1, keepdim=True)[1].view(-1).cpu()
				acc_array[i] = 100 * metrics.accuracy_score(np.array(full_labels), np.array(ensemble_pred))
			# ids = acc_array.argsort()[::-1][:len(self.task_explorer_ranks)]
			self.most_related_task = acc_array.argsort()[::-1][:len(self.task_explorer_ranks)]
			self.logger.info("task {} ensemble peer accuracy: {}, most related {}".format(self.task_id, acc_array, self.most_related_task))
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
