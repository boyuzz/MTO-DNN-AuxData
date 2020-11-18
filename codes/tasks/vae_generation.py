import logging
import model.vae as vae
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.optim import lr_scheduler
from sklearn.svm import LinearSVC
from sklearn import metrics

from collections import OrderedDict

try:
	import horovod.torch as hvd
	USE_HVD = True
except ImportError:
	USE_HVD = False

from .base_task import BaseTask


class VaeGenerationTask(BaseTask):
	def __init__(self, opt, comm, device):
		super(VaeGenerationTask, self).__init__(opt, comm, device)
		# TODO 到底怎么交互？？？
		train_opt = opt['train']
		self.training_step = 0

		# -----model-----
		data_name, model_name = opt['network'].split('-')
		self.network = getattr(vae, model_name)(**opt['param']).cuda(self.device)

		# ---load pretrained model if exists---
		self.load()

		# -----loss------
		# loss_type = train_opt['loss']
		# self.loss_func = self.loss_function

		# -----optimizer-----
		optim_params = []
		for k, v in self.network.named_parameters():  # can optimize for a part of the model
			if v.requires_grad:
				optim_params.append(v)
			else:
				self.logger.warning('Params [{:s}] will not optimize.'.format(k))
		optim_type = train_opt['optim']
		self.optimizer = getattr(optim, optim_type)(self.network.parameters(), **opt['train']['optimizer_param'])
		self.optimizers.append(self.optimizer)

		if train_opt['lr_scheme'] == 'MultiStepLR':
			for optimizer in self.optimizers:
				self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
				                                                train_opt['lr_steps'], train_opt['lr_gamma']))
		else:
			raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

		# -----define log_dict-----
		self.log_dict = OrderedDict()
		self.log_dict['transfer_count'] = 0

	def loss_function(self, recon_x, x, mu, logvar):
		BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.opt['param']['input_size']), reduction='sum')

		# see Appendix B from VAE paper:
		# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
		# https://arxiv.org/abs/1312.6114
		# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

		return BCE + KLD

	def full_training(self, log_freq, val_freq, save_freq, tb_logger=None):
		self.network.train()
		train_loss = 0
		# minibatch_size = self.opt['datasets']['train']['batch_size']

		for epoch in range(self.start_epoch+1, self.total_epochs+1):
			for data in self.train_loader:
				self.training_step += 1
				data = data.cuda(self.device)
				self.optimizer.zero_grad()
				recon_batch, self.mu, self.logvar = self.network(data)
				loss = self.loss_function(recon_batch, data, self.mu, self.logvar)
				loss.backward()
				train_loss += loss.item()
				self.log_dict['Training_loss'] = loss.item()
				self.optimizer.step()

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

					# if tb_logger is not None:
					# 	for name, param in self.network.named_parameters():
					# 		tb_logger.add_histogram(name, param.clone().cpu().data.numpy(), self.training_step)
					self.logger.info(message)

				# validation
				if self.training_step % val_freq == 0:
					# log
					test_accuracy = self.validation()
					step = 'last' if self.training_step == self.total_iters else self.training_step
					self.logger.info(
						'<Task {}, epoch:{:3d}, # Iter: {} Accuracy: {:.2f}%>'.format(
							self.task_id, epoch, step, test_accuracy * 100))

					# tensorboard self.logger
					if tb_logger is not None:
						tb_logger.add_scalar('GAN_Accuracy', test_accuracy, self.training_step)

				# save tasks and training states
				if self.training_step % save_freq == 0:
					self.logger.info('Saving tasks and training states for task {}.'.format(self.task_id))
					self.save(self.training_step)
					self.save_training_state(epoch, self.training_step)


		self.logger.info("knowledge transferred {:d} times".format(self.log_dict['transfer_count']))

	def val(self):
		self.network.eval()
		test_loss = 0
		with torch.no_grad():
			for i, (data, _) in enumerate(self.val_loader):
				data = data.cuda(self.device)
				recon_batch, mu, logvar = self.network(data)
				test_loss += self.loss_function(recon_batch, data, mu, logvar).item()
				if i == 0:
					n = min(data.size(0), 8)
					comparison = torch.cat([data[:n],
					                        recon_batch.view(self.opt['datasets']['train']['batch_size'], 1, 28, 28)[:n]])

		test_loss /= len(self.val_loader)
		print('====> Test set loss: {:.4f}'.format(test_loss))

	def validation(self, **kwargs):
		self.network.eval()

		with torch.no_grad():
			z = self.network.reparameterize(self.mu, self.logvar)

			d_fake_data = self.network.decode(z)
			fake_data = d_fake_data.data.cpu().numpy()
			X = fake_data[:, :-1]
			y = np.round(fake_data[:, -1]).astype(np.int)
			clf = LinearSVC()
			clf.fit(X, y)

			full_preds = []
			full_labels = []
			for sample in self.train_loader:
				data = sample.data.cpu().numpy()
				X = data[:, :-1]
				y = data[:, -1]
				pred = clf.predict(X)
				full_preds.extend(pred)
				full_labels.extend(y)

		test_accuracy = metrics.accuracy_score(np.array(full_labels), np.array(full_preds))
		if 'verbose' in kwargs and kwargs['verbose']:
			self.logger.info('# Task {}  # Validation Accuracy: {:.2f}%'.format(self.task_id, test_accuracy * 100))
		return test_accuracy

	def get_current_log(self):
		return self.log_dict

	def get_distribution_sampler(self, mu, sigma):
		return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian

	def get_generator_input_sampler(self):
		return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian

	def load(self):
		load_path = self.opt['path']['pretrain_model']
		if load_path is not None:
			self.logger.info('Loading pretrained model [{:s}] ...'.format(load_path))
			self.load_network(load_path, self.network)

	def save(self, iter_step):
		self.save_network(self.network, iter_step)
