from collections import OrderedDict
from sklearn import metrics
import numpy as np
from scipy import special
from mpi4py import MPI

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

import model.cifar as cifar
import model.mnist as mnist
import model.imagenet as imagenet
import warnings
import copy
# from model.networks import init_weights

from .base_task import BaseTask


class ClassificationTask(BaseTask):
	def __init__(self, opt, **kwargs):
		super(ClassificationTask, self).__init__(opt, kwargs["comm"], kwargs["device"])
		train_opt = opt['train']
		self.ran_cl = opt['rancl']
		self.num_classes = opt['datasets']['train']['num_classes']
		self.kd_transfer = opt['kd_transfer']
		self.att_transfer = opt['att_transfer']
		self.fsp_transfer = opt['fsp_transfer']
		self.w_transfer = opt['w_transfer']
		self.ws_transfer = opt['ws_transfer']
		self.replace_classifier = opt['varyOnData']
		# self.device = kwargs['device']

		self.device = torch.device("cuda:{}".format(kwargs['device']) if torch.cuda.is_available() else "cpu")
		# self.logger.info(self.device)
		# -----early stopping-------
		self.best_weights = None
		self.best_metric = None
		self.wait = 0
		self.stop_training = False
		self.patience = opt['patience']

		# -----prepare for transfer-------------
		self.most_related_task = -1

		if self.fsp_transfer or self.att_transfer:
			self.activation = OrderedDict()

		# -----define network and load pretrained tasks-----
		data_name, model_name = opt['network'].split('-')
		self.model_name = model_name
		if data_name.lower() == 'mnist':
			self.network = getattr(mnist, model_name)(num_classes=self.num_classes).to(self.device)
		elif data_name.lower() == 'cifar':
			self.network = getattr(cifar, model_name)(num_classes=self.num_classes).to(self.device)
			if self.att_transfer and 'resnet' in model_name.lower():
				self.network.layer1[-1].register_forward_hook(self.get_activation('b1_out'))
				self.network.layer2[-1].register_forward_hook(self.get_activation('b2_out'))
				self.network.layer3[-1].register_forward_hook(self.get_activation('b3_out'))
				# self.network.layer4[-1].register_forward_hook(self.get_activation('b4_out'))
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

			if self.att_transfer:
				if 'resnet' in self.model_name.lower():
					self.network.layer1[-1].register_forward_hook(self.get_activation('b1_out'))
					self.network.layer2[-1].register_forward_hook(self.get_activation('b2_out'))
					self.network.layer3[-1].register_forward_hook(self.get_activation('b3_out'))
					# self.network.layer4[-1].register_forward_hook(self.get_activation('b4_out'))
				# elif 'dense' in self.model_name.lower():
		else:
			raise NotImplementedError('Network [{:s}, {:s}] is not defined.'.format(data_name, model_name))

		# make starts the same
		# if USE_HVD:
		# 	hvd.broadcast_parameters(self.network.state_dict(), root_rank=0)
		# test if different task has the same initialization under same seed
		# for name, param in self.network.named_parameters():
		# 	print(param[0])

		# load pretrained model if exists
		if self._is_solver():
			# init_weights(self.network)
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

			self.logits_loss = nn.KLDivLoss(reduction='batchmean').to(self.device)
			self.norm_loss = nn.MSELoss(reduction='batchmean').to(self.device)
			self.at_weight = train_opt['at_weight']
			self.kd_weight = train_opt['kd_weight']
			self.ws_weight = train_opt['ws_weight']

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

			# -----prepare for transfer-----
			if self.kd_transfer or self.att_transfer:
				# set seeds of all tasks the same to ensure the dataloader is in the same order
				torch.manual_seed(0)

	# def update_learning_rate(self):
	# 	steps = np.array([self.opt['niter']*0.2, self.opt['niter']*0.7])
	#
	# 	if self._is_solver():
	# 		self.lr = self.lr * (0.1 ** sum(self.training_step>steps))
	# 	else:
	# 		self.lr = self.lr
	#
	# 	for param_group in self.optimizer.param_groups:
	# 		param_group["lr"] = self.lr

	def get_activation(self, name):
		def hook(model, input, output):
			self.activation[name] = output
		return hook

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

		# apply attention_transfer
		if self.num_tasks > 1 and self.att_transfer:  # self.num_tasks > 1 and
			att_loss = self.attention_distillation()
			self.log_dict['att_loss'] = att_loss.item()
			loss += self.at_weight * att_loss

		# apply knowledge distillation  self.num_tasks > 1 and  or self.w_transfer
		if self.num_tasks > 1 and self.kd_transfer:
			kd_loss = self.knowledge_distillation_transfer(logits,
			                                               in_group=True if self.w_transfer else False)
			self.log_dict['kd_loss'] = kd_loss.item()
			loss += self.kd_weight * kd_loss

		loss.backward()
		self.optimizer.step()
		self.update_learning_rate()
		# for name, param in self.network.named_parameters():
		# 	print(param[0])
		# 	break

	def full_training(self, log_freq, val_freq, save_freq, tb_logger=None):
		# epoch = self.start_epoch
		if self.w_transfer:
			self.weights_allocate()

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
			stop = self.comm.bcast(stop, root=0)
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

			# if self.training_step % val_freq == 0 and self.world_size > 1:
			# 	[self.comm.send(self.training_step, dest=i, tag=1024) for i in self.task_explorer_ranks]

			# save tasks and training states
			if self.training_step % save_freq == 0:
				self.logger.info('Saving tasks and training states for task {}.'.format(self.task_id))
				self.save(self.training_step)
				self.save_training_state(epoch, self.training_step)
			# else:
			# 	req = self.comm.irecv(source=self.task_solver_rank, tag=1024)
			# 	count = 0
			# 	while True:
			# 		count += 1
			# 		found, self.training_step = req.test()
			# 		if self.training_step == -1:
			# 			self.logger.info('Reach maximal steps')
			# 			return
			#
			# 		if found:
			# 			self.logger.info('rank {} updated {} times.'.format(self.rank, count))
			# 			break
			# 		self.step()

			# validation
			if self.training_step % val_freq == 0 and hasattr(self, 'val_loader'):
				# log
				self.solver_update(tb_logger)

			if self.training_step % val_freq == 0 and self.w_transfer:
				self.weights_allocate()

	def solver_update(self, tb_logger):
		# validation
		if self.val_loader is not None:
			test_loss, test_accuracy, l2_reg = self.validation(verbose=False, split=self.opt['val_split'])

			if self.w_transfer and self.world_size > len(self.task_explorer_ranks) and len(self.task_explorer_ranks) > 0:
				if not self._is_solver():
					req = self.comm.isend({'val': test_accuracy, 'l2': l2_reg, 'loss': test_loss, 'model': self.network.state_dict(),
									  'optim': self.optimizer.state_dict(), 'scheduler': self.schedulers}, dest=self.task_solver_rank, tag=2048)
					req.wait()
				else:
					val_model_dicts = [self.comm.recv(source=i, tag=2048) for i in self.task_explorer_ranks]
					all_val = np.array([vmd['loss'] for vmd in val_model_dicts])

					if not self.ran_cl:
						corr_delta = np.tanh(test_loss - all_val)
						for i, cd in enumerate(corr_delta):
							self.corr_list[self.selected_tasks[i]] += cd
							# self.corr_list[self.selected_tasks[i]] = 0.8 * self.corr_list[self.selected_tasks[i]] + cd

					if all_val.min() < test_loss:
						self.network.load_state_dict(val_model_dicts[all_val.argmin()]['model'])
						self.set_requires_grad(True)

						self.optimizer.load_state_dict(val_model_dicts[all_val.argmin()]['optim'])
						# optim_type = self.opt['train']['optim']
						# optim_param = self.opt['train']['optimizer_param']
						# optim_param['lr'] = self.lr
						# self.optimizer = getattr(optim, optim_type)(self.network.parameters(), **optim_param)

						for i, scheduler in enumerate(val_model_dicts[all_val.argmin()]['scheduler']):
							if scheduler is not None:
								self.schedulers[i].load_state_dict(scheduler.state_dict())

						self.transfer_count += 1
						test_loss = val_model_dicts[all_val.argmin()]['loss']
						test_accuracy = val_model_dicts[all_val.argmin()]['val']
						l2_reg = val_model_dicts[all_val.argmin()]['l2']

			if self.best_metric is None or self.best_metric > test_loss:
				self.best_metric = test_loss
				self.best_weights = copy.deepcopy(self.network)
				self.wait = 0
			else:
				self.wait += 1
				if self.wait >= self.patience > 0:
					self.stop_training = True

			step = 'last' if self.training_step == self.total_iters else self.training_step
			message = '<Task {}-rank {}, # Iter: {} Lowest loss: {:.4e}, Loss: {:.4e}, Accuracy: {:.2f}%, l2_reg: {:.4f}, Transfer Count: {}, corr_list: {}>'.format(
				self.task_id, self.rank, step, self.best_metric, test_loss, test_accuracy, l2_reg, self.transfer_count,
				self.corr_list)
			self.logger.info(message)

			if self._is_solver():
				if self.opt['use_tb_logger']:
					tb_logger.add_scalar('Test_loss', test_loss, self.training_step)
					tb_logger.add_scalar('Accuracy', test_accuracy, self.training_step)
					tb_logger.add_scalar('L2', l2_reg, self.training_step)
					for i, cl in enumerate(self.corr_list):
						if i >= self.task_id:
							relate_task_id = i + 1
						else:
							relate_task_id = i

						tb_logger.add_scalar('task {} helpfulness'.format(relate_task_id), cl, self.training_step)

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
					# bs, ncrops, c, h, w = data.size()
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
					# full_preds.extend(pred.view(-1).cpu())
					full_preds.extend(pred.view(-1).cpu())
					full_labels.extend(labels.cpu())

				test_accuracy = 100 * metrics.accuracy_score(np.array(full_labels), np.array(full_preds))
				folds_accuracy.append(test_accuracy)

		hmean = stats.hmean(folds_accuracy)
		self.logger.info('# Task {} # Harmonic Accuracy: {:.2f}%'.format(self.task_id, hmean))
		result = {'id': self.task_id, 'acc': hmean}
		all_hmean = self.exploit_comm.allgather(result)

		return all_hmean

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

	def show_attention(self, image):
		def at(x):
			return F.normalize(x.pow(2).mean(1))

		from PIL import Image
		import torchvision.transforms.functional as TF
		import matplotlib.pyplot as plt

		full_img = Image.open(image)
		data = TF.resize(full_img, 256)
		data = TF.center_crop(data, 224)
		data = TF.to_tensor(data)
		data = TF.normalize(data, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
		data = data.unsqueeze(0)
		data = data.to(self.device)

		with torch.no_grad():
			logits = self.network(data)
			if self.prob_est:
				logits = F.softmax(logits, dim=1)
			# get the index of the max log-probability
			pred = logits.data.max(1, keepdim=True)[1][0][0].cpu().numpy() + 1
			# pred_five.append(pred.view(-1).cpu())
			att_group = []
			for i, key in enumerate(self.activation):
				at_out = at(self.activation[key]).cpu()
				# at_im = TF.to_pil_image(at_out.cpu())
				# at_im = at_im.resize((224, 224), resample=Image.BICUBIC)
				# at_im.show()
				# att_group.append(at_im)
				plt.imshow(at_out[0], interpolation='bicubic')
				plt.title(f'g{i}')
				plt.show()

	def attention_distillation(self):
		def at(x):
			return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

		att_group = []
		for key in self.activation:
			at_out = at(self.activation[key])
			att_group.append(at_out)

		rec_atts = self.exploit_comm.allgather(att_group)

		att_loss = 0
		for ra in rec_atts:
			# att_loss += self.norm_loss(att_group[idx], rec_att.mean(0).to(self.device_id))
			for i in range(0, len(att_group)):
				if i != self.task_id:
					# self.logger.info('tagtag: {} and {}'.format(att_group[i][0,:10], ra[i][0, :10]))
					att_loss += self.norm_loss(att_group[i], ra[i].to(self.device))

		att_loss /= (len(rec_atts) - 1)
		return att_loss

	def attention_transfer(self, data):
		def at(x):
			return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

		att_group = []
		for key in self.activation:
			at_out = at(self.activation[key])
			at_out = at_out.detach()
			att_group.append(at_out)

		batchsize = data.size(0)
		n_pick = max(1, batchsize//self.num_tasks)
		random_pick = torch.randperm(batchsize)[:n_pick]
		t_data = data[random_pick]
		att_group = [a[random_pick] for a in att_group]

		shared_knowledge = {"data": t_data, "attention": att_group, "id": self.task_id}

		rec_atts = self.exploit_comm.allgather(shared_knowledge)
		inputs = []
		targets = []
		sources = []
		for item in rec_atts:
			inputs.append(item['data'])
			targets.append(item['attention'])
			# targets.append(item['attention'])
			sources.append(item['id'])

		inputs = torch.cat(tuple(inputs), 0)
		targets = np.array(targets)
		targets = [torch.cat(tuple(targets[:, i]), 0) for i in range(targets.shape[1])]
		n_select = min(batchsize, inputs.size(0))
		random_pick = torch.randperm(inputs.size(0))[:n_select]
		rec_data = inputs[random_pick]
		rec_attention = [tg[random_pick] for tg in targets]

		_ = self.network(rec_data.to(self.device))
		att_loss = 0

		att_group = []
		for i in range(0, len(self.activation)):
			for key in self.activation:
				at_out = at(self.activation[key])
				att_group.append(at_out)
			if i != self.task_id:
				#self.logger.info("input device {}, output device {}".format(att_group[i].get_device(), rec_attention[i].to(self.device).get_device()))
				att_loss += self.norm_loss(att_group[i], rec_attention[i].to(self.device))

		att_loss /= (len(rec_atts) - 1)

		return att_loss

	def knowledge_distillation_transfer(self, logits, in_group=False):
		'''
		To use this function, make sure the input dataloader to each task is exactly the same,
		otherwise this function is meaningless.
		:return:
		'''
		T = 4
		# logits_data = logits.data
		logit_dict = {'rank': self.rank, 'logits': logits}
		if in_group:
			rec_knowledge = self.task_comm.allgather(logit_dict)
		else:
			rec_knowledge = self.exploit_comm.allgather(logit_dict)
		# self.logger.info('rec_knowledge length {}, {}'.format(len(rec_knowledge), [rc[0,:10] for rc in rec_knowledge]))

		kd_loss = 0
		# batchsize = int(rec_knowledge.size(0) // hvd.size())
		# self.logger.info("knowledge size {}, batchsize {}".format(rec_knowledge.size(), batchsize))
		# Note, the KDDivloss in pytorch says:
		# As with NLLLoss, the input given is expected to contain log-probabilities. However, unlike NLLLoss,
		# input is not restricted to a 2D Tensor. The targets are given as probabilities
		# (i.e. without taking the logarithm).
		# KLDivLoss: l_n = y_n*(log(y_n)-x_n)
		for rk in rec_knowledge:
			# self.logger.info(
			# 	'rank {} i {}, self logits {}, rec logits {}'.format(self.rank, i, logits[0], rec_knowledge[i][0]))
			if rk['rank'] != self.rank:
				# logits_from_other = rec_knowledge[i * batchsize:(i + 1) * batchsize, :].to(self.device_id)
				kd_loss += self.logits_loss(F.log_softmax(logits/T, dim=1), F.softmax(rk['logits'].to(self.device)/T, dim=1))
		kd_loss /= (len(rec_knowledge)-1)
		return kd_loss

	def weights_allocate(self):
		"""
		Allocate model and param weights from solver to random arxiv
		:return:
		"""
		if len(self.all_explore_id) == 0:
			warnings.warn('world size is not enough, running on 1 node!')
			return

		if self._is_solver():
			# TODO: examine the logits from other tasks, which one improves the performance most, ask that to transfer.
			if self.most_related_task == -1:
				prob = special.softmax(self.corr_list / self.temperature)
				self.selected_tasks = np.random.choice(len(prob), len(self.task_explorer_ranks), p=prob,
				                                       replace=False).astype(np.int32)
				related_tasks = np.array(list(map(lambda x: x + 1 if x >= self.task_id else x, self.selected_tasks)))
			else:
				related_tasks = self.most_related_task
			related_ranks = related_tasks * self.device_per_task
			# the first place is the target rank, while the others are source ranks
			related_ranks = np.insert(related_ranks, 0, self.rank).astype(np.int32)

			# print('rank {}, related rank {}'.format(rank, related_ranks))
			related_table = np.zeros([self.exploit_comm.Get_size(), len(related_ranks)], dtype=np.int32)
			self.exploit_comm.Allgather(related_ranks, related_table)
			# self.logger.info('task {} rtable: {}'.format(self.task_id, related_table))
			# related_table = exploit_comm.allgather(related_ranks)

			for target_ranks in related_table:
				if self.rank in target_ranks[1:]:
					target_exploit_rank = target_ranks[0]
					target_explore_rank = np.where(target_ranks[1:] == self.rank)[0][
						                      0] + 1 + target_exploit_rank  # suppose no duplicate
					self.logger.info('task {} target_exploit_rank {} target_explore_rank: {}'.format(self.task_id,target_exploit_rank, target_explore_rank))
					full_param = self.network.state_dict()
					# params = {k: v for k, v in full_param.items() if 'features.1.' in k}
					req = self.comm.isend({'model': full_param, 'optim': self.optimizer.state_dict(), 'scheduler': self.schedulers},
					                      dest=target_explore_rank, tag=target_exploit_rank)
					req.wait()
		else:
			receive_dict = self.comm.recv(source=MPI.ANY_SOURCE, tag=self.task_solver_rank)

			if not self.replace_classifier:
				self.network.load_state_dict(receive_dict['model'])
				self.optimizer.load_state_dict(receive_dict['optim'])
				for i, scheduler in enumerate(receive_dict['scheduler']):
					if scheduler is not None:
						self.schedulers[i].load_state_dict(scheduler.state_dict())
			else:
				def weights_replace(new_dict, self_dict, model_name):
					if 'resnet' in model_name:
						new_dict = {k: v for k, v in new_dict.items() if 'fc' not in k}
					else:
						# new_dict = {k: v for k, v in new_dict.items() if v.shape == self_dict[k].shape}
						new_dict = {k: v for k, v in new_dict.items() if 'classifier' not in k}
					self_dict.update(new_dict)
					return self_dict

				updated_dict = weights_replace(receive_dict['model'], self.network.state_dict(), self.model_name)
				self.network.load_state_dict(updated_dict)
				optim_type = self.opt['train']['optim']
				optim_param = self.opt['train']['optimizer_param']
				# # optim_param['lr'] = self.lr

				self.set_requires_grad(False)
				# unfreeze = list(filter(lambda p: p.requires_grad, self.network.parameters()))
				# self.logger.info('unfreeze dict {}'.format(unfreeze))
				self.optimizer = getattr(optim, optim_type)(filter(lambda p: p.requires_grad, self.network.parameters()), **optim_param)

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
		if self.best_weights is not None:
			self.save_network(self.best_weights, iter_step)
