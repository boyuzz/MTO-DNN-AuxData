from collections import OrderedDict
from sklearn import metrics
import numpy as np
from scipy import special
from mpi4py import MPI
import os
import tqdm

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
		# self.replace_classifier = opt['varyOnData']
		self.replace_classifier = False
		self.metric = 5     # Used for selecting best model for main task
		self.historic_best = []     # record historical best validation

		# -----prepare for transfer-------------
		if self.kd_transfer:
			# set seeds of all tasks the same to ensure the dataloader is in the same order
			torch.manual_seed(0)

		if self.fsp_transfer or self.att_transfer:
			self.activation = OrderedDict()

		# -----define network and load pretrained tasks-----
		data_name, model_name = opt['network'].split('-')
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
				if 'efficient' in self.model_name:
					self.network = imagenet.EfficientNet.from_pretrained('efficientnet-b5', num_classes=self.num_classes)
				else:
					self.network = getattr(imagenet, model_name)(pretrained=True)
				if opt['train_lastlayer']:
					for param in self.network.parameters():
						param.requires_grad = False

				if 'resnet' in self.model_name or 'resnext' in model_name:
					self.network.fc = nn.Linear(self.network.fc.in_features, self.num_classes)
				elif 'vgg' in self.model_name:
					self.network.classifier[-1] = nn.Linear(4096, self.num_classes)
				elif 'squeeze' in self.model_name:
					self.network.num_classes = self.num_classes
					self.network.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=1)
				elif 'dense' in self.model_name:
					num_features = self.network.classifier.in_features
					self.network.classifier = nn.Linear(num_features, self.num_classes)
				elif 'mobile' in self.model_name:
					num_features = self.network.classifier[-1].in_features
					self.network.classifier[-1] = nn.Linear(num_features, self.num_classes)

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

			self.logits_loss = nn.KLDivLoss(reduction='batchmean').cuda(self.device_id)
			self.norm_loss = nn.MSELoss().cuda(self.device_id)
			self.at_weight = train_opt['at_weight']
			self.kd_weight = train_opt['kd_weight']
			self.ws_weight = train_opt['ws_weight']

			# -----define optimizers-----
			optim_type = train_opt['optim']
			self.init_lr = opt['train']['optimizer_param']['lr']

			# if opt['imagenet_pretrained']:
			# 	import warnings
			# 	warnings.warn("Since using pretrained model, the lr will be reset as 1e-6 for 1st layer, 1e-4 for middle layer,"
			# 	              "and lr for last layer.")
			param_groups = []
			# for name, param in self.network.named_parameters():
			if 'resnet' in model_name or 'resnext' in model_name:
				param_groups.append(
					{'params': [param for name, param in self.network.named_parameters() if 'fc' not in name]}) # , 'lr': 0.1*self.init_lr
				param_groups.append({'params': self.network.fc.parameters()})
			elif 'efficient' in model_name:
				param_groups.append(
					{'params': [param for name, param in self.network.named_parameters() if
					            'fc' not in name]})  # , 'lr': 0.1*self.init_lr
				param_groups.append({'params': self.network._fc.parameters()})
			elif 'vgg' in model_name:
				param_groups.append(
					{'params': [param for name, param in self.network.named_parameters() if 'classifier' not in name]})
				param_groups.append({'params': self.network.classifier.parameters()})
			elif 'squeeze' in model_name or 'dense' in model_name or 'mobile' in model_name:
				param_groups.append(
					{'params': [param for name, param in self.network.named_parameters() if 'classifier' not in name]})
				param_groups.append({'params': self.network.classifier.parameters()})

			self.optimizer = getattr(optim, optim_type)(param_groups, **opt['train']['optimizer_param'])
			self.optimizers.append(self.optimizer)

			# -----define schedulers-----
			if train_opt['lr_scheme'] == 'MultiStepLR':
				for optimizer in self.optimizers:
					# self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
					# 	train_opt['lr_steps'], train_opt['lr_gamma']))
					# lambda1 = lambda step: train_opt['lr_gamma'] ** sum([step >= mst for mst in train_opt['lr_steps']])
					# # TODO I make the lr for classifer constant!!
					# lambda2 = lambda step: 1
					self.scheduler = lr_scheduler.MultiStepLR(optimizer, **opt['train']['lr_scheme_param'])
					self.schedulers.append(self.scheduler)
			elif train_opt['lr_scheme'] == 'CycleLR':
				for optimizer in self.optimizers:
					self.scheduler = lr_scheduler.CyclicLR(optimizer, **opt['train']['lr_scheme_param'])
					self.schedulers.append(self.scheduler)
			elif train_opt['lr_scheme'] == 'ReduceLROnPlateau':
				for optimizer in self.optimizers:
					self.scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **opt['train']['lr_scheme_param'])
					self.schedulers.append(self.scheduler)
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

	def set_requires_grad(self, val):
		for k, p in self.network.named_parameters():
			if 'fc' not in k:
				p.requires_grad = val

	def step(self, rank, log_freq, save_freq, is_solver=False):
		(data, target) = next(self.train_iter)

		self.training_step += 1
		# update learning rate
		self.update_learning_rate(self.training_step)
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

		# for name, param in self.network.named_parameters():
		# 	if param.grad is not None:
		# 		a = param.grad.clone().cpu().data.numpy()
		# 		print(self.task_id, 'grad', name, a.max(), a.min())

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

	def synchronize(self, root, computation_utility):
		comm = computation_utility['world_comm']
		sync_items = {
			"network_dict": self.network.state_dict(),
			"optim_dict": self.optimizer.state_dict(),
			"scheduler_dict": self.scheduler.state_dict(),
			"step": self.training_step,
			"corrlist": self.corr_list,
			"best_score": self.metric,
			"temperature": self.temperature,
			"transfer_count": self.received_count,
		}
		sync_items = comm.bcast(sync_items, root)
		self.network.load_state_dict(sync_items['network_dict'])
		self.optimizer.load_state_dict(sync_items['optim_dict'])
		self.scheduler.load_state_dict(sync_items['scheduler_dict'])
		self.training_step = sync_items['step']
		self.corr_list = sync_items['corrlist']
		self.metric = sync_items['best_score']
		self.temperature = sync_items['temperature']
		self.received_count = sync_items['transfer_count']

	def update_best(self, task_list):
		t_best = self.metric
		best_index = 0
		for i, t in enumerate(task_list):
			if t.metric > t_best:
				best_index = i
				t_best = t.metric
		task = task_list[best_index]

		self.network.load_state_dict(task.network.state_dict())
		self.optimizer.load_state_dict(task.optimizer.state_dict())
		self.scheduler.load_state_dict(task.scheduler.state_dict())
		self.corr_list = task.corr_list
		self.temperature = task.temperature
		self.historic_best.append(t_best)

	def solver_update(self, computation_utility):
		comm = computation_utility['world_comm']
		rank = computation_utility['rank']
		task_solver_rank = computation_utility['task_solver_rank']
		task_arxiv_ranks = computation_utility['task_arxiv_ranks']
		is_solver = computation_utility['is_solver']
		world_size = computation_utility['world_size']

		# validation
		if self.val_loader is not None:
			test_loss, test_accuracy, test_f1 = self.validation(rank=rank, verbose=False, split=self.opt['val_split'])

			if self.w_transfer and world_size > len(task_arxiv_ranks) and len(task_arxiv_ranks)>0:
				if not is_solver:
					req = comm.isend({'val': test_accuracy, 'f1': test_f1, 'loss': test_loss, 'model': self.network.state_dict(),
									  'optim': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict()}, dest=task_solver_rank, tag=0)
					req.wait()
				else:
					val_model_dicts = [comm.recv(source=i, tag=0) for i in task_arxiv_ranks]
					# all_val = np.array([vmd['val'] for vmd in val_model_dicts])
					all_loss = np.array([vmd['loss'] for vmd in val_model_dicts])

					# if not self.ran_cl:
					# 	corr_delta = np.tanh(all_val - test_accuracy)
					# 	for i, cd in enumerate(corr_delta):
					# 		# self.corr_list[self.selected_tasks[i]] += cd
					# 		self.corr_list[self.selected_tasks[i]] = 0.8 * self.corr_list[self.selected_tasks[i]] + cd

					# if all_val.max() > test_accuracy:
					# 	self.network.load_state_dict(val_model_dicts[all_val.argmax()]['model'])
					# 	self.optimizer.load_state_dict(val_model_dicts[all_val.argmax()]['optim'])
					# 	self.scheduler.load_state_dict(val_model_dicts[all_val.argmax()]['scheduler'])
					# 	self.received_count += 1
					# 	test_accuracy = all_val.max()
					# 	test_f1 = val_model_dicts[all_val.argmax()]['f1']
					if all_loss.min() < test_loss:
						self.network.load_state_dict(val_model_dicts[all_loss.argmin()]['model'])
						self.optimizer.load_state_dict(val_model_dicts[all_loss.argmin()]['optim'])
						self.scheduler.load_state_dict(val_model_dicts[all_loss.argmin()]['scheduler'])
						self.received_count += 1
						test_loss = all_loss.min()
						test_accuracy = val_model_dicts[all_loss.argmin()]['val']
						test_f1 = val_model_dicts[all_loss.argmin()]['f1']

			# message += 'corr_list: {}'.format(self.corr_list)
			# if test_accuracy > self.metric:
			# 	self.save(str(test_accuracy))
			if test_loss < self.metric:
				self.save(str(test_loss))

			self.metric = test_loss

			step = 'last' if self.training_step == self.total_iters else self.training_step
			message = '<Task {}-rank {}, # Iter: {} Loss: {:.4e}, Accuracy: {:.2f}%, F1 score: {:.4f}, Transfer Count: {}, corr_list: {}>'.format(
				self.task_id, rank, step, test_loss, test_accuracy, test_f1, self.received_count,
				self.corr_list)
			self.logger.info(message)

			# tensorboard self.logger
			if is_solver:
				if self.opt['use_tb_logger']:
					self.tb_logger.add_scalar('Test_loss', test_loss, self.training_step)
					self.tb_logger.add_scalar('Accuracy', test_accuracy, self.training_step)
					self.tb_logger.add_scalar('F1', test_f1, self.training_step)
				for i, cl in enumerate(self.corr_list):
					if i >= self.task_id:
						relate_task_id = i + 1
					else:
						relate_task_id = i
					if self.opt['use_tb_logger']:
						self.tb_logger.add_scalar('task {} helpfulness'.format(relate_task_id), cl, self.training_step)

	# current_lr = self.get_current_learning_rate()[0]
	# self.schedulers[0].base_lrs[-1] = 0.01
	#
	# val_freq = self.opt['val_freq']
	# exponential_param = pow(current_lr / 0.01, 1 / val_freq)
	# # step+1: the step in scheduler starts from -1
	# lambda2 = lambda step: exponential_param ** ((step + 1) % val_freq if (step + 1) % val_freq != 0 else val_freq)
	# self.schedulers[0].lr_lambdas[-1] = lambda2

	def validation(self, **kwargs):
		self.network.eval()
		test_loss = 0.
		# test_accuracy = 0.
		full_preds = []
		full_labels = []

		if 'split' in kwargs.keys() and kwargs['split']=='test' and self.test_loader is not None:
			dataloader = self.test_loader
		elif 'split' in kwargs.keys() and kwargs['split'] == 'train' and self.train_loader is not None:
			dataloader = self.train_loader
		# elif 'split' in kwargs.keys() and kwargs['split'] == 'val_mto' and self.train_loader is not None:
		# 	dataloader = self.val_mto_loader
		else:
			dataloader = self.val_loader

		with torch.no_grad():
			for data, target in dataloader:
				data, target = data.cuda(self.device_id), target.cuda(self.device_id)
				# bs, ncrops, c, h, w = data.size()
				labels = target.data
				if self.one_hot:
					target = self._convert_int_onehot(target)

				logits = self.network(data)
				# result = self.network(data.view(-1, c, h, w))
				# result_avg = result.view(bs, ncrops, -1).mean(1)
				if self.prob_est:
					logits = F.softmax(logits, dim=1)
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
		test_accuracy = 100*metrics.accuracy_score(np.array(full_labels), np.array(full_preds))
		test_f1 = metrics.f1_score(np.array(full_labels), np.array(full_preds), average='macro')
		# test_roc_auc = metrics.roc_auc_score(np.array(full_labels), np.array(full_preds))
		# TODO: add mAP metrics
		if 'verbose' in kwargs.keys() and kwargs['verbose']:
			self.logger.info('# Task {} # Validation loss: {:.2f}, Accuracy: {:.2f}%, F1 score: {:.4f}'.format(self.task_id, test_loss, test_accuracy, test_f1))
		if 'report' in kwargs.keys() and kwargs['report']:
			from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
			cm = confusion_matrix(full_labels, full_preds)
			report = classification_report(np.array(full_labels), np.array(full_preds), digits=4)

			self.logger.info("overall accuracy: {}, F1: {}.".format(test_accuracy, test_f1))
			self.logger.info("confusion matrix\n {}.".format(cm))
			self.logger.info("classification report\n {}.".format(report))
		self.network.train()
		return test_loss, test_accuracy, test_f1

	def inference(self, directory):
		'''
		Perform testing. Calculate overall accuracy, per-class accuracy, confusion matrix,
		:param directory: path for testing images
		:return: None
		'''
		from utils.util import is_image_file
		import torchvision.transforms.functional as TF
		from PIL import Image
		self.network.eval()
		# test_loss = 0.
		# test_accuracy = 0.
		full_preds = []

		image_list = []
		for (dirpath, dirnames, filenames) in os.walk(directory):
			image_list.extend(filenames)
			break

		with torch.no_grad():
			for i, filename in enumerate(tqdm.tqdm(image_list)):
				if not is_image_file(filename):
					continue

				full_img = Image.open(os.path.join(directory, filename))
				data = TF.resize(full_img, 256)
				data = TF.center_crop(data, 224)
				# data = data.unsqueeze(0)
			# # pred_five = []
			# # for data in data_five:
				data = TF.to_tensor(data)
				data = TF.normalize(data, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
				data = data.unsqueeze(0)
			# for data, target in self.test_loader:
			# 	data, target = data.cuda(self.device_id), target.cuda(self.device_id)

				data = data.cuda(self.device_id)

				logits = self.network(data)
				if self.prob_est:
					logits = F.softmax(logits, dim=1)
				# get the index of the max log-probability
				pred = logits.data.max(1, keepdim=True)[1][0][0].cpu().numpy()+1
				# pred_five.append(pred.view(-1).cpu())
				# final_pred = np.argmax(np.bincount(np.array(pred_five)))
				full_preds.append((filename, pred))

				# if (i+1) % 5 == 0:
				# 	break

		with open('classification.txt', 'w', newline='\n') as f:
			for item in full_preds:
				f.writelines('{} {}\n'.format(item[0], item[1]))

		# test_accuracy += pred.eq(labels.view_as(pred)).cpu().float().sum()

	def knowledge_distillation_transfer(self, logits):
		'''
		To use this function, make sure the input dataloader to each task is exactly the same,
		otherwise this function is meaningless.
		:return:
		'''

		logits_data = logits.data.unsqueeze(0)
		rec_knowledge = self.exploit_comm.allgather(logits_data)

		kd_loss = 0
		# batchsize = int(rec_knowledge.size(0) // hvd.size())
		# self.logger.info("knowledge size {}, batchsize {}".format(rec_knowledge.size(), batchsize))
		# Note, the KDDivloss in pytorch says:
		# As with NLLLoss, the input given is expected to contain log-probabilities. However, unlike NLLLoss,
		# input is not restricted to a 2D Tensor. The targets are given as probabilities
		# (i.e. without taking the logarithm).
		# KLDivLoss: l_n = y_n*(log(y_n)-x_n)
		for i in range(0, len(rec_knowledge)):
			if i != self.task_id:
				# logits_from_other = rec_knowledge[i * batchsize:(i + 1) * batchsize, :].cuda(self.device_id)
				kd_loss += self.logits_loss(F.log_softmax(logits, dim=1), F.softmax(rec_knowledge[i].cuda(self.device_id), dim=1))
		kd_loss /= (len(rec_knowledge)-1)

		self.received_count += 1
		return kd_loss

	def attention_transfer(self):
		def at(x):
			return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

		att_group = []
		for key in self.activation:
			at_out = at(self.activation[key])
			att_group.append(at_out)

		rec_atts = self.exploit_comm.allgather(att_group)

		att_loss = 0
		for ra in rec_atts:
			# att_loss += self.norm_loss(att_group[idx], rec_att.mean(0).cuda(self.device_id))
			for i in range(0, len(att_group)):
				if i != self.task_id:
					att_loss += self.norm_loss(att_group[i], ra[i].cuda(self.device_id))

		att_loss /= (len(att_group) - 1)
		self.received_count += 1
		return att_loss

	def fsp_matrix_transfer(self):
		'''
		obtain the feature maps of bottlenecks (h*w*m), reshape it to (hw*m), then do matrix multiplication (m*n)
		allgather the mm, use L2 loss on it
		:return:
		'''
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

		rec_fsps = self.exploit_comm.allgather(matrix_group)

		fsp_loss = 0
		for rf in rec_fsps:
			for i in range(0, len(matrix_group)):
				if i != self.task_id:
					fsp_loss += self.norm_loss(matrix_group[i], rf[i].cuda(self.device_id))
		fsp_loss /= (len(matrix_group)-1)
		self.received_count += 1
		return fsp_loss

	def weights_statistic_transfer(self):
		mean_per_layers = []
		std_per_layers = []
		for name, param in self.network.named_parameters():
			if 'weight' in name:
				mean_per_layers.append(param.mean())
				std_per_layers.append(param.std())

		tensor_mean = torch.tensor(mean_per_layers).view([1,1,-1]).cuda(self.device_id)
		tensor_std = torch.tensor(std_per_layers).view([1,1,-1]).cuda(self.device_id)
		statistic = torch.cat([tensor_mean, tensor_std], dim=1)

		rec_statistic = self.exploit_comm.allgather(statistic).cuda(self.device_id)
		ws_loss = 0
		for i in range(0, len(rec_statistic)):
			if i != self.task_id:
				ws_loss += self.norm_loss(tensor_mean, rec_statistic[i][0]) \
						   +self.norm_loss(tensor_std, rec_statistic[i][1])
		ws_loss /= (len(rec_statistic)-1)

		self.received_count += 1
		return ws_loss

	def weights_allocate(self, computation_utility):
		'''
		allocate model and param weights from solver to random arxiv
		:return:
		'''
		comm = computation_utility['world_comm']
		solver_comm = computation_utility['solver_comm']
		rank = computation_utility['rank']
		task_arxiv_ranks = computation_utility['task_arxiv_ranks']
		is_solver = computation_utility['is_solver']
		world_size = computation_utility['world_size']
		running_tasks = computation_utility['running_tasks']

		if world_size < len(task_arxiv_ranks) or len(task_arxiv_ranks)<1:
			warnings.warn('world size is not enough, running on 1 node!')
			return

		if is_solver:
			self_index = np.where(running_tasks == self.task_id)
			other_running_tasks = np.delete(running_tasks, self_index)
			tasks_index = np.array(list(map(lambda x: x - 1 if x >= self.task_id else x, other_running_tasks)))
			temp_corrlist = np.array([self.corr_list[i] for i in tasks_index])

			prob = special.softmax(temp_corrlist / self.temperature)
			selected_index = np.random.choice(len(prob), self.device_per_task-1, p=prob, replace=False).astype(np.int32)
			self.selected_tasks = tasks_index[selected_index]

			selected_tasks = other_running_tasks[selected_index]
			related_index = np.array([np.where(running_tasks == i)[0][0] for i in selected_tasks])
			related_ranks = related_index * self.device_per_task
			# the first place is the target rank, while the others are source ranks
			related_ranks = np.insert(related_ranks, 0, rank).astype(np.int32)
			# self.logger.info('rank {}, related ranks {} will help me.'.format(rank, related_ranks))

			related_table = np.zeros([solver_comm.Get_size(), len(related_ranks)], dtype=np.int32)
			solver_comm.Allgather(related_ranks, related_table)

			for target_ranks in related_table:
				if rank in target_ranks[1:]:
					target_solver_rank = target_ranks[0]
					target_arxiv_rank = np.where(target_ranks == rank)[0][0] + target_solver_rank  # suppose no duplicate
					# self.logger.info('target explore rank {}, self rank {}'.format(target_explore_rank, rank))
					req = comm.isend({'model': self.network.state_dict(), 'optim': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict()},
					                 dest=target_arxiv_rank, tag=1)
					req.wait()
		else:
			# self.logger.info('rank {} receiving'.format(rank))
			receive_dict = comm.recv(source=MPI.ANY_SOURCE, tag=1)

			if not self.replace_classifier:
				self.network.load_state_dict(receive_dict['model'])
				self.optimizer.load_state_dict(receive_dict['optim'])
				self.scheduler.load_state_dict(receive_dict['scheduler'])
			else:
				#TODO alter the learning rate of classifier layer, rem to alter back when transferring to solver
				def weights_replace(new_dict, self_dict, model_name):
					# TODO Add support to other networks
					if 'resnet' in model_name:
						new_dict = {k: v for k, v in new_dict.items() if 'fc' not in k}
					else:
						new_dict = {k: v for k, v in new_dict.items() if v.shape == self_dict[k].shape}
					self_dict.update(new_dict)
					return self_dict

				updated_dict = weights_replace(receive_dict['model'], self.network.state_dict(), self.model_name)
				self.network.load_state_dict(updated_dict)

				#### reset the lr for the classifier to init lr #####
				source_optim = receive_dict['optim']
				current_lr = source_optim['param_groups'][-1]['lr']

				val_freq = self.opt['val_freq']
				exponential_param = pow(current_lr / self.init_lr, 1 / val_freq)
				# step+1: the step in scheduler starts from -1
				lambda2 = lambda step: exponential_param ** (step % val_freq if step % val_freq != 0 else val_freq)

				for i in range(len(self.schedulers[0].base_lrs)):
					self.schedulers[0].base_lrs[i] = self.init_lr
					self.schedulers[0].lr_lambdas[i] = lambda2

			# for param_group in self.optimizer.param_groups:
			# 	param_group['lr'] = lr
			# new_dict = receive_dict['optim']
			# self_dict = self.optimizer.state_dict()
			# self_dict['param_groups'][0].update(new_dict['param_groups'][0])
			# self.optimizer.load_state_dict(self_dict)

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
