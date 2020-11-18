from collections import OrderedDict
from PIL import Image
import numpy as np
from scipy import special
from mpi4py import MPI
import cv2

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from model import fcn
import warnings

from utils.segmentation import get_loss_function
from utils.segmentation.metrics import runningScore
from model.networks import init_weights
import os
# from prefetch_generator import BackgroundGenerator

from .scalable_base_task import BaseTask


class SegmentationTask(BaseTask):
	def __init__(self, opt, **kwargs):
		super(SegmentationTask, self).__init__(opt, kwargs['logger'])
		train_opt = opt['train']
		self.ran_cl = opt['rancl']
		self.num_classes = opt['datasets']['train']['num_classes']
		self.dataset_name = opt['datasets']['train']['name']
		self.kd_transfer = opt['kd_transfer']
		self.att_transfer = opt['att_transfer']
		self.fsp_transfer = opt['fsp_transfer']
		self.w_transfer = opt['w_transfer']
		self.ws_transfer = opt['ws_transfer']
		self.tb_logger = kwargs['tb_logger']
		self.device_id = kwargs['device_id']
		# self.replace_classifier = opt['varyOnData']
		self.replace_classifier = False

		# -----prepare for transfer-------------
		if self.kd_transfer:
			# set seeds of all tasks the same to ensure the dataloader is in the same order
			torch.manual_seed(0)

		if self.fsp_transfer or self.att_transfer:
			self.activation = OrderedDict()

		# -----define network and load pretrained tasks-----
		self.model_name = opt['network']
		self.network = fcn.get_model(self.model_name, self.num_classes)
		self.network = self.network.cuda(self.device_id)

		# init_weights(self.network)
		self.load()

		# for name, param in self.network.named_parameters():
		# 	a = param.clone().cpu().data.numpy()
		# 	print(self.task_id, 'scale', name, a.max(), a.min())

		# -----define loss function------
		loss_type = train_opt['loss']
		loss_param = train_opt['loss_param']
		if 'weight' in loss_param.keys():
			loss_param['weight'] = torch.from_numpy(np.array(loss_param['weight']).astype(np.float32)).cuda(self.device_id)
		self.loss_func = get_loss_function(loss_type, loss_param)

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

			self.optimizer = getattr(optim, optim_type)(self.network.parameters(), **opt['train']['optimizer_param'])
			self.optimizers.append(self.optimizer)

			# -----define schedulers-----
			if train_opt['lr_scheme'] == 'MultiStepLR':
				for optimizer in self.optimizers:
					# self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
					# 	train_opt['lr_steps'], train_opt['lr_gamma']))
					# lambda1 = lambda step: train_opt['lr_gamma'] ** sum([step >= mst for mst in train_opt['lr_steps']])
					# # TODO I make the lr for classifer constant!!
					# lambda2 = lambda step: 1
					self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, **opt['train']['lr_scheme_param']))
			elif train_opt['lr_scheme'] == 'CycleLR':
				for optimizer in self.optimizers:
					self.schedulers.append(lr_scheduler.CyclicLR(optimizer, **opt['train']['lr_scheme_param']))
			elif train_opt['lr_scheme'] == 'ReduceLROnPlateau':
				for optimizer in self.optimizers:
					self.schedulers.append(lr_scheduler.ReduceLROnPlateau(optimizer, **opt['train']['lr_scheme_param']))
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

	def binarize_image(self, img):
		img = cv2.GaussianBlur(img, (5, 5), 0)
		ret, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		return binary_img

	def smooth_image(self, img):
		# denoise
		denoise_img = cv2.fastNlMeansDenoising(img, None, 20)
		# dilation & erosion
		kernel = np.ones((5, 5), np.uint8)
		erosion_1 = cv2.erode(denoise_img, kernel, iterations=3)
		dilation_1 = cv2.dilate(erosion_1, kernel, iterations=3)
		erosion_2 = cv2.erode(dilation_1, kernel, iterations=2)
		result = cv2.dilate(erosion_2, kernel, iterations=2)
		return result

	def set_requires_grad(self, val):
		for k, p in self.network.named_parameters():
			if 'fc' not in k:
				p.requires_grad = val

	def step(self, rank, log_freq, save_freq, is_solver=False):
		# update learning rate
		self.update_learning_rate(self.training_step)

		(data, target) = next(self.train_iter)
		# with torch.autograd.profiler.profile() as prof:
		self.training_step += 1

		data, target = data.cuda(self.device_id), target.cuda(self.device_id)
		self.optimizer.zero_grad()
		logits = self.network(data)
		loss = self.loss_func(logits, target)
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

				self.tb_logger.add_image('input', data[0], self.training_step)
				pred = F.sigmoid(logits[0])
				self.tb_logger.add_image('pred', pred, self.training_step)
				self.tb_logger.add_image('label', target[0], self.training_step)
				# self.logger.info('pred range: {}-{}, label range: {}-{}'.format(pred.min(), pred.max(), target[0].min(),
				#                                                                 target[0].max()))

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
			"step": self.training_step,
			"corrlist": self.corr_list,
			"temperature": self.temperature,
			"transfer_count": self.received_count,
		}
		sync_items = comm.bcast(sync_items, root)
		self.network.load_state_dict(sync_items['network_dict'])
		self.optimizer.load_state_dict(sync_items['optim_dict'])
		self.training_step = sync_items['step']
		self.corr_list = sync_items['corrlist']
		self.temperature = sync_items['temperature']
		self.received_count = sync_items['transfer_count']

	def validation(self, **kwargs):
		self.network.eval()
		running_metrics = runningScore(self.num_classes)

		with torch.no_grad():
			for data, target in self.val_loader:
				data = data.cuda(self.device_id)
				# if self.one_hot:
				# 	target = self._convert_int_onehot(target)
				if 'unet' in self.model_name:
					padded_border = list(self._pad_image(data))
					data = F.pad(data, padded_border, "reflect")
					target = F.pad(target, padded_border, 'reflect')
				logits = self.network(data)
				# if self.prob_est:
				# 	logits = F.softmax(logits, dim=1)
				# sum up batch loss
				# get the index of the max log-probability
				# pred = logits.data.max(1)[1].cpu().numpy()
				logits = F.sigmoid(logits)
				# logits[logits > 0.3] = 1
				# logits[logits <= 0.3] = 0
				pred = logits.data.cpu().numpy().squeeze()
				pred = (pred * 255).astype(np.uint8)
				pred = self.binarize_image(pred)
				pred = self.smooth_image(pred)
				pred = pred/255.
				running_metrics.update(target.data.cpu().numpy().astype(np.int), pred.astype(np.int))

			if self.opt['use_tb_logger']:
				self.tb_logger.add_image('val_input', data[0], self.training_step)
				self.tb_logger.add_image('val_pred', pred, self.training_step)
				self.tb_logger.add_image('val_label', target[0], self.training_step)

		score, cls_iou = running_metrics.get_scores()
		# test_accuracy /= len(self.testing_set)
		tp_iou = cls_iou[1]

		# test_roc_auc = metrics.roc_auc_score(np.array(full_labels), np.array(full_preds))
		# TODO: add mAP metrics
		if kwargs['verbose']:
			self.logger.info(
				'# Task {}-rank {}  # Validation # Scores: {}, TP IoU: {:.4f}'.format(self.task_id, kwargs['rank'],
				                                                                      score, tp_iou))
		self.network.train()
		return score, tp_iou #, test_roc_auc

	def inference(self, directory, labels=None):
		from utils.segmentation.decode import decode, encode
		import torchvision.transforms.functional as TF
		from skimage import io
		# from utils.segmentation import crf
		from utils.util import is_image_file
		image_list = []
		for (dirpath, dirnames, filenames) in os.walk(directory):
			image_list.extend(filenames)
			break

		if labels is not None:
			running_metrics = runningScore(self.num_classes)
			label_list = []
			for (dirpath, dirnames, filenames) in os.walk(labels):
				label_list.extend(filenames)
				break

		with torch.no_grad():
			for i, filename in enumerate(image_list):
				ext = filename.split('.')[1]
				if not is_image_file(filename):
					continue
				full_img = Image.open(os.path.join(directory, filename))
				data = TF.to_tensor(full_img)
				data = TF.normalize(data, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
				data = data.unsqueeze(0)
				data = data.cuda(self.device_id)
				# if self.one_hot:
				# 	target = self._convert_int_onehot(target)
				n, c, h, w = data.size()
				if h < 1500 and w < 1500:
					if 'unet' in self.model_name:
						# new_shape = (2048, 2048)
						padded_border = list(self._pad_image(data))
						data = F.pad(data, padded_border, "reflect")
						# data = F.interpolate(data, size=new_shape, mode="bicubic", align_corners=True)

					logits = self.network(data)
					if 'unet' in self.model_name:
						padded_border = [-1*p for p in padded_border]
						logits = F.pad(logits, padded_border, "constant", 0)
				else:
					hspace = np.linspace(0, h, 5).astype(np.int)
					wspace = np.linspace(0, w, 5).astype(np.int)
					sub_images = []
					for i in range(len(hspace)-1):
						for j in range(len(wspace)-1):
							sub_images.append(data[:,:,hspace[i]:hspace[i+1], wspace[j]:wspace[j+1]])
					# sub_images = [data[:,:,0:h//3, 0:w//3], data[:,:,0:h//2, w//3:2*w//3], data[:,:,0:h//2, 2*w//3:w],
					#               data[:, :, h//3:2*h // 3, 0:w // 3], data[:, :, h//3:2*h // 3, w // 3:2 * w // 3], data[:, :, h//3:2*h // 3, 2 * w // 3:w],
					#               data[:, :, 2*h //3:h, 0:w // 3], data[:, :, 2*h //3:h, w // 3:2 * w // 3], data[:, :, 2*h //3:h, 2 * w // 3:w]]
					sub_results = []
					for img in sub_images:
						if 'unet' in self.model_name:
							# new_shape = (2048, 2048)
							padded_border = list(self._pad_image(img))
							img = F.pad(img, padded_border, "reflect")
						# data = F.interpolate(data, size=new_shape, mode="bicubic", align_corners=True)
						img = self.network(img)
						if 'unet' in self.model_name:
							padded_border = [-1 * p for p in padded_border]
							img = F.pad(img, padded_border, "constant", 0)
						sub_results.append(img)
					tile1 = torch.cat(sub_results[0:4], 3)
					tile2 = torch.cat(sub_results[4:8], 3)
					tile3 = torch.cat(sub_results[8:12], 3)
					tile4 = torch.cat(sub_results[12:16], 3)
					logits = torch.cat([tile1, tile2, tile3, tile4], 2)
					del tile1, tile2, tile3, tile4, sub_results

				# pred = logits.data.max(1)[1].cpu().numpy()
				pred = F.sigmoid(logits)
				# pred[pred > 0.3] = 1
				# pred[pred <= 0.3] = 0
				# pred = pred.data[0].cpu().numpy()[0]
				pred = pred.data.cpu().numpy().squeeze()
				pred = (pred*255).astype(np.uint8)
				pred = self.binarize_image(pred)
				pred = self.smooth_image(pred)
				if labels is not None:
					l = Image.open(os.path.join(labels, label_list[i]))
					running_metrics.update(np.asarray(encode(np.asarray(l, dtype=np.uint8).squeeze(), self.dataset_name)), pred.astype(np.int))
				# p2 = nn.functional.softmax(logits, dim=1)
				# pred == p2
				# pred = decode(pred.data.cpu().numpy(), self.dataset_name).astype(np.uint8)
				# pred = crf.CRFs(np.array(full_img), pred)

				(filename, extension) = os.path.splitext(filename)
				io.imsave(os.path.join(self.opt['path']['pred'], filename+'.{}'.format(ext)), pred)
			# pred_img = Image.fromarray(pred)
			# pred_img.save(os.path.join(self.opt['path']['pred'], filename))
		if labels is not None:
			score, cls_iou = running_metrics.get_scores()
			print(score, cls_iou)

	@staticmethod
	def _pad_image(img):
		height, width = img.shape[2:]
		if height % 32 == 0:
			y_min_pad = 0
			y_max_pad = 0
		else:
			y_pad = 32 - height % 32
			y_min_pad = int(y_pad / 2)
			y_max_pad = y_pad - y_min_pad

		if width % 32 == 0:
			x_min_pad = 0
			x_max_pad = 0
		else:
			x_pad = 32 - width % 32
			x_min_pad = int(x_pad / 2)
			x_max_pad = x_pad - x_min_pad
		return x_min_pad, x_max_pad, y_min_pad, y_max_pad

	def solver_update(self, computation_utility):
		comm = computation_utility['world_comm']
		rank = computation_utility['rank']
		task_solver_rank = computation_utility['task_solver_rank']
		task_arxiv_ranks = computation_utility['task_arxiv_ranks']
		is_solver = computation_utility['is_solver']
		world_size = computation_utility['world_size']

		# validation
		if self.val_loader is not None:
			test_score, test_iou = self.validation(rank=rank, verbose=False)
			if self.w_transfer and world_size > len(task_arxiv_ranks) > 0:
				if not is_solver:
					req = comm.isend({'score': test_score, 'iou': test_iou, 'model': self.network.state_dict(),
					                  'optim': self.optimizer.state_dict()}, dest=task_solver_rank, tag=0)
					req.wait()
				else:
					val_model_dicts = [comm.recv(source=i, tag=0) for i in task_arxiv_ranks]
					all_iou = np.array([vmd['iou'] for vmd in val_model_dicts])

					if not self.ran_cl:
						corr_delta = np.tanh(all_iou - test_iou)
						for i, cd in enumerate(corr_delta):
							# self.corr_list[self.selected_tasks[i]] += cd
							self.corr_list[self.selected_tasks[i]] = 0.8 * self.corr_list[self.selected_tasks[i]] + cd

					if all_iou.max() > test_iou:
						self.network.load_state_dict(val_model_dicts[all_iou.argmax()]['model'])
						self.optimizer.load_state_dict(val_model_dicts[all_iou.argmax()]['optim'])
						self.received_count += 1
						test_score = val_model_dicts[all_iou.argmax()]['score']
						test_iou = all_iou.max()

			# message += 'corr_list: {}'.format(self.corr_list)
			step = 'last' if self.training_step == self.total_iters else self.training_step
			message = '<Task {}-rank {}, # Iter: {}, Scores: {}, TP IoU: {:.4f} Transfer Count: {}, corr_list: {}>'.format(
				self.task_id, rank, step, test_score, test_iou, self.received_count, self.corr_list)
			self.logger.info(message)

			# tensorboard self.logger
			if is_solver:
				if self.opt['use_tb_logger']:
					# self.tb_logger.add_scalar('Test_loss', test_loss, self.training_step)
					self.tb_logger.add_scalar('Overall_Accuracy', test_score['Overall_Acc'], self.training_step)
					self.tb_logger.add_scalar('Mean_Acc', test_score['Mean_Acc'], self.training_step)
					self.tb_logger.add_scalar('FreqW_Acc', test_score['FreqW_Acc'], self.training_step)
					self.tb_logger.add_scalar('Mean_IoU', test_score['Mean_IoU'], self.training_step)
					self.tb_logger.add_scalar('TP_IoU', test_iou, self.training_step)
				for i, cl in enumerate(self.corr_list):
					if i >= self.task_id:
						relate_task_id = i + 1
					else:
						relate_task_id = i
					if self.opt['use_tb_logger']:
						self.tb_logger.add_scalar('task {} helpfulness'.format(relate_task_id), cl, self.training_step)

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
			self.logger.info('rank {}, related ranks {} will help me.'.format(rank, related_ranks))

			related_table = np.zeros([solver_comm.Get_size(), len(related_ranks)], dtype=np.int32)
			solver_comm.Allgather(related_ranks, related_table)

			for target_ranks in related_table:
				if rank in target_ranks[1:]:
					target_solver_rank = target_ranks[0]
					target_arxiv_rank = np.where(target_ranks == rank)[0][0] + target_solver_rank  # suppose no duplicate
					# self.logger.info('target explore rank {}, self rank {}'.format(target_explore_rank, rank))
					req = comm.isend({'model': self.network.state_dict(), 'optim': self.optimizer.state_dict()}, dest=target_arxiv_rank, tag=1)
					req.wait()
		else:
			# self.logger.info('rank {} receiving'.format(rank))
			receive_dict = comm.recv(source=MPI.ANY_SOURCE, tag=1)

			if not self.replace_classifier:
				self.network.load_state_dict(receive_dict['model'])
				self.optimizer.load_state_dict(receive_dict['optim'])
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
