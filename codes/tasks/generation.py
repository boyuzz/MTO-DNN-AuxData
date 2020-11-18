import logging
import model.gan as gan
import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torchvision.utils as vutils
from torch.optim import lr_scheduler
from sklearn.svm import LinearSVC
from sklearn import metrics
import warnings
from scipy import special
from mpi4py import MPI
from model.networks import init_weights

from collections import OrderedDict

try:
	import horovod.torch as hvd
	USE_HVD = True
except ImportError:
	USE_HVD = False

from .base_task import BaseTask


class GenerationTask(BaseTask):
	def __init__(self, opt, **kwargs):
		super(GenerationTask, self).__init__(opt, kwargs["comm"], kwargs["device"])
		# TODO 到底怎么交互？？？
		self.w_transfer = opt['w_transfer']
		self.device = kwargs['device']
		world_size = self.comm.Get_size()
		if world_size == 1:
			self.sto = True
		else:
			self.sto = False
		# self.tb_logger = kwargs['tb_logger']

		train_opt = opt['train']
		self.training_step = 0

		# -----model-----
		data_name_G, self.model_name_G = opt['network_G'].split('_')
		data_name_D, model_name_D = opt['network_D'].split('_')
		# if data_name_G.lower() == 'uci':
		self.netG = getattr(gan, self.model_name_G)(**opt['G_param']).cuda(self.device)
		init_weights(self.netG, 'normal')
		# if data_name_D.lower() == 'uci':
		self.netD = getattr(gan, model_name_D)(**opt['D_param']).cuda(self.device)
		self.netD_ax = getattr(gan, model_name_D)(**opt['D_param']).cuda(self.device)
		init_weights(self.netD, 'normal')
		init_weights(self.netD_ax, 'normal')

		# ---load pretrained model if exists---
		self.load()

		# -----loss------
		loss_type = train_opt['loss']
		if loss_type == 'bce':
			self.loss_func = nn.BCELoss().cuda(self.device)

		# -----optimizer-----
		optim_params = []
		for k, v in self.netG.named_parameters():  # can optimize for a part of the model
			if v.requires_grad:
				optim_params.append(v)
			else:
				self.logger.warning('Params [{:s}] will not optimize.'.format(k))
		optim_type = train_opt['optim']
		self.optimizer_G = getattr(optim, optim_type)(self.netG.parameters(), **opt['train']['optimizer_param_G'])
		self.optimizers.append(self.optimizer_G)
		self.g_steps = train_opt['g_steps']

		self.optimizer_D = getattr(optim, optim_type)(self.netD.parameters(), **opt['train']['optimizer_param_D'])
		self.optimizers.append(self.optimizer_D)
		self.d_steps = train_opt['d_steps']

		# if train_opt['lr_scheme'] == 'MultiStepLR':
		# 	for optimizer in self.optimizers:
		# 		self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
		# 		                                                train_opt['lr_steps'], train_opt['lr_gamma']))
		# else:
		# 	raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

		# -----define log_dict-----
		self.log_dict = OrderedDict()
		self.log_dict['transfer_count'] = 0

		# -----transfer parameters-----
		self.transfer_count = 0

	def full_training(self, log_freq, val_freq, save_freq, tb_logger=None):
		self.netG.train()
		self.netD.train()
		gi_sampler = self.get_generator_input_sampler()
		minibatch_size = self.opt['datasets']['train']['batch_size']
		label_fake = torch.ones(minibatch_size).cuda(self.device) * 0.1
		label_real = torch.ones(minibatch_size).cuda(self.device) * 0.9

		epoch = self.start_epoch
		while True:
			epoch += 1
			for step in range(0, len(self.train_loader)):
				if self.w_transfer and self.training_step % self.opt['trans_freq'] == 0:
					self.weights_allocate()

				self.training_step += 1
				if self.training_step > self.total_iters:
					self.logger.info('Reach maximal steps')
					return

				# ----update D-------
				# for d_index in range(self.d_steps):
				self.optimizer_D.zero_grad()

				# update learning rate
				# self.update_learning_rate()
				d_real_data = iter(self.train_loader).next().cuda(self.device)

				d_real_decision = self.netD(d_real_data).view(-1)
				D_real = d_real_decision.mean().item()

				d_real_error = self.loss_func(d_real_decision, label_real)  # ones = true
				self.log_dict['d_real_loss'] = d_real_error.item()
				d_real_error.backward()

				d_gen_input = gi_sampler(minibatch_size, self.opt['G_param']['input_size']).cuda(self.device)
				d_fake_data = self.netG(d_gen_input).detach()  # detach to avoid training G on these labels
				d_fake_decision = self.netD(d_fake_data).view(-1)
				D_fake = d_fake_decision.mean().item()
				d_fake_error = self.loss_func(d_fake_decision, label_fake)  # zeros = fake
				self.log_dict['d_fake_loss'] = d_fake_error.item()
				d_fake_error.backward()
				self.optimizer_D.step()

				# ----update G-------
				# for g_index in range(self.g_steps):
				self.optimizer_G.zero_grad()
				gen_input = gi_sampler(minibatch_size, self.opt['G_param']['input_size']).cuda(self.device)
				g_fake_data = self.netG(gen_input)
				dg_fake_decision = self.netD(g_fake_data).view(-1)
				G_fake = dg_fake_decision.mean().item()

				if self.sto:
					g_error = self.loss_func(dg_fake_decision, label_real)
				else:
					dg_ax_fake_decision = self.netD_ax(g_fake_data)
					g_error = self.loss_func(dg_fake_decision, label_real) + 0.1*self.loss_func(dg_ax_fake_decision, label_real)  # we want to fool, so pretend it's all genuine
				self.log_dict['g_loss'] = g_error.item()
				g_error.backward()
				self.optimizer_G.step()

				self.log_dict['D_real'] = D_real
				self.log_dict['D_fake'] = D_fake
				self.log_dict['G_fake'] = G_fake

				# log
				if self.training_step % log_freq == 0:
					message = '<Task {}, epoch:{:3d}, iter:{:3d}> '.format(
						self.task_id, epoch, self.training_step)

					logs = self.get_current_log()
					for k, v in logs.items():
						message += '{:s}: {:.4e} '.format(k, v)
						# tensorboard self.logger
						if tb_logger is not None:
							tb_logger.add_scalar(k, v, self.training_step)

					# if tb_logger is not None:
					# 	x = vutils.make_grid(g_fake_data[:4], normalize=True, scale_each=True)
					# 	tb_logger.add_image('Image_fake', x, self.training_step)  # Tensor d_real_data
					#
					# 	y = vutils.make_grid(d_real_data[:4], normalize=True, scale_each=True)
					# 	tb_logger.add_image('Image_real', y, self.training_step)  # Tensor d_real_data

					# if tb_logger is not None:
					# 	for name, param in self.network.named_parameters():
					# 		tb_logger.add_histogram(name, param.clone().cpu().data.numpy(), self.training_step)
					self.logger.info(message)

				# validation
				if self.training_step % val_freq == 0:
					self.solver_update(tb_logger)

				# save tasks and training states
				if self.training_step % save_freq == 0:
					self.logger.info('Saving tasks and training states for task {}.'.format(self.task_id))
					self.save(self.training_step)
					self.save_training_state(epoch, self.training_step)


	def validation(self, **kwargs):
		self.netG.eval()
		gi_sampler = self.get_generator_input_sampler()
		minibatch_size = self.opt['datasets']['train']['batch_size']
		fid_score = 0

		if 'DCG' in self.model_name_G:
			from utils.inception import InceptionV3
			from torch.nn.functional import adaptive_avg_pool2d
			from utils.fid_score import calculate_frechet_distance
			real_arr = []
			fake_arr = []

			def get_activation(model, batch):
				pred = model(batch)[0]
				if pred.shape[2] != 1 or pred.shape[3] != 1:
					pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
				return pred

			block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
			incep_model = InceptionV3([block_idx])
			# if self.device:
			incep_model = incep_model.cuda(self.device)

			with torch.no_grad():
				for step in range(0, len(self.train_loader)):
					d_real_data = iter(self.train_loader).next().cuda(self.device)
					d_gen_input = gi_sampler(minibatch_size, self.opt['G_param']['input_size']).cuda(self.device)
					d_fake_data = self.netG(d_gen_input)
					p_real = get_activation(incep_model, d_real_data)
					p_fake = get_activation(incep_model, d_fake_data)
					real_arr.append(p_real.cpu().data.numpy().reshape(minibatch_size, -1))
					fake_arr.append(p_fake.cpu().data.numpy().reshape(minibatch_size, -1))

				real_arr = np.concatenate(real_arr, axis=0)
				fake_arr = np.concatenate(fake_arr, axis=0)
				m1 = np.mean(real_arr, axis=0)
				s1 = np.cov(real_arr, rowvar=False)
				m2 = np.mean(fake_arr, axis=0)
				s2 = np.cov(fake_arr, rowvar=False)

				fid_score = calculate_frechet_distance(m1, s1, m2, s2)
		elif 'LinearG' in self.model_name_G:
			fake_X = []
			fake_y = []
			for step in range(0, len(self.train_loader)):
				d_gen_input = gi_sampler(minibatch_size, self.opt['G_param']['input_size']).cuda(self.device)
				d_fake_data = self.netG(d_gen_input)
				fake_data = d_fake_data.data.cpu().numpy()
				X = fake_data[:, :-1]
				# y = np.round(fake_data[:, -1]).astype(np.int)
				y = fake_data[:, -1]
				fake_X.extend(X)
				fake_y.extend(y)

			fake_y = self.train_set.reverse_norm(np.array(fake_y))
			clf = LinearSVC()
			clf.fit(fake_X, fake_y)

			full_preds = []
			full_labels = []
			for sample in self.train_loader:
				data = sample.data.cpu().numpy()
				X = data[:, :-1]
				y = data[:, -1]
				real_y = self.train_set.reverse_norm(np.array(y))
				pred = clf.predict(X)
				full_preds.extend(pred)
				full_labels.extend(real_y)

			fid_score = metrics.accuracy_score(np.array(full_labels), np.array(full_preds))
		if 'verbose' in kwargs and kwargs['verbose']:
			self.logger.info('# Task {}  # FID score: {:.2f}'.format(self.task_id, fid_score))
		return fid_score

	def weights_allocate(self):
		'''
		allocate model and param weights from solver to random arxiv
		:return:
		'''
		# comm = computation_utility['world_comm']
		# solver_comm = computation_utility['solver_comm']
		# rank = computation_utility['rank']
		# task_arxiv_ranks = computation_utility['task_arxiv_ranks']
		# is_solver = computation_utility['is_solver']
		# world_size = computation_utility['world_size']
		# running_tasks = computation_utility['running_tasks']

		if len(self.all_explore_id) == 0:
			warnings.warn('world size is not enough, running on 1 node!')
			return

		if self._is_solver():
			prob = special.softmax(self.corr_list / self.temperature)
			self.selected_tasks = np.random.choice(len(prob), len(self.task_explorer_ranks), p=prob,
			                                       replace=False).astype(np.int32)
			related_tasks = np.array(list(map(lambda x: x + 1 if x >= self.task_id else x, self.selected_tasks)))

			related_ranks = related_tasks * self.device_per_task
			# the first place is the target rank, while the others are source ranks
			related_ranks = np.insert(related_ranks, 0, self.rank).astype(np.int32)

			# print('rank {}, related rank {}'.format(rank, related_ranks))
			related_table = np.zeros([self.exploit_comm.Get_size(), len(related_ranks)], dtype=np.int32)
			self.exploit_comm.Allgather(related_ranks, related_table)
			self.logger.info('task {} rtable: {}'.format(self.task_id, related_table))
			# related_table = exploit_comm.allgather(related_ranks)

			for target_ranks in related_table:
				if self.rank in target_ranks[1:]:
					target_exploit_rank = target_ranks[0]
					target_explore_rank = np.where(target_ranks[1:] == self.rank)[0][
						                      0] + 1 + target_exploit_rank  # suppose no duplicate
					# self.logger.info('target explore rank {}, self rank {}'.format(target_explore_rank, rank))
					req = self.comm.isend(self.netD.state_dict(),
					                 dest=target_explore_rank, tag=1)
					req.wait()
		else:
			# self.logger.info('rank {} receiving'.format(rank))
			receive_dict = self.comm.recv(source=MPI.ANY_SOURCE, tag=1)
			self.netD_ax.load_state_dict(receive_dict)

	def solver_update(self, tb_logger):
		# validation
		# if self.val_loader is not None:
		fid_score = self.validation(verbose=False)

		if self.w_transfer and self.world_size > len(self.task_explorer_ranks) and len(self.task_explorer_ranks) > 0:
			if not self._is_solver():
				req = self.comm.isend(
					{'score': fid_score,
					 'model_d': self.netD.state_dict(),
					 'model_dax': self.netD_ax.state_dict(),
					 'model_g': self.netG.state_dict(),
					 'optim_d': self.optimizer_D.state_dict(),
					 'optim_g': self.optimizer_G.state_dict()
					 },
					dest=self.task_solver_rank, tag=0)
				req.wait()
			else:
				val_model_dicts = [self.comm.recv(source=i, tag=0) for i in self.task_explorer_ranks]
				# all_val = np.array([vmd['val'] for vmd in val_model_dicts])
				all_score = np.array([vmd['score'] for vmd in val_model_dicts])

				if all_score.min() < fid_score:
					self.netG.load_state_dict(val_model_dicts[all_score.argmin()]['model_g'])
					self.netD.load_state_dict(val_model_dicts[all_score.argmin()]['model_d'])
					self.netD_ax.load_state_dict(val_model_dicts[all_score.argmin()]['model_dax'])
					self.optimizer_G.load_state_dict(val_model_dicts[all_score.argmin()]['optim_g'])
					self.optimizer_D.load_state_dict(val_model_dicts[all_score.argmin()]['optim_d'])
					self.transfer_count += 1
					fid_score = all_score.min()

		step = 'last' if self.training_step == self.total_iters else self.training_step
		message = '<Task {}-rank {}, # Iter: {} score: {:.4e}, Transfer Count: {}, corr_list: {}>'.format(
			self.task_id, self.rank, step, fid_score, self.transfer_count, self.corr_list)
		self.logger.info(message)

		# tensorboard self.logger
		if self._is_solver():
			if self.opt['use_tb_logger']:
				tb_logger.add_scalar('Test_score', fid_score, self.training_step)
			for i, cl in enumerate(self.corr_list):
				if i >= self.task_id:
					relate_task_id = i + 1
				else:
					relate_task_id = i
				if self.opt['use_tb_logger']:
					tb_logger.add_scalar('task {} helpfulness'.format(relate_task_id), cl, self.training_step)

	def get_current_log(self):
		return self.log_dict

	def get_distribution_sampler(self, mu, sigma):
		return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian

	def get_generator_input_sampler(self):
		# 4-d for image and 2-d for vector
		# return lambda m, n: torch.rand(m, n, 1, 1)  # Uniform-dist data into generator, _NOT_ Gaussian
		return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian

	def load(self):
		load_path_G = self.opt['path']['pretrain_model_G']
		if load_path_G is not None:
			self.logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
			self.load_network(load_path_G, self.netG)
		load_path_D = self.opt['path']['pretrain_model_D']
		if self.opt['is_train'] and load_path_D is not None:
			self.logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
			self.load_network(load_path_D, self.netD)

	def save(self, iter_step):
		self.save_network(self.netG, iter_step, 'G')
		self.save_network(self.netD, iter_step, 'D')
