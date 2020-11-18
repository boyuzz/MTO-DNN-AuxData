#!/usr/bin/env python
# -*- coding:latin-1 -*-

from mpi4py import MPI
import argparse
import os
import json
import logging
from time import time
import torch
import warnings
import numpy as np
# import random
# import cProfile

import options.options as option
from utils import util
from tasks import create_task


def main():
	# options
	parser = argparse.ArgumentParser()
	parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
	parser.add_argument('-name', type=str, help='Name of the experiment.')
	parser.add_argument('-ntasks', type=int, required=True, help='Number of tasks. Equals to #main+#auxiliary')
	parser.add_argument('-seed', type=int, help='Random seed.')
	parser.add_argument('-gsize', type=int, default=1, help='Num of devices per task.')
	parser.add_argument('-train', action='store_true', help='Specify if training.')
	parser.add_argument('-lmdb', action='store_true', help='create or use lmdb datasets for accelerating I/O.')
	parser.add_argument('-droot', type=str, help='dataroot if need specify.')
	parser.add_argument('-rancl', action='store_true', help='random task for transfer?')
	parser.add_argument('-tfreq', type=int, help='transferring frequence')
	parser.add_argument('-model', type=str, help='specify model to use')
	parser.add_argument('-k', action='store_true', help='Use knowledge distillation based transfer.')
	parser.add_argument('-a', action='store_true', help='Use attention based transfer.')
	parser.add_argument('-f', action='store_true', help='Use fsp matrix based transfer.')
	parser.add_argument('-w', action='store_true', help='Use weights transfer.')
	parser.add_argument('-ws', action='store_true', help='Use weights statistical transfer.')
	parser.add_argument('-VL', action='store_true', help='Variate on loss function.')
	parser.add_argument('-VO', action='store_true', help='Variate on optimization algorithm.')
	parser.add_argument('-VH', action='store_true', help='Variate on hyperparameters.')
	parser.add_argument('-VD', action='store_true', help='Variate on datasets.')
	parser.add_argument('-VS', type=float, help='Variate on resampling dataset.')
	args = parser.parse_args()

	if not args.train and args.gsize > 1:
		warnings.warn('1 device for validation is enough! The gsize will be reset to 1')
		args.gsize = 1

	if not args.w and args.gsize > 1:
		warnings.warn('The exploit-explorer mode is not activated! The gsize will be reset to 1')
		args.gsize = 1

	ntasks = args.ntasks
	comm = MPI.COMM_WORLD
	world_size = comm.Get_size()
	rank = comm.Get_rank()
	devices_per_node = torch.cuda.device_count()
	device_id = rank % devices_per_node
	torch.cuda.set_device(device_id)    # set cuda device
	devices_per_task = args.gsize
	cu_id = rank // devices_per_task
	all_solver_id = [i for i in list(range(world_size)) if i % devices_per_task == 0]
	ndevices = world_size // devices_per_task

	task_solver_rank = rank - rank % devices_per_task
	task_arxiv_ranks = [task_solver_rank + i for i in range(1, devices_per_task)]
	group = comm.Get_group()
	solver_group = MPI.Group.Incl(group, all_solver_id)
	solver_comm = comm.Create(solver_group)

	computation_utility = {
		"rank": rank,
		"task_solver_rank": task_solver_rank,
		"task_arxiv_ranks": task_arxiv_ranks,
		"is_solver": rank == task_solver_rank,
		"world_comm": comm,
		"solver_comm": solver_comm,
		"world_size": world_size
	}

	# If you don't use the following setting, the results will still be non-deterministic.
	# However, if you set them so, the speed may be slower, depending on models.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	tasks = []
	for task_id in range(ntasks):
		opt = option.parse(args, task_id)
		opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

		if rank == 0 and args.train:
			# Setup directory
			if not opt['resume'] or not opt['path']['resume_state']:
				if opt['is_train'] and task_id == 0:
					util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old folder if exists

				util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
				             and 'pretrain_model' not in key and 'resume' not in key and 'data_config' not in key))
				# save json file to task folder
				if opt['is_train']:
					with open(os.path.join(opt['path']['task'], 'task.json'), 'w') as outfile:
						json.dump(opt, outfile, indent=4, ensure_ascii=False)
		comm.barrier()

		# config loggers. Before it, the log will not work
		# util.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO, task_id=task_id, rank=rank)
		util.setup_logger(str(task_id), opt['path']['log'], 'train', level=logging.INFO, screen=True, task_id=task_id)
		logger = logging.getLogger(str(task_id))
		if rank == 0:
			if task_id == 0:
				logger.info(option.dict2str(opt))   # display options
			else:
				logger.info('Auxiliary task {} configuration: network: {}, optim: {}, loss: {}, data: {}'.format(
					task_id, opt['network'], opt['train']['optim'], opt['train']['loss'], opt['datasets']['train']['name']))

		# tensorboard logger
		tb_logger = None
		if opt['use_tb_logger'] and opt['is_train']:
			from tensorboardX import SummaryWriter
			if rank == 0:
				util.mkdir_and_rename(os.path.join('../tb_logger/', opt['name']))
			comm.barrier()

			tb_logger_path = os.path.join('../tb_logger/', opt['name'], str(task_id))
			tb_logger = SummaryWriter(log_dir=tb_logger_path)

		# TODO: without device_id, assign device_id in run-time
		# create tasks
		task = create_task(opt, logger=logger, tb_logger=tb_logger, device_id=device_id)
		if args.train:
			task.resume_training()
		tasks.append(task)

	logger.info(computation_utility)

	# Start training or testing
	start = time()
	if args.train:
		# Main task always fixed on CU_0
		# running_tasks_id = np.arange(1, ntasks)
		running_tasks_id = np.arange(ntasks)
		if rank == 0:
			np.random.shuffle(running_tasks_id)
			# running_tasks_id = np.insert(running_tasks_id, 0, 0)
			running_tasks_id = running_tasks_id[:ndevices]
		running_tasks_id = comm.bcast(running_tasks_id, root=0)
		computation_utility['running_tasks'] = running_tasks_id
		tasks[running_tasks_id[cu_id]].weights_allocate(computation_utility)

		# for i in range(1, int(opt['niter']*multiplier)+1):
		nstep = 0
		while True:
			# for step in range(0, max_step):
			# task.step()
			nstep += 1
			# tasks[running_tasks_id[cu_id]].solver_update(computation_utility)
			tasks[running_tasks_id[cu_id]].step(rank, opt['logger']['print_freq'],
			                                    opt['logger']['save_checkpoint_freq'],
			                                    computation_utility['is_solver'])

			# if nstep % opt['val_freq'] == 0:
			# 	tasks[running_tasks_id[cu_id]].validation(rank=rank, verbose=True)

			if nstep % opt['val_freq'] == 0:
				tasks[running_tasks_id[cu_id]].solver_update(computation_utility)
				if ntasks > 1:
					for idx, j in enumerate(running_tasks_id):
						tasks[j].synchronize(idx*devices_per_task, computation_utility)
					# tasks[0].update_best(tasks)

				running_tasks_id = np.arange(ntasks)
				if rank == 0:
					np.random.shuffle(running_tasks_id)
					# running_tasks_id = np.insert(running_tasks_id, 0, 0)
					running_tasks_id = running_tasks_id[:len(all_solver_id)]
				running_tasks_id = comm.bcast(running_tasks_id, root=0)
				computation_utility['running_tasks'] = running_tasks_id

				tasks[running_tasks_id[cu_id]].weights_allocate(computation_utility)

			if tasks[0].training_step >= opt['niter']:
				# logger.info('main task historic best: {}'.format(tasks[0].historic_best))
				break

		if rank == 0:
			logger.info('Saving the final task.')
			for task in tasks:
				task.save('latest')
			logger.info('End of training.')
			# tasks[0].inference()
			tasks[0].validation(verbose=True, report=True, split='test')
	else:
		if rank == 0:
			logger.info('Task {} start testing'.format(task_id))
			# tasks[0].validation(verbose=True, report=True)
			# tasks[0].inference(directory='../data/INRIA/test')
			# tasks[0].inference('../data/RSSRAI/test')

	duration = time() - start
	logger.info('The program tasks time {}'.format(duration))


if __name__ == '__main__':
	# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
	# cProfile.run('main()')
	main()
