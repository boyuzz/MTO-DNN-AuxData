from mpi4py import MPI
import argparse
import os
import json
import logging
from time import sleep, time
import torch
import warnings
import numpy as np

import options.options as option
from utils import util
from tasks import create_task


def main():
	# options
	parser = argparse.ArgumentParser()
	parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
	parser.add_argument('-name', type=str, help='Name of the experiment.')
	parser.add_argument('-seed', type=int, default=2, help='Random seed.')
	parser.add_argument('-gsize', type=int, default=1, help='Num of devices per task.')
	parser.add_argument('-val_freq', type=int, help='Num of transfer frequency.')
	parser.add_argument('-train', action='store_true', help='Specify if training.')
	parser.add_argument('-lmdb', action='store_true', help='create or use lmdb datasets for accelerating I/O.')
	parser.add_argument('-droot', type=str, help='dataroot if need specify.')
	parser.add_argument('-rancl', action='store_true', help='random task for transfer?')
	parser.add_argument('-create_val', action='store_true', help='create validation set from training set')
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
	parser.add_argument('-VF', action='store_true', help='Variate on cross validation.')
	parser.add_argument('-VS', type=float, help='fraction of bagging dataset.')
	opt = parser.parse_args()

	if not opt.train and opt.gsize > 1:
		warnings.warn('1 device for validation is enough! The gsize will be reset to 1')
		opt.gsize = 1

	if not opt.w and opt.gsize > 1:
		warnings.warn('The exploit-explorer mode is not activated! The gsize will be reset to 1')
		opt.gsize = 1

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	gpus_per_node = torch.cuda.device_count()
	if gpus_per_node == 0:
		device_id = 'cpu'
	else:
		device_id = rank % gpus_per_node
	print('start running! {} gpus are in use'.format(gpus_per_node))
	print('Torch version {}'.format(torch.__version__))

	# run to here is fine!!
	# torch.cuda.set_device(device_id)
	device_per_task = opt.gsize
	task_id = rank // device_per_task
	task_exploit_rank = rank - rank % device_per_task
	world_size = comm.Get_size()
	opt.ntasks = world_size

	opt = option.parse(opt, task_id)
	opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

	# Setup directory
	if not opt['resume'] and task_exploit_rank == rank and not opt['path']['resume_state']:
		try:
			if opt['is_train'] and rank == 0:
				util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old folder if exists
		except FileNotFoundError:
			raise FileNotFoundError("Issue from task {} and rank {}".format(task_id, rank))
	comm.Barrier()

	if not opt['resume'] and task_exploit_rank == rank and not opt['path']['resume_state']:
		util.mkdirs((path for key, path in opt['path'].items() if key not in
					 ['experiments_root', 'log', 'root', 'pretrain_model', 'pretrain_model_G', 'resume_state', 'data_config']))
		# save json file to task folder
		if opt['is_train']:
			with open(os.path.join(opt['path']['task'], 'task.json'), 'w') as outfile:
				json.dump(opt, outfile, indent=4, ensure_ascii=False)

	comm.Barrier()

	# config loggers. Before it, the log will not work
	util.setup_logger(str(task_id), opt['path']['log'], 'train', level=logging.INFO, screen=True, task_id=task_id, rank=rank)
	logger = logging.getLogger(str(task_id))

	if task_exploit_rank == rank:
		if task_id == 0:
			logger.info(option.dict2str(opt))   # display options
		else:
			logger.info('Auxiliary task {} configuration: network: {}, optim: {}, loss: {}, data: {}'.format(
				task_id, opt['network'], opt['train']['optim'], opt['train']['loss'], opt['datasets']['train']['name']))

	# tensorboard logger
	tb_logger = None
	if opt['use_tb_logger'] and opt['is_train'] and task_exploit_rank == rank:  # and 'debug' not in opt['name']
		from tensorboardX import SummaryWriter
		if rank == 0:
			util.mkdir_and_rename(os.path.join('../tb_logger/', opt['name']))
		comm.barrier()

		tb_logger_path = os.path.join('../tb_logger/', opt['name'], str(task_id))
		tb_logger = SummaryWriter(log_dir=tb_logger_path)

	# create task
	# If you don't use the following setting, the results will still be non-deterministic.
	# However, if you set them so, the speed may be slower, depending on models.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	task = create_task(opt, comm=comm, device=device_id)

	# Start training or testing
	start = time()
	if opt['is_train']:
		task.resume_training()
		task.full_training(opt['logger']['print_freq'], opt['val_freq'], opt['logger']['save_checkpoint_freq'], tb_logger)

		if rank == 0:
			if opt['varyOnCV']:
				all_hmean = task.cv_validation()
				results = [r['acc'] for r in all_hmean]
				idx = np.argmax(results)
				best_task_id = all_hmean[idx]['id']
				best_task_rank = device_per_task * best_task_id
				logger.info('All hmean is {} and select rank {} to run validation'.format(all_hmean, best_task_rank))
				if task_exploit_rank == best_task_rank:
					task.validation(verbose=True, report=True, split='test', best=True)
			else:
				task.validation(verbose=True, report=True, split='test', best=True)

			logger.info('End of training.')
	else:
		logger.info('Task {} start validation'.format(task_id))
		task.validation(verbose=True, report=True, split='test', best=True)

	duration = time() - start
	logger.info('The program tasks time {}'.format(duration))


if __name__ == '__main__':
	main()
