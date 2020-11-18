import os
import os.path as osp
import logging
from collections import OrderedDict
import json
import numpy as np
from utils import util
import platform


def variate_loss(opt):
	loss_options = ['l1', 'l2', 'l1_pro', 'l2_pro', 'cross_entropy', 'marginloss']
	# main_loss = opt['train']['loss']
	# if main_loss in loss_options:
	# 	loss_options.remove(main_loss)
	length = len(loss_options)
	loss_idx = np.random.randint(length)
	opt['train']['loss'] = loss_options[loss_idx]
	return opt


def variate_optim(opt):
	optim_options = ['SGD',
	                 'Adam', 'Adadelta', 'Adagrad']
	optim_params = [{'momentum':0.9},
	                {'betas':(0.9, 0.999)},
	                {'rho':0.9},
	                {'lr_decay':0}]
	# main_optim = opt['train']['optim']
	lr = opt['train']['optimizer_param']['lr']
	weight_decay = opt['train']['optimizer_param']['weight_decay']

	# if main_optim in optim_options:
	# 	idx = optim_options.index(main_optim)
	# 	optim_options.remove(main_optim)
	# 	del optim_params[idx]
	length = len(optim_options)
	optim_idx = np.random.randint(length)
	opt['train']['optim'] = optim_options[optim_idx]
	opt_param = optim_params[optim_idx]
	opt_param['lr'] = lr
	opt_param['weight_decay'] = weight_decay
	opt['train']['optimizer_param'] = opt_param
	return opt


def variate_hyperparam(opt):
	for k, v in opt['train']['optimizer_param'].items():
		if isinstance(v, float):
			if 'momentum' in k:
				opt['train']['optimizer_param'][k] = 0.8 + 0.2 * np.random.rand()
			else:
				opt['train']['optimizer_param'][k] = v * (0.5 + np.random.rand())

	for k, v in opt['train']['lr_scheme_param'].items():
		if isinstance(v, float):
			if 'base_momentum' in k:
				opt['train']['lr_scheme_param'][k] = 0.8 + 0.2 * np.random.rand()
			elif 'max_momentum' in k:
				base_m = opt['train']['lr_scheme_param']['base_momentum']
				opt['train']['lr_scheme_param'][k] = base_m + (1-base_m) * np.random.rand()
			else:
				opt['train']['lr_scheme_param'][k] = v * (0.5 + np.random.rand())

	return opt


def variate_model(opt):
	'''
	Just for testing!!
	:param opt:
	:return: changed opt
	'''
	if 'resnet' in opt['network'].lower():
		print("You are varying models!! This is only for testing!!")
		res_option = ['18', '34', '50', '101', '152']
		length = len(res_option)
		main_option = opt['network'].lower().split('resnet')[-1]
		model_idx = np.random.randint(length)
		opt['network'] = opt['network'].replace(main_option, res_option[model_idx])
	return opt


def variate_sample(opt):
	opt['datasets']['train']['resample'] = opt['varyOnSample']
	return opt


def variate_cv(opt):
	opt['datasets']['train']['fold'] = opt['task_id']
	return opt


def variate_dataset(opt, task_id):
	length = len(opt['datasets'])
	# data_idx = np.random.randint(length)
	data_idx = task_id % length

	for idx, (phase, dataset_opt) in enumerate(opt['datasets'].items()):
		if data_idx == idx:
			opt['datasets'] = {}
			opt['datasets'] = dataset_opt

	return opt


def get_random_seed(seed):
	random_array = [81841, 82684, 79917, 90352, 30648, 34509, 28490, 22093, 78832,
        2726, 92261, 95367, 31926, 13802, 33594,  7363, 54524, 16064,
       45119, 19590, 61275, 19905, 82184, 87843, 22175,  5122, 52753,
       85702, 19000, 24631, 25072, 49850, 92835, 74718, 74066, 59800,
       56101, 21446, 41267, 85812, 88094, 95600, 38636, 62170, 34532,
       83904, 65802,  2609, 65972, 96104, 52330, 36597, 26773, 78675,
       43959, 48363, 90149, 36880, 24330, 61334, 44836, 70951, 72347,
       81827, 37159, 44001, 18636,  2277, 82548, 93100, 59198, 14284,
       87218, 82413, 74296, 71197, 55979, 55032,  4595, 84230, 59562,
       89016, 61006, 47732, 50116, 47141, 85664, 10977, 47504, 34438,
       87737, 77848, 57584, 57598, 31536, 60259, 77452, 41150, 70093,
       54833]
	index = seed % len(random_array)
	return random_array[index]


def parse(parser, task_id):
	# remove comments starting with '//'
	opt_path = parser.opt
	json_str = ''
	with open(opt_path, 'r') as f:
		for line in f:
			line = line.split('//')[0] + '\n'
			json_str += line
	opt = json.loads(json_str, object_pairs_hook=OrderedDict)

	# Need to set seed here, for deterministic variate options
	if parser.seed is None:
		seed = np.random.randint(100000)
	else:
		seed = parser.seed
		seed = get_random_seed(seed)
	opt['ntasks'] = parser.ntasks
	np.random.seed(seed)
	seed_list = np.random.randint(1, 10000, opt['ntasks'])
	seed_list[0] = seed
	opt['manual_seed'] = seed

	# TODO: Warning, if initializing multiple tasks in one rank, all seeds will be set to that of the last task
	# print('task id {} use seed {}'.format(task_id, seed_list[task_id]))
	util.set_random_seed(seed_list[task_id])
	# util.set_random_seed(opt['manual_seed'])

	opt['is_train'] = parser.train
	opt['task_id'] = task_id
	if parser.val_freq:
		opt['val_freq'] = parser.val_freq

	opt['device_per_task'] = parser.gsize

	if parser.name:
		opt['name'] = parser.name

	if parser.model:
		new_model = opt['network'].split('-')
		new_model[-1] = parser.model
		opt['network'] = '-'.join(new_model)

	# path
	for key, path in opt['path'].items():
		if path and key in opt['path']:
			opt['path'][key] = os.path.expanduser(path)
	experiments_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
	opt['path']['experiments_root'] = experiments_root
	opt['path']['log'] = experiments_root
	opt['path']['task'] = os.path.join(experiments_root, 'task{}'.format(task_id))

	# datasets
	dconfig_path = opt['path']['data_config']
	with open(dconfig_path, 'r') as dconfig_file:
		try:
			data_config = json.load(dconfig_file, object_pairs_hook=OrderedDict)
		except json.decoder.JSONDecodeError:
			print(dconfig_file.readlines())

	sysstr = platform.system()
	for kind, set_name in opt['datasets'].items():
		opt['datasets'][kind] = data_config[set_name]
		for phase, dataset in opt['datasets'][kind].items():
			dataset['name'] = set_name
			dataset['run'] = parser.seed
			# if phase == 'train':
			# 	dataset['is_train'] = True
			# else:
			# 	dataset['is_train'] = False
			# dataset['phase'] = phase
			# dataset['no_split'] = True
			dataset['lmdb'] = parser.lmdb
			if sysstr == 'Windows':
				dataset['n_workers'] = 0
			else:
				dataset['n_workers'] = 0    ## TODO: n_workers > 0 will cause multiprocessing error
			if parser.droot is not None:
				dataset['dataroot'] = os.path.expanduser(os.path.join(parser.droot, dataset['name']))
			elif 'dataroot' in dataset and dataset['dataroot'] is not None:
				dataset['dataroot'] = os.path.expanduser(os.path.join(dataset['dataroot'], dataset['name']))

	opt['path']['models'] = os.path.join(opt['path']['task'], 'models')
	# if not opt['is_train']:
	# if opt['path']['pretrain_model'] is not None:
	# 	opt['path']['pretrain_model'] = os.path.join(opt['path']['models'], opt['path']['pretrain_model'])
	# else:
	# 	opt['path']['pretrain_model'] = None

	opt['path']['training_state'] = os.path.join(opt['path']['task'], 'training_state')
	opt['path']['pred'] = os.path.join(opt['path']['task'], 'pred')
	opt['att_transfer'] = parser.a

	if opt['is_train']:
		opt['kd_transfer'] = parser.k
		opt['att_transfer'] = parser.a
		opt['fsp_transfer'] = parser.f
		opt['w_transfer'] = parser.w
		opt['ws_transfer'] = parser.ws
		opt['varyOnLoss'] = parser.VL
		opt['varyOnOptim'] = parser.VO
		opt['varyOnHyper'] = parser.VH
		opt['varyOnData'] = parser.VD
		opt['varyOnSample'] = parser.VS
		opt['varyOnCV'] = parser.VF
		opt['rancl'] = parser.rancl
		opt['create_val'] = parser.create_val

		if 'resume_state' in opt['path'].keys() and opt['path']['resume_state']:
			main_state = opt['path']['resume_state']
			opt['path']['resume_state'] = os.path.join(opt['path']['training_state'], main_state)

		# opt = variate_dataset(opt, 1)
		if opt['varyOnData']:
			# max_classes = max([dataset_opt['train']['num_classes'] for phase, dataset_opt in opt['datasets'].items()])
			# for idx, (phase, dataset_opt) in enumerate(opt['datasets'].items()):
			# 	opt['datasets'][phase]['train']['num_classes'] = max_classes
			opt = variate_dataset(opt, task_id)
		else:
			opt['datasets'] = opt['datasets']['main']

		if opt['varyOnCV']:
			opt = variate_cv(opt)

		if task_id != 0:
			if opt['varyOnLoss']:
				opt = variate_loss(opt)

			if opt['varyOnOptim']:
				opt = variate_optim(opt)

			if opt['varyOnHyper']:
				opt = variate_hyperparam(opt)

			if opt['varyOnSample']:
				opt = variate_sample(opt)
			# for testing only!!
			# opt = variate_model(opt)
			# opt['network'] = "Cifar_ResNet34"
		else:
			# opt['datasets'] = opt['datasets']['main']
			opt['datasets']['train']['resample'] = False
			# if opt['varyOnSample']:
			# 	opt['datasets']['train']['no_split'] = False
			# 	opt['datasets']['val']['no_split'] = False

		# change some options for debug mode
		if 'debug' in opt['name']:
			opt['niter'] = 30
			opt['train']['lr_scheme_param']['milestones'] = [10, 20]
			opt['trans_freq'] = 5
			opt['val_freq'] = 5
			opt['logger']['print_freq'] = 1
			opt['logger']['save_checkpoint_freq'] = 4
			# opt['train']['lr_decay_iter'] = 10
	else:  # test
		opt['path']['results'] = os.path.join(opt['path']['task'], 'results'.format(task_id))
		opt['datasets'] = opt['datasets']['main']

	return opt


def mtl_parse(opt):
	return opt


class NoneDict(dict):
	def __missing__(self, key):
		return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
	if isinstance(opt, dict):
		new_opt = dict()
		for key, sub_opt in opt.items():
			new_opt[key] = dict_to_nonedict(sub_opt)
		return NoneDict(**new_opt)
	elif isinstance(opt, list):
		return [dict_to_nonedict(sub_opt) for sub_opt in opt]
	else:
		return opt


def dict2str(opt, indent_l=1):
	'''dict to string for logger'''
	msg = ''
	for k, v in opt.items():
		if isinstance(v, dict):
			msg += ' ' * (indent_l * 2) + k + ':[\n'
			msg += dict2str(v, indent_l + 1)
			msg += ' ' * (indent_l * 2) + ']\n'
		else:
			msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
	return msg


def check_resume(opt):
	'''Check resume states and pretrain_model paths'''
	logger = logging.getLogger('base')
	if opt['path']['resume_state']:
		if opt['path']['pretrain_model']:
			logger.warning('pretrain_model path will be ignored when resuming training.')

		# main_pretrained = opt['path']['pretrain_model']
		# opt['path']['pretrain_model'] = main_pretrained.replace('task0', 'task{}'.format(task_id))

		state_idx = osp.basename(opt['path']['resume_state']).split('.')[0]
		opt['path']['pretrain_model'] = osp.join(opt['path']['models'],
												   '{}.pth'.format(state_idx))
		logger.info('Set [pretrain_model] to ' + opt['path']['pretrain_model'])
		if 'gan' in opt['model']:
			opt['path']['pretrain_model_D'] = osp.join(opt['path']['models'],
													   '{}_D.pth'.format(state_idx))
			logger.info('Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])
