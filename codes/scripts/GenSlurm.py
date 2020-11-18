import warnings
import os
import shutil
from utils.util import mkdir


def gen_slurm(name, nrank, nrealtask, mem, hours, seed=0, trans_option=None, vary_option=None, resample_fraction=0., single=True, rancl=False, create_val=None,
              val_freq=None, model=None, prefix='', ozstar=True):
	gsize = 2

	if single and trans_option is not None:
		trans_option = None
		warnings.warn("when running with single option, the trans_option would be set as None!!")
	if prefix != '':
		prefix = '_'+prefix

	suffix = ''
	if single:
		suffix += "_single"
	if trans_option:
		suffix += "_{}".format(trans_option)
	if vary_option:
		suffix += "_{}".format(vary_option)
		if vary_option == 'VS':
			suffix += "_{}".format(resample_fraction)
	if rancl:
		suffix += "_rancl"
	if val_freq:
		suffix += "_{}".format(val_freq)
	if model:
		suffix += "_{}".format(model)
	if create_val:
		suffix += "_rval"
	if ozstar:
		suffix += "_ozstar"
	else:
		suffix += "_sstar"
	if prefix != '':
		suffix += prefix

	file_name = "run_{}_n{}_{}".format(name[0].lower(), nrealtask,  suffix)

	with open(os.path.join('shell', file_name), 'w', newline='\n') as f:
		f.writelines("#!/bin/bash\n")
		name_line = name[0]

		name_line += '{}_n{}'.format(suffix, nrealtask)

		if single:
			ntask_per_node = 1
		else:
			ntask_per_node = 2
		f.writelines("#SBATCH --job-name={}\n".format(name_line))
		f.writelines("#SBATCH --output={}.txt\n".format(name_line))
		f.writelines("#SBATCH --ntasks={}\n".format(nrank))
		f.writelines("#SBATCH --ntasks-per-node={}\n".format(ntask_per_node))
		f.writelines("#SBATCH --mem-per-cpu={}G\n".format(mem))
		f.writelines("#SBATCH --tmp=20g\n")
		f.writelines("#SBATCH --time={}:00:00\n".format(hours))
		if ozstar:
			f.writelines("#SBATCH --partition=skylake-gpu\n")
		f.writelines("#SBATCH --gres=gpu:{}\n".format(ntask_per_node))

		f.writelines("\n")
		# f.writelines("module load gcc/7.3.0\n")
		# f.writelines("module load anaconda3/5.1.0\n")
		f.writelines("module load cuda/9.2.88\n")
		f.writelines("module load cudnn/7.2.1-cuda-9.2.88\n")
		if ozstar:
			f.writelines(". /apps/skylake/software/core/anaconda3/5.1.0/etc/profile.d/conda.sh\n")
			f.writelines("conda activate /fred/oz019/envs/pytorch\n")
		else:
			f.writelines("module load gcc/7.3.0\n")
			f.writelines("module load openmpi/3.0.3\n")
			f.writelines(". /apps/sandybridge/software/core/anaconda3/5.1.0/etc/profile.d/conda.sh\n")
			f.writelines("conda activate /fred/oz019/envs/sstar\n")

		f.writelines("\n")
		# f.writelines("export PSM2_CUDA=1\n")
		f.writelines("exp_name=$SLURM_JOB_NAME\n")

		f.writelines("cd ..\n")

		for n in name:
			f.writelines("srun -N $SLURM_NNODES -n $SLURM_NNODES cp -a \"/fred/oz019/BasicMTO/data/{}\" \"$JOBFS/\"\n".format(n))

		for seed in range(5):
			last_line = "srun python mto.py -opt options/{}_classification.json -name {}_seed{} -train".format(name[0].lower(), name_line, seed)
			# last_line = "srun python mto.py -opt options/{}_classification_mtl.json -name \"$exp_name\" -train".format(name[0].lower())
			# last_line = "srun python scalable_mto.py -opt options/{}_classification.json -name \"$exp_name\" -train -ntask {}".format(name[0].lower(), nrealtask)
			if vary_option:
				last_line += " -{}".format(vary_option)
				if vary_option == 'VS':
					last_line += " {}".format(resample_fraction)

			if rancl:
				last_line += " -rancl"

			if create_val:
				last_line += " -create_val"

			if val_freq:
				last_line += " -val_freq {}".format(val_freq)

			if model:
				last_line += " -model {}".format(model)

			if trans_option:
				last_line += " -{}".format(trans_option)
				if trans_option == 'w':
					last_line += " -gsize {}".format(gsize)

			last_line += " -seed {}".format(seed)
			last_line += " -droot $JOBFS/"
			last_line += " -lmdb"
			last_line += "\n"
			f.writelines(last_line)
			# f.writelines("fi")

	return file_name


def batch_rs_tasks(model, is_train=True):
	# datasets = ['WHU19', 'RSSCN7', 'UCMerced', 'AID']
	datasets = ['RSSCN7', 'OxfordPets', 'UCMerced']
	# datasets = ['UCMerced']
	# datasets = ['CIFAR10', 'CIFAR100']
	if is_train:
		n_rank = 8
		real_tasks = 4
	else:
		n_rank = 1
		real_tasks = 1

	# [order.writelines(
	# 	'sbatch {}\n'.format(gen_slurm([d], 1, 1, 4, 8, seed, None, None, single=True, model=model, prefix='', ozstar=True))) for d in datasets]
	[order.writelines(
		'sbatch {}\n'.format(
			gen_slurm([d], n_rank, real_tasks, 6, 8, seed, trans_option='w', single=False, create_val=True,
			          rancl=True, model=model, prefix='', ozstar=True))) for d in datasets]
	# [order.writelines(
	# 		'sbatch {}\n'.format(
	# 			gen_slurm([d], n_rank, real_tasks, 16, 4, seed, vary_option='VF', resample_fraction=2, single=False,
	# 					  rancl=True, model=model, prefix=''))) for d in datasets]
	# [order.writelines(
	# 	'sbatch {}\n'.format(
	# 		gen_slurm([d], n_rank, real_tasks, 16, 4, seed, "w", 'VS', resample_fraction=2, single=False,
	# 		          rancl=True, val_freq=100000, model=model, prefix=''))) for d in datasets]
	# [order.writelines(
	# 	'sbatch {}\n'.format(
	# 		gen_slurm([d], 10, 5, 16, 4, seed, "w", 'VS', resample_fraction=2, single=False,
	# 		          rancl=True, model=model, prefix=''))) for d in datasets]
	# [order.writelines(
	# 	'sbatch {}\n'.format(
	# 		gen_slurm([d], 12, 6, 16, 4, seed, "w", 'VS', resample_fraction=2, single=False,
	# 		          rancl=True, model=model, prefix=''))) for d in datasets]
	# [order.writelines(
	# 	'sbatch {}\n'.format(
	# 		gen_slurm([d], 14, 7, 16, 4, seed, "w", 'VS', resample_fraction=2, single=False,
	# 		          rancl=True, model=model, prefix=''))) for d in datasets]

	# [order.writelines(
	# 	'sbatch {}\n'.format(
	# 		gen_slurm([d], n_rank, real_tasks, 16, 4, seed, "w", 'VD', single=False,
	# 		          rancl=True, model=model, prefix=''))) for d in datasets]

	# for rf in [0.6, 0.8, 1.0, 1.2, 1.4]:
	# 	[order.writelines(
	# 		'sbatch {}\n'.format(
	# 			gen_slurm([d], n_rank, real_tasks, 16, 10, seed, "k", None, resample_fraction=rf, single=False,
	# 			          rancl=True, model=model, prefix=''))) for d in datasets]

	# order.writelines(
	# 	'sbatch {}\n'.format(
	# 		gen_slurm(["RSSCN7"], n_rank, real_tasks, 16, 10, seed, "w", 'VS', resample_fraction=0.8, single=False,
	# 		          rancl=True, model=model, prefix='')))
	# order.writelines(
	# 	'sbatch {}\n'.format(
	# 		gen_slurm(["RSSCN7"], n_rank, real_tasks, 16, 10, seed, "w", 'VS', resample_fraction=0.6, single=False,
	# 		          rancl=True, model=model, prefix='')))
	# order.writelines(
	# 	'sbatch {}\n'.format(
	# 		gen_slurm(["RSSCN7"], n_rank, real_tasks, 16, 10, seed, "w", 'VS', resample_fraction=0.4, single=False,
	# 		          rancl=True, model=model, prefix='')))
	# order.writelines(
	# 	'sbatch {}\n'.format(gen_slurm(["AID"], 1, 1, 8, 4, seed, None, None, single=True, model=model, prefix='')))
	# order.writelines(
	# 	'sbatch {}\n'.format(
	# 		gen_slurm(["AID"], n_rank, real_tasks, 16, 4, seed, "w", 'VS', single=False, rancl=True, model=model, prefix='')))
	# order.writelines(
	# 	'sbatch {}\n'.format(gen_slurm(["UCMerced"], 1, 1, 8, 4, seed, None, None, single=True, model=model, prefix='')))
	# order.writelines(
	# 	'sbatch {}\n'.format(
	# 		gen_slurm(["UCMerced"], n_rank, real_tasks, 16, 4, seed, "w", 'VS', single=False, rancl=True, model=model, prefix='')))


with open("batchSlurm.sh", 'w', newline='\n') as order:
	if os.path.exists('shell'):
		shutil.rmtree('shell')
	mkdir('shell')

	for i in range(1):
		seed = i
		# batch_rs_tasks('ResNet50', is_train=True)
		batch_rs_tasks('squeezenet1_1', is_train=True)
		# batch_rs_tasks('MobileNet', is_train=True)
		batch_rs_tasks('densenet121', is_train=True)
		batch_rs_tasks('mobilenet_v2', is_train=True)
