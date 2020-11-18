# An MTO-based DNN Training Algorithm with the Auxiliary Tasks Formulated From the Aspect of Data
The Pytorch source code of Chapter 5 in thesis [Training Deep Neural Networks via Multi-task Optimisation]
If you use this code, please cite the thesis.


****
# Usage

## Download data
First, you need to download the following datasets and put them in the corresponding directories in "./data/".
1. [AID](https://captain-whu.github.io/AID/)
2. [RSSCN7](https://github.com/palewithout/RSSCN7)
3. [UCMerced](http://weegee.vision.ucmerced.edu/datasets/landuse.html)
4. [WHU19](http://www.escience.cn/people/yangwen/WHU-RS19.html)

After unzip, each dataset folder contains categorical folders and 15 csv files.

## Training
Enter the "/codes" directory, and then,
1. Use the following script on Ozstar for MTO based training (e.g., on rsscn7 dataset). Note some parameters should be adapted to your own settings.
```
#!/bin/bash
#SBATCH --job-name=RSSCN7_w_rancl_densenet121_ozstar_n4
#SBATCH --output=RSSCN7_w_rancl_densenet121_ozstar_n4.txt
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=6G
#SBATCH --tmp=20g
#SBATCH --time=8:00:00
#SBATCH --partition=skylake-gpu
#SBATCH --gres=gpu:2

module load cuda/9.2.88
module load cudnn/7.2.1-cuda-9.2.88
. /apps/skylake/software/core/anaconda3/5.1.0/etc/profile.d/conda.sh
conda activate your_env_name

exp_name=$SLURM_JOB_NAME
cd ..
srun -N $SLURM_NNODES -n $SLURM_NNODES cp -a "/path/to/your/data/RSSCN7" "$JOBFS/"
srun python mto.py -opt options/rsscn7_classification.json -name RSSCN7_w_rancl_densenet121_ozstar_n4_seed0 -train -rancl -model densenet121 -w -gsize 2 -seed 0 -droot $JOBFS/ -lmdb
```

2. Use the following script for conventional training,
```
#!/bin/bash
#SBATCH --job-name=RSSCN7_single_densenet121_ozstar_n1_seed0
#SBATCH --output=RSSCN7_single_densenet121_ozstar_n1_seed0.txt
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=20g
#SBATCH --time=8:00:00
#SBATCH --partition=skylake-gpu
#SBATCH --gres=gpu:1

module load cuda/9.2.88
module load cudnn/7.2.1-cuda-9.2.88
. /apps/skylake/software/core/anaconda3/5.1.0/etc/profile.d/conda.sh
conda activate your_env_name

exp_name=$SLURM_JOB_NAME
cd ..
srun -N $SLURM_NNODES -n $SLURM_NNODES cp -a "/path/to/your/data/RSSCN7" "$JOBFS/"
srun python mto.py -opt options/rsscn7_classification.json -name "$exp_name" -train -model densenet121 -seed 0 -droot $JOBFS/ -lmdb
```
Note you can also run the conventional training on a single machine by the following command,
```
python mto.py -opt options/rsscn7_classification.json [-name <your_exp_name>] -train -model densenet121 -seed 0 -lmdb
```

After training, the generated models are stored in folder "experiments" which is in the same folder as "codes" and "data". If tensorboard is used, the results will be located in folder "tb_logger".

## Testing
In the default setting of training, the testing will be executed in the end. You can also just run the testing by using the following code,
```
python mto.py -opt options/rsscn7_classification.json [-name <your_exp_name>] -model densenet121 -seed 0 -lmdb
```
