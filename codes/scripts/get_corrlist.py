import numpy as np
import pandas


def get_corrlist(path):
	metric_dict = {}
	with open(path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			if 'last' in line and 'corr_list' in line:
				task_id = int(line.split('Task ')[-1].split('-')[0])
				corrlist = line.split('corr_list: ')[-1].strip('\n').strip('[]').split()
				# f1_score = float(line.split('F1 score: ')[-1])
				metric_dict[task_id] = list(map(lambda x: float(x), corrlist))

	return [metric_dict[i] for i in range(len(metric_dict))]


list1 = get_corrlist('./cifar10_wt_VS_1000.txt')
list_array = []
header = []
for i in range(len(list1)):
	list1[i].insert(i, 0)
	list_array.append(list1[i])
	header.append(i)
np_array = np.array(list_array).transpose()
df = pandas.DataFrame(np_array).round(2)
df.to_csv('./cifar10_wt_VS_1000.csv')
print(df)
