import numpy as np
from scipy import special
# import pysnooper

# np.set_printoptions(precision=3)
# corr_list = np.ones(5)
# print('select_prob, choice,  accuracy_changed,  corr_delta,  corr_list')
# for i in range(20):
#
# 	se_prob = special.softmax(corr_list)
#
# 	choice = np.random.choice(len(se_prob), 1, p=se_prob, replace=False)
#
# 	accuracy_changed = np.random.randn(1)
#
# 	corr_delta = np.tanh(accuracy_changed)
#
# 	for i, cd in enumerate(corr_delta):
# 		corr_list[choice[i]] += cd
#
# 	print('{}|{}|{}|{}|{}'.format(se_prob, choice, accuracy_changed, corr_delta, corr_list))
# 	# print(choice, se_prob, accuracy_changed, corr_delta, corr_list)
# @pysnooper.snoop()
def test():
	task_id = 7
	running_tasks = np.array([2, 4, 7, 5])
	ranks = [1,3,5]
	corr_list = [0, 0, 0, 0.4, -1, 0.01, 1.7, 0.6]
	temperature = 1
	device_per_task = 3
	rank = 1
	num_solver = 3

	self_index = np.where(running_tasks==task_id)
	other_running_tasks = np.delete(running_tasks, self_index)
	tasks_index = list(map(lambda x: x - 1 if x >= task_id else x, other_running_tasks))
	temp_corrlist = np.array([corr_list[i] for i in tasks_index])

	prob = special.softmax(temp_corrlist / temperature)
	selected_index = np.random.choice(len(prob), device_per_task-1, p=prob, replace=False).astype(np.int32)
	selected_tasks = other_running_tasks[selected_index]
	related_index = np.array([np.where(running_tasks==i)[0][0] for i in selected_tasks])
	related_ranks = related_index * device_per_task
	# the first place is the target rank, while the others are source ranks
	related_ranks = np.insert(related_ranks, 0, rank).astype(np.int32)

	related_table = np.zeros([num_solver, len(related_ranks)], dtype=np.int32)
	print(related_table)
test()