import logging
logger = logging.getLogger('base')

def create_task(opt, **kwargs):
	model = opt['model']

	if model == 'classification':
		from .classification import ClassificationTask as Task
	elif model == 'classification_cv':
		from .classification_cv import ClassificationTask as Task
	elif model == 'classification_mtl':
		from .classification_mtl import ClassificationMTLTask as Task
	elif model == 'scalable_classification':
		from .scalable_classification_backup import ClassificationTask as Task
	elif model == 'generation':
		from .generation import GenerationTask as Task
	elif model == 'vae':
		from .vae_generation import VaeGenerationTask as Task
	elif model == 'scalable_segmentation':
		from .scalable_segmentation import SegmentationTask as Task
	else:
		raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
	task = Task(opt, **kwargs)
	logger.info('Task [{:s}] of {} is created.'.format(task.__class__.__name__, opt['task_id']))
	return task
