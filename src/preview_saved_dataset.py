import argparse
import logging

from args import add_args
import enums
from data.dataset import init_dataset


logger = logging.getLogger(__name__)


def preview_saved_dataset(args):
	logger.info('Loading and parsing datasets')
	dataset = init_dataset(args=args, mode=enums.TRAINING_MODE_PRE_TRAIN, load_if_saved=True)
	print('Dataset length: ' + str(len(dataset)))
	for task in enums.PRE_TRAIN_TASKS:
		dataset.set_task(task)
		print('PRE_TRAIN_TASK:')
		print(task)
		print()
		print()
		print('SAMPLE 1:')
		_input, _output = dataset[0]
		print('input:')
		print(_input)
		print('output:')
		print(_output)
		print()
		print()
		print('SAMPLE 2:')
		_input, _output = dataset[1]
		print('input:')
		print(_input)
		print('output:')
		print(_output)
		print()
		print()
		print('SAMPLE 3:')
		_input, _output = dataset[2]
		print('input:')
		print(_input)
		print('output:')
		print(_output)
		print()
		print()
	logger.info(f'The size of pre_training set: {len(dataset)}')

	dataset_bug_fix_small_train =  init_dataset(args=args, mode=enums.TRAINING_MODE_FINE_TUNE, task=enums.TASK_BUG_FIX, language='small', split='valid')
	dataset_bug_fix_small_train.set_task('bug_fix')
	print('PRE_TRAIN_TASK:')
	print('bug_fix')
	print()
	print()
	print('SAMPLE 1:')
	_input, _output = dataset_bug_fix_small_train[0]
	print('input:')
	print(_input)
	print('output:')
	print(_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])

    add_args(parser)

    main_args = parser.parse_args()

    preview_saved_dataset(main_args)
