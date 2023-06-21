import os
import sys
import time
import argparse
import logging
from prettytable import PrettyTable

from args import add_args
import enums
from data.dataset import init_dataset


logger = logging.getLogger(__name__)


def prepare_dataset(args):
	logger.info('Loading and parsing datasets')
	# dataset = init_dataset(args=args, mode=enums.TRAINING_MODE_PRE_TRAIN, load_if_saved=False)
	dataset_bug_fix_small_train = init_dataset(args=args, mode=enums.TRAINING_MODE_FINE_TUNE, task=enums.TASK_BUG_FIX, language='small', split='train')
	dataset_bug_fix_small_valid = init_dataset(args=args, mode=enums.TRAINING_MODE_FINE_TUNE, task=enums.TASK_BUG_FIX, language='small', split='valid')
	dataset_bug_fix_small_test = init_dataset(args=args, mode=enums.TRAINING_MODE_FINE_TUNE, task=enums.TASK_BUG_FIX, language='small', split='test')
	# logger.info(f'The size of pre_training set: {len(dataset)}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])

	add_args(parser)

	main_args = parser.parse_args()

	if not os.path.exists(main_args.logging_file_path):
		os.makedirs(main_args.logging_file_path)

	# logging, log to both console and file, debug level only to file
	logger = logging.getLogger()
	logger.setLevel(level=logging.DEBUG)

	console = logging.StreamHandler()
	console.setLevel(level=logging.INFO)
	logger.addHandler(console)

	file = logging.FileHandler(os.path.join(main_args.logging_file_path, 'info.log'))
	file.setLevel(level=logging.INFO)
	formatter = logging.Formatter('[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s')
	file.setFormatter(formatter)
	logger.addHandler(file)

	# log command and configs
	logger.debug('COMMAND: {}'.format(' '.join(sys.argv)))

	config_table = PrettyTable()
	config_table.field_names = ["Configuration", "Value"]
	config_table.align["Configuration"] = "l"
	config_table.align["Value"] = "l"
	for config, value in vars(main_args).items():
		config_table.add_row([config, str(value)])
	logger.debug('Configurations:\n{}'.format(config_table))

	prepare_dataset(main_args)
