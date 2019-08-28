import tensorflow as tf
import numpy as np
from model import Model
from model_inter import Model_bi
import os, time, argparse
from data_loader import *
import sys
import pandas as pd
from WeightedMatrixFac import *

sys.path.append('../Recurrent_embedding')
from helper import *


def DKVMN_exp(dataset_file, mode='new user', item='skill', pretrain=True, model='DKVMN_bi'):

	item_id = item + '_id'
	file_path = dataset_file
	dataset = 'assist2009_updated'

	parser = argparse.ArgumentParser()
	parser.add_argument('--num_epochs', type=int, default=300)
	parser.add_argument('--train', type=str2bool, default='t')
	parser.add_argument('--init_from', type=str2bool, default='f')
	parser.add_argument('--show', type=str2bool, default='f')
	parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
	parser.add_argument('--log_dir', type=str, default='logs')
	parser.add_argument('--data_dir', type=str, default='data')
	parser.add_argument('--anneal_interval', type=int, default=20)
	parser.add_argument('--maxgradnorm', type=float, default=50.0)
	parser.add_argument('--momentum', type=float, default=0.9)
	parser.add_argument('--initial_lr', type=float, default=0.05)
	parser.add_argument('--l2_reg', type=float, default=0.003)
	parser.add_argument('--model', type=str, default=model)
	parser.add_argument('--item_id', type=str, default=item_id)

	# synthetic / assist2009_updated / assist2015 / STATIC
	if dataset == 'assist2009_updated' and model == 'DKVMN':
		parser.add_argument('--batch_size', type=int, default=32)
		parser.add_argument('--memory_size', type=int, default=20)
		parser.add_argument('--memory_key_state_dim', type=int, default=50)
		parser.add_argument('--memory_value_state_dim', type=int, default=200)
		parser.add_argument('--final_fc_dim', type=int, default=50)
		parser.add_argument('--seq_len', type=int, default=200)

	elif dataset == 'assist2009_updated' and model == 'DKVMN_bi':
		parser.add_argument('--batch_size', type=int, default=32)
		parser.add_argument('--memory_size', type=int, default=1)
		parser.add_argument('--memory_key_state_dim', type=int, default=50)
		parser.add_argument('--memory_value_state_dim', type=int, default=50)
		parser.add_argument('--final_fc_dim', type=int, default=50)
		parser.add_argument('--seq_len', type=int, default=200)

	elif dataset == 'kdd2005' and model == 'DKVMN':
		parser.add_argument('--batch_size', type=int, default=32)
		parser.add_argument('--memory_size', type=int, default=20)
		parser.add_argument('--memory_key_state_dim', type=int, default=50)
		parser.add_argument('--memory_value_state_dim', type=int, default=200)
		parser.add_argument('--final_fc_dim', type=int, default=50)
		parser.add_argument('--seq_len', type=int, default=200)

	elif dataset == 'kdd2005' and model == 'DKVMN_bi':
		parser.add_argument('--batch_size', type=int, default=32)
		parser.add_argument('--memory_size', type=int, default=1)
		parser.add_argument('--memory_key_state_dim', type=int, default=50)
		parser.add_argument('--memory_value_state_dim', type=int, default=50)
		parser.add_argument('--final_fc_dim', type=int, default=50)
		parser.add_argument('--seq_len', type=int, default=200)

	elif dataset == 'synthetic':
		parser.add_argument('--batch_size', type=int, default=32)
		parser.add_argument('--memory_size', type=int, default=5)
		parser.add_argument('--memory_key_state_dim', type=int, default=10)
		parser.add_argument('--memory_value_state_dim', type=int, default=10)
		parser.add_argument('--final_fc_dim', type=int, default=50)
		parser.add_argument('--n_questions', type=int, default=50)
		parser.add_argument('--seq_len', type=int, default=50)

	elif dataset == 'assist2015':
		parser.add_argument('--batch_size', type=int, default=50)
		parser.add_argument('--memory_size', type=int, default=20)
		parser.add_argument('--memory_key_state_dim', type=int, default=50)
		parser.add_argument('--memory_value_state_dim', type=int, default=100)
		parser.add_argument('--final_fc_dim', type=int, default=50)
		parser.add_argument('--n_questions', type=int, default=100)
		parser.add_argument('--seq_len', type=int, default=200)

	elif dataset == 'STATICS':
		parser.add_argument('--batch_size', type=int, default=10)
		parser.add_argument('--memory_size', type=int, default=50)
		parser.add_argument('--memory_key_state_dim', type=int, default=50)
		parser.add_argument('--memory_value_state_dim', type=int, default=100)
		parser.add_argument('--final_fc_dim', type=int, default=50)
		parser.add_argument('--n_questions', type=int, default=1223)
		parser.add_argument('--seq_len', type=int, default=200)

	args = parser.parse_args()
	args.dataset = dataset

	print(args)
	if not os.path.exists(args.checkpoint_dir):
		os.mkdir(args.checkpoint_dir)
	if not os.path.exists(args.log_dir):
		os.mkdir(args.log_dir)
	if not os.path.exists(args.data_dir):
		os.mkdir(args.data_dir)
		raise Exception('Need data set')

	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth = True
	train_data, test_data, train_user_data, test_user_data, stats = prepare_data(file_path=file_path, item_id=args.item_id)
	print(stats)
	args.n_questions = stats['num_items']

	data = DATA_LOADER(args.n_questions, args.seq_len, ',')
	data_directory = os.path.join(args.data_dir, args.dataset)

	with tf.Session(config=run_config) as sess:
		if args.model=='DKVMN':
			dkvmn = Model(args, sess, name='DKVMN')
		elif args.model=='DKVMN_bi':
			dkvmn = Model_bi(args, sess, name='DKVMN_bi')
		if args.train:
			train_q_data, train_qa_data = data.load_data(train_user_data)
			print('Train data loaded')
			valid_q_data, valid_qa_data = data.load_data(test_user_data)
			print('Valid data loaded')
			print('Shape of train data : %s, valid data : %s' % (train_q_data.shape, valid_q_data.shape))
			print('Start training')
			dkvmn.train(train_q_data, train_qa_data, valid_q_data, valid_qa_data)
			#print('Best epoch %d' % (best_epoch))
		else:
			test_q_data, test_qa_data = data.load_data(test_user_data)
			print('Test data loaded')
			dkvmn.test(test_q_data, test_qa_data)


def pretrain(item, embedding_size, train_data, test_data, stats):
	num_users = stats['num_users']
	num_skills = stats['num_items']
	num_probs = stats['num_items']
	# First get the embeddings by runing simple matrix factorization
	if item == 'problem' or item == 'concat':
		fm = MatrixFac(epsilon=40, _lambda=0.02, momentum=0.8, maxepoch=30, num_batches=300, batch_size=10000,
				problem=True, MF_prob=True, num_feat=embedding_size, user=True,  global_bias=False, MF_skill=False, skill=False,
				problem_dyn_embedding=False, patience=3)
		if 0:
			fm = MatrixFac(epsilon=10, _lambda=0.1, momentum=0.8, maxepoch=30, num_batches=300, batch_size=1000,
					problem=True, MF_prob=True, num_feat=embedding_size, user=True,  global_bias=False, MF_skill=False, skill=False,
					problem_dyn_embedding=False, patience=3)
	elif item == 'skill':
		fm = MatrixFac(epsilon=40, _lambda=0.1, momentum=0.8, maxepoch=30, num_batches=300, batch_size=10000,
				problem=False, MF_prob=False, num_feat=embedding_size, user=True,  global_bias=False, MF_skill=True, skill=True,
				problem_dyn_embedding=False, patience=3)
	ewflag = True
	train_data = add_weights(train_data, 5, ewflag)
	test_data = add_weights(test_data, 1, ewflag)
	fm.fit(train_data, test_data, num_users, num_probs, num_skills, prob_skill_map=None)
	metric = dict()
	metric['auc'] = fm.auc_test[-1]
	metric['acc'] = fm.acc_test[-1]
	metric['pre'] = fm.pre_test[-1]
	return fm, metric

def main():
	dataset_file = '../StudentLearningProcess/Assistment09-problem-single_skill.csv'
	DKVMN_exp(dataset_file, mode='new user', item='skill', pretrain=True, model='DKVMN_bi')


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'n', 'f', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Not expected boolean type')

if __name__ == "__main__":
	main()



