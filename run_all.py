from main import *
import logging
import pandas as pd
import sys
import pickle
import time

# set the output
time_str = time.strftime("%Y%m%d-%H%M%S")
log_filename = 'run_all_' + time_str + '.log'
sys.stdout = open(log_filename, 'w')

# record all results
all_exp_metas = []
all_metrics = []

data_dir = '../StudentLearningProcess/'
arguments = []
funcs = []

run_info = 'dkv'
sum_file = 'log_summary.txt'
f = open(sum_file, 'a')
f.write(time_str + '\n')
f.write(run_info + '\n')
f.flush()
f.close()

# Generate experiments
# experiments for DKT
all_sets = ['Assistment09-problem-single_skill.csv', 'Assistment12-problem-single_skill.csv', 'Assistment15-skill.csv',
            'kdd_data_2005.csv', 'kdd_data_2006.csv', 'kdd_data_bridge_2006.csv', 'kdd_data_2005_2.csv',
            'kdd_data_2006_2.csv', 'kdd_data_bridge_2006_2.csv']

batch_sizes = dict(zip(all_sets, [64, 32, 16, 16, 16, 16, 32, 16, 16]))

if 1:
	lrs = [0.03, 0.1, 0.3, 1.0]
	l2_regs = [0.0, 1e-4, 3e-4, 1e-3, 3e-3]
	items = ['skill', 'problem']
	modes = ['new user', 'most recent']
	for mode in modes:
		for item in items:
			for lr in lrs:
				for l2_reg in l2_regs:
					exp_meta = {}
					exp_meta['data_file'] = all_sets[0]
					exp_meta['mode'] = mode
					exp_meta['item'] = item
					exp_meta['pretrain_flag'] = False
					exp_meta['model'] = 'DKVMN_bi'
					exp_meta['initial_lr'] = lr
					exp_meta['l2_reg'] = l2_reg
					arguments.append((data_dir + exp_meta['data_file'], exp_meta['mode'], exp_meta['item'],
					                  exp_meta['pretrain_flag'], exp_meta['model'], exp_meta['l2_reg'], exp_meta['initial_lr']))
					funcs.append(DKVMN_exp)
					all_exp_metas.append(exp_meta)


if 1:
	lrs = [0.01, 0.03, 0.1, 0.3, 1.0]
	l2_regs = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
	items = ['skill', 'problem']
	modes = ['new user', 'most recent']
	for mode in modes:
		for item in items:
			for lr in lrs:
				for l2_reg in l2_regs:
					exp_meta = {}
					exp_meta['data_file'] = all_sets[3]
					exp_meta['mode'] = mode
					exp_meta['item'] = item
					exp_meta['pretrain_flag'] = False
					exp_meta['model'] = 'DKVMN_bi'
					exp_meta['initial_lr'] = lr
					exp_meta['l2_reg'] = l2_reg
					arguments.append((data_dir + exp_meta['data_file'], exp_meta['mode'], exp_meta['item'],
					                  exp_meta['pretrain_flag'], exp_meta['model'], exp_meta['l2_reg'], exp_meta['initial_lr']))
					funcs.append(DKVMN_exp)
					all_exp_metas.append(exp_meta)


if 0:
	file_names = [all_sets[i] for i in [0, 3]]
	modes = ['new user']
	items = ['skill', 'problem']
	pretrain_flags = [False, True]
	for file_name in file_names:
		for mode in modes:
			for item in items:
				for pretrain_flag in pretrain_flags:
					exp_meta = {}
					exp_meta['model'] = 'DKT_embedding'
					exp_meta['data_file'] = file_name
					exp_meta['mode'] = mode
					exp_meta['item'] = item
					exp_meta['pretrain_flag'] = pretrain_flag
					exp_meta['model'] = 'DKVMN_bi'
					arguments.append((data_dir + exp_meta['data_file'], exp_meta['mode'], exp_meta['item'],
					                  exp_meta['pretrain_flag'], exp_meta['model']))
					funcs.append(DKVMN_exp)
					all_exp_metas.append(exp_meta)

# Perform experiments
results_file_name = 'all_results_' + time_str + '.log'
f = open(results_file_name, 'w')

file_name = 'all_results_' + time_str + '.pickle'

for idx, exp_meta in enumerate(all_exp_metas):
	print(idx, exp_meta)

exception_list = []
good_exp_metas = []
print(arguments)
for idx, argument in enumerate(arguments):
	try:
		t = time.time()
		print('=' * 150)
		print(all_exp_metas[idx])
		print('=' * 150)
		# metric = Factorization_exp(*argument)
		metric = funcs[idx](*argument)
		all_metrics.append(metric)
		good_exp_metas.append(all_exp_metas[idx])
		print('Elapsed time: ', time.time() - t)
		print(good_exp_metas[-1], all_metrics[-1])
		f.write(str([good_exp_metas[-1], all_metrics[-1]]).strip('[]') + '\n')
		f.flush()
		dbfile = open(file_name, 'wb')
		results = pd.DataFrame(list(zip(good_exp_metas, all_metrics)), columns=['settings', 'testing_results'])
		pickle.dump(results, dbfile)
		dbfile.flush()
	except Exception as e:
		exception_list.append(all_exp_metas[idx])
		print("An exception occurred when ", idx, all_exp_metas[idx])
		print('Exception info', e)
		print(logging.exception(e))
		logging.exception(e)
		continue
	finally:
		pass

f.close()
dbfile.close()

# save results to csv files
results_file_name = 'all_results_' + time_str + '.csv'
results.to_csv(results_file_name)

# read out to verify
dbfile = open(file_name, 'rb')
results = pickle.load(dbfile)
print(results)
dbfile.close()
