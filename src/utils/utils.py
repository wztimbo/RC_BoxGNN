# -*- coding: UTF-8 -*-

import os
import random
import logging
import torch
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, NoReturn, Any


def init_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def df_to_dict(df: pd.DataFrame) -> dict:
	res = df.to_dict('list')
	for key in res:
		res[key] = np.array(res[key])
	return res


def batch_to_gpu(batch: dict, device) -> dict:
	for c in batch:
		if type(batch[c]) is torch.Tensor:
			batch[c] = batch[c].to(device)
	return batch


def check(check_list: List[tuple]) -> NoReturn:
	# observe selected tensors during training.
	logging.info('')
	for i, t in enumerate(check_list):
		d = np.array(t[1].detach().cpu())
		logging.info(os.linesep.join(
			[t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]
		) + os.linesep)


def eval_list_columns(df: pd.DataFrame) -> pd.DataFrame:
	for col in df.columns:
		if pd.api.types.is_string_dtype(df[col]):
			df[col] = df[col].apply(lambda x: eval(str(x)))  # some list-value columns
	return df


def format_metric(result_dict: Dict[str, Any]) -> str:
	assert type(result_dict) == dict
	format_str = []
	metrics = np.unique([k.split('@')[0] for k in result_dict.keys()])
	topks = np.unique([int(k.split('@')[1]) for k in result_dict.keys() if '@' in k])
	if not len(topks):
		topks = ['All']
	for topk in np.sort(topks):
		for metric in np.sort(metrics):
			name = '{}@{}'.format(metric, topk)
			m = result_dict[name] if topk != 'All' else result_dict[metric]
			if isinstance(m, (float, np.floating)):
				format_str.append('{}:{:<.4f}'.format(name, m))
			elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
				format_str.append('{}:{}'.format(name, m))
	return ','.join(format_str)


def format_arg_str(args, exclude_lst: list, max_len=20) -> str:
	linesep = os.linesep
	arg_dict = vars(args)
	keys = [k for k in arg_dict.keys() if k not in exclude_lst]
	values = [arg_dict[k] for k in keys]
	key_title, value_title = 'Arguments', 'Values'
	key_max_len = max(map(lambda x: len(str(x)), keys))
	value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
	key_max_len, value_max_len = max([len(key_title), key_max_len]), max([len(value_title), value_max_len])
	horizon_len = key_max_len + value_max_len + 5
	res_str = linesep + '=' * horizon_len + linesep
	res_str += ' ' + key_title + ' ' * (key_max_len - len(key_title)) + ' | ' \
			   + value_title + ' ' * (value_max_len - len(value_title)) + ' ' + linesep + '=' * horizon_len + linesep
	for key in sorted(keys):
		value = arg_dict[key]
		if value is not None:
			key, value = str(key), str(value).replace('\t', '\\t')
			value = value[:max_len-3] + '...' if len(value) > max_len else value
			res_str += ' ' + key + ' ' * (key_max_len - len(key)) + ' | ' \
					   + value + ' ' * (value_max_len - len(value)) + linesep
	res_str += '=' * horizon_len
	return res_str


def check_dir(file_name: str):
	dir_path = os.path.dirname(file_name)
	if not os.path.exists(dir_path):
		print('make dirs:', dir_path)
		os.makedirs(dir_path)


def non_increasing(lst: list) -> bool:
	return all(x >= y for x, y in zip([lst[0]]*(len(lst)-1), lst[1:])) # update the calculation of non_increasing to fit ealry stopping, 2023.5.14, Jiayu Li


def get_time():
	return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)
def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=100):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop
def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, ground_truth, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain

        Low but correct defination
    """
    GT = set(ground_truth)
    if len(GT) > k :
        sent_list = [1.0] * k
    else:
        sent_list = [1.0]*len(GT) + [0.0]*(k-len(GT))
    dcg_max = dcg_at_k(sent_list, k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.
	
from sklearn.metrics import roc_auc_score
def AUC(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res