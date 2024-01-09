import pandas as pd 
import numpy as np
import os 
from scipy.stats import ttest_ind
import argparse

def calculate_run_file_mrr_score(run_file, k, qrels):
    run_file_original = {}
    with open(run_file, 'r') as f_run:
        for line in f_run:
            qid, did, _, _= line.strip().split("\t")
            if qid not in run_file_original: 
                run_file_original[qid] = []
            run_file_original[qid].append(did)

    mrr_score = []
    for qid in run_file_original:
        rr = 0.0
        for i, did in enumerate(run_file_original[qid][:k]):
            if qid in qrels and did in qrels[qid] and qrels[qid][did] > 0:
                rr = 1 / (i+1)
                break
        mrr_score.append([qid, rr])
    return mrr_score

def make_comparison_on_half_worst_queries(mrr_df, num_buckets):
    number_of_queries = int(len(mrr_df)/2)
    mrr_df = mrr_df.iloc[:number_of_queries, :] 
    num_queries = len(mrr_df)
    bucket_size = int(num_queries / num_buckets)
    mrr_scores_original = []
    mrr_scores_ours = []
    for i in range(0, num_buckets):
        tmp = mrr_df.iloc[ i * bucket_size: (i+1) * bucket_size , :]
        t, p = ttest_ind(tmp['mrr_original'].values.tolist(), tmp['mrr_ours'].values.tolist())
        print("------- chunk: {} -------".format(i + 1))
        print("p-value: {}".format(p))
        mrr_scores_original.append(tmp['mrr_original'].mean())
        mrr_scores_ours.append(tmp['mrr_ours'].mean())
    return mrr_scores_original, mrr_scores_ours

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-qrels', type=str, default='')
    parser.add_argument('-run_file_original', type=str, default='')
    parser.add_argument('-run_file_ours', type=str, default='')
    parser.add_argument('-metric', type=str, default='mrr_cut_10')
    parser.add_argument('-result', type=str, default='')
    args = parser.parse_args()

    
    metric = args.metric
    k = int(metric.split('_')[-1])
    
    qrels = {}
    with open(args.qrels, 'r') as f_qrel:
        for line in f_qrel:
            qid, _, did, label = line.strip().split()
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = int(label)
    
    mrr_score_original = pd.DataFrame(calculate_run_file_mrr_score(args.run_file_original, k, qrels), columns = ['qid', 'mrr_original'])
    mrr_score_original.sort_values(by = ['qid'],inplace = True)

    mrr_score_ours = pd.DataFrame(calculate_run_file_mrr_score(args.run_file_ours, k, qrels), columns = ['qid', 'mrr_ours'])
    mrr_score_ours.sort_values(by = ['qid'],inplace = True)

    mrr_df = pd.merge(mrr_score_original, mrr_score_ours, on = "qid", how = "left")
    mrr_df['qid'] = mrr_df['qid'].astype(int)
    mrr_df.sort_values(['mrr_original', 'qid'], ascending = [True, True], inplace = True)

    results = []
    performance_original_buckets, performance_ours_buckets = make_comparison_on_half_worst_queries(mrr_df, 5)
    stats_original = []
    stats_original.append(mrr_df['mrr_original'].mean())
    stats_original.append(-1)
    stats_original.extend(performance_original_buckets)
    stats_ours = []
    stats_ours.append(mrr_df['mrr_ours'].mean())
    stats_ours.extend(performance_ours_buckets)
    results.append(stats_original)
    results.append(stats_ours)
    results = pd.DataFrame(results)
    results.to_csv(args.result, index = False)

if __name__ == "__main__":
    main()
