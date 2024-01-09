import pandas as pd 
from sentence_transformers import SentenceTransformer
import os
import pandas as pd 
import os
from scipy import spatial
from tqdm import tqdm
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_original', type=str) #path to the original training of the model
    parser.add_argument('-model_ours', type=str) #path to the our training of the model
    parser.add_argument('-run_file_original', type=str) #path to the original run file
    parser.add_argument('-run_file_ours', type=str) #path to our run file
    parser.add_argument('-collection', type=str) #path to collection
    parser.add_argument('-queries', type=str) #path to queries(TSV format)
    parser.add_argument('-qrels', type=str) #path to qrels
    parser.add_argument('-output', type=str) #path to qrels
    args = parser.parse_args()

    queries = pd.read_csv(args.queries, sep ="\t", names = ['qid', 'query'])
    qrels = {}
    qrels_dict = {}
    with open(args.qrels, 'r') as f_qrel:
        for line in f_qrel:
            qid, _, did, label = line.strip().split()
            if qid not in qrels:
                qrels_dict[qid] = did
                qrels[qid] = {}
            qrels[qid][did] = int(label)

    corpus = {}
    collection_filepath = args.collection
    with open(collection_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")
            corpus[pid] = passage
            
    mrr_score_original = pd.DataFrame(calculate_run_file_mrr_score(args.run_file_original, 10, qrels), columns = ['qid', 'mrr_original'])
    mrr_score_original.sort_values(by = ['qid'],inplace = True)

    mrr_score_ours = pd.DataFrame(calculate_run_file_mrr_score(args.run_file_ours, 10, qrels), columns = ['qid', 'mrr_ours'])
    mrr_score_ours.sort_values(by = ['qid'],inplace = True)

    mrr_df = pd.merge(mrr_score_original, mrr_score_ours, on = "qid", how = "left")
    mrr_df['qid'] = mrr_df['qid'].astype(int)
    mrr_df.sort_values(['mrr_original', 'qid'], ascending = [True, True], inplace = True)

    number_of_queries = int(len(mrr_df)/2)
    mrr_df = mrr_df.iloc[:number_of_queries, :] 
    unique_qids = mrr_df['qid'].values.tolist()

    model_original = SentenceTransformer(args.model_original)
    sim_score_org = []
    average_sim_original = 0
    for qid in tqdm(unique_qids):
        query_text = queries[queries['qid']== qid]['query'].values.tolist()[0]
        passage = corpus[qrels_dict[str(qid)]]
        query_embedding = model_original.encode(query_text)
        passage_embedding = model_original.encode(passage)
        consine_sim_org = 1 - spatial.distance.cosine(query_embedding, passage_embedding)
        sim_score_org.append(consine_sim_org)
        average_sim_original += consine_sim_org

    model_ours = SentenceTransformer(args.model_ours)
    sim_score_ours = []
    average_sim_ours = 0
    for qid in tqdm(unique_qids):
        query_text = queries[queries['qid']== qid]['query'].values.tolist()[0]
        passage = corpus[qrels_dict[str(qid)]]
        query_embedding = model_ours.encode(query_text)
        passage_embedding = model_ours.encode(passage)
        consine_sim_ours = 1 - spatial.distance.cosine(query_embedding, passage_embedding)
        sim_score_ours.append(consine_sim_ours)
        average_sim_ours += consine_sim_ours

    final_df = pd.DataFrame()
    final_df['qid'] = unique_qids
    final_df['sim_score_org'] = sim_score_org
    final_df['sim_score_ours'] = sim_score_ours
    final_df.to_csv(args.output + "similarity_scores.csv", index = False)

if __name__ == "__main__":
    main()
