import suppress_logs 

import argparse
import pandas as pd
import json
import os
from tqdm import tqdm
from dataloader import load_graph_from_zip
from retriever import EmbeddingRetriever
from llm_judge import QwenJudge
from notifier import TelegramLogger 

def get_args():
    parser = argparse.ArgumentParser(description="Taxonomy Enrichment Runner")
    
    parser.add_argument("--zip_path", type=str, default="data/ruwordnet.zip", help="Path to taxonomy zip")
    parser.add_argument("--mode", type=str, choices=["nouns", "verbs"], required=True, help="Track: nouns or verbs")
    parser.add_argument("--phase", type=str, choices=["public", "private"], default="public", help="Test set phase")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--output_dir", type=str, default=".", help="Where to save results")
    
    return parser.parse_args()

def get_ancestors(node_id, df_relations, max_depth=5):
    chain = []
    curr = node_id
    for _ in range(max_depth):
        parents = df_relations[df_relations['child_id'] == curr]
        if len(parents) == 0: break
        curr = parents.iloc[0]['parent_id']
        chain.append(curr)
    return chain

def main():
    args = get_args()
    
    with TelegramLogger():
        print(f"Configuration: {args.mode.upper()} | {args.phase.upper()} | Batch: {args.batch_size}")

        
        xml_type = "N" if args.mode == "nouns" else "V"
        test_file = f"data/{args.phase}_test/{args.mode}_{args.phase}.tsv"
        
        print(f"Loading Graph ({xml_type})...")
        df_search, df_relations = load_graph_from_zip(args.zip_path, xml_filter=f".{xml_type}.xml")
        
        print(f"Loading Test File: {test_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Missing test file: {test_file}")
            
        df_test = pd.read_csv(test_file, sep='\t', header=None, names=['text', 'context'])
        
        
        retriever = EmbeddingRetriever(df_search)
        judge = QwenJudge() 

        submission = {}
        all_orphans = df_test['text'].tolist()
        all_contexts = df_test['context'].tolist()
        
        # 3. Batch Loop
        print(f"Starting Inference on {len(all_orphans)} items...")
        
        for i in tqdm(range(0, len(all_orphans), args.batch_size)):
            batch_orphans = all_orphans[i : i + args.batch_size]
            batch_contexts = all_contexts[i : i + args.batch_size]
            
            # Combine
            batch_queries = [f"query: {o} . {c}" for o, c in zip(batch_orphans, batch_contexts)]
            
            
            candidates_batch = retriever.get_candidates_batch(batch_queries, k=50) 
            best_ids = judge.select_best_candidate_batch(batch_orphans, candidates_batch)
            
            
            for orphan, best_id, candidates in zip(batch_orphans, best_ids, candidates_batch):
                ancestors = get_ancestors(best_id, df_relations)
                final_list = [best_id] + ancestors
                
                for cid, _ in candidates:
                    if cid not in final_list: final_list.append(cid)
                    if len(final_list) >= 10: break
                
                submission[orphan] = final_list[:10]
                
           
            if i % (args.batch_size * 5) == 0:
                temp_path = os.path.join(args.output_dir, "submission_checkpoint.json")
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(submission, f, ensure_ascii=False)

        output_filename = f"submission_{args.mode}_{args.phase}.json"
        save_path = os.path.join(args.output_dir, output_filename)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(submission, f, ensure_ascii=False, indent=4)
            
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()