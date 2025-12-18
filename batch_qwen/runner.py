import pandas as pd
import json
from tqdm import tqdm
from dataloader import load_graph_from_zip
from retriever import EmbeddingRetriever
from llm_judge import QwenJudge

# BATCH SIZE: Increase this if VRAM is still empty. 
# 8 is safe. 16 might OOM.
BATCH_SIZE = 42

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
    # 1. Load Data
    df_search, df_relations = load_graph_from_zip('/kaggle/working/taxonomy-enrichment/data/ruwordnet.zip')
    
    # Ensure this is PUBLIC VERBS
    df_test = pd.read_csv('/kaggle/working/taxonomy-enrichment/data/private_test/nouns_private.tsv', sep='\t', header=None, names=['text', 'context'])
    
    # 2. Init Models
    retriever = EmbeddingRetriever(df_search)
    judge = QwenJudge()

    submission = {}
    
    # Prepare Lists
    all_orphans = df_test['text'].tolist()
    all_contexts = df_test['context'].tolist()
    total_items = len(all_orphans)

    print(f"🚀 Processing {total_items} items in batches of {BATCH_SIZE}...")

    # 3. Batch Loop
    for i in tqdm(range(0, total_items, BATCH_SIZE)):
        # Slice the batch
        batch_orphans = all_orphans[i : i + BATCH_SIZE]
        batch_contexts = all_contexts[i : i + BATCH_SIZE]
        
        # Combine for Retrieval
        batch_queries = [f"{o} . {c}" for o, c in zip(batch_orphans, batch_contexts)]
        
        # A. Batch Retrieve
        candidates_batch = retriever.get_candidates_batch(batch_queries, k=25)
        
        # B. Batch Judge
        best_ids = judge.select_best_candidate_batch(batch_orphans, candidates_batch)
        
        # C. Expand & Save (Sequential part - fast enough on CPU)
        for orphan, best_id, candidates in zip(batch_orphans, best_ids, candidates_batch):
            ancestors = get_ancestors(best_id, df_relations)
            final_list = [best_id] + ancestors
            
            # Fill
            for cid, _ in candidates:
                if cid not in final_list: final_list.append(cid)
                if len(final_list) >= 10: break
            
            submission[orphan] = final_list[:10]

    # Save
    with open('sub_nouns.json', 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)
        
    print("✅ Done.")

if __name__ == "__main__":
    main()