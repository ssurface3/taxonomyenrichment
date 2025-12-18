import pandas as pd
import json
from tqdm import tqdm
from dataloader import load_graph_from_zip
from retriever import EmbeddingRetriever
from llm_judge import QwenJudge

def get_ancestors(node_id, df_relations, max_depth=5):
    chain = []
    curr = node_id
    for _ in range(max_depth):
        parents = df_relations[df_relations['child_id'] == curr]
        if len(parents) == 0:
            break
        curr = parents.iloc[0]['parent_id']
        chain.append(curr)
    return chain

def main():
    df_search, df_relations = load_graph_from_zip('/kaggle/working/taxonomy-enrichment/data/ruwordnet.zip')
    
    df_test = pd.read_csv('/kaggle/working/taxonomy-enrichment/data/public_test/verbs_public.tsv', sep='\t', header=None, names=['text', 'context'])
    
   
    retriever = EmbeddingRetriever(df_search)
    judge = QwenJudge()

    submission = {}

    print(f"Predicting {len(df_test)} verbs using Context...")

    for i, row in tqdm(df_test.iterrows(), total=len(df_test)):
        orphan = row['text']
        context = row['context']
        
        search_query = f"{orphan} . {context}" 
        candidates = retriever.get_candidates(search_query, k=25) 
        
        best_id = judge.select_best_candidate(orphan, candidates)
        
        
        ancestors = get_ancestors(best_id, df_relations)
        final_list = [best_id] + ancestors
        
    
        for cid, _ in candidates:
            if cid not in final_list: final_list.append(cid)
            if len(final_list) >= 10: break
                
        submission[orphan] = final_list[:10]

    
    with open('submission_verbs.json', 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)
    
    print("✅ Logic Updated. Run the converter next.")

if __name__ == "__main__":
    main()