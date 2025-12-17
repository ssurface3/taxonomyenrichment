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
    orphans = df_test['text'].tolist()

    retriever = EmbeddingRetriever(df_search)
    judge = QwenJudge()

    submission = {}


    print("Starting Prediction Loop...")
    for i, orphan in tqdm(enumerate(orphans), total=len(orphans)):

        candidates = retriever.get_candidates(orphan, k=20)
        
        best_id = judge.select_best_candidate(orphan, candidates)
        
        ancestors = get_ancestors(best_id, df_relations)
        
        final_list = [best_id] + ancestors
        if i % 20 == 0:
            best_name = df_search[df_search['id'] == best_id].iloc[0]['name']
            
            print(f"\n{i} Word: '{orphan}' -> Selected: '{best_name}' ({best_id})")
            print(f"top 3 Candidates were: {[c[1] for c in candidates[:3]]}")
        
        for cid, _ in candidates:
            if cid not in final_list:
                final_list.append(cid)
            if len(final_list) >= 10:
                break
                
        submission[orphan] = final_list[:10]

    with open('submission.json', 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)
    
    print("Done! Saved to submission.json")

if __name__ == "__main__":
    main()