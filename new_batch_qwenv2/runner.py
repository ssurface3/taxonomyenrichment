import suppress_logs 
import argparse
import pandas as pd
import json
import os
import torch
import time
from tqdm import tqdm
from dataloader import load_graph_from_zip
from retriever import EmbeddingRetriever
from llm_judge import QwenJudge
from reranker import CrossEncoderReranker
from notifier import TelegramLogger

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="taxonomy-enrichment/data")
    parser.add_argument("--zip_name", type=str, default="ruwordnet.zip")
    parser.add_argument("--mode", type=str, choices=["nouns", "verbs"], required=True)
    parser.add_argument("--phase", type=str, choices=["public", "private"], default="public")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--top_k_retrieve", type=int, default=100)
    parser.add_argument("--top_k_rerank", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--skip_check", action="store_true")
    return parser.parse_args()

def get_ancestors(node_id, df_relations, max_depth=10):
    chain = []
    curr = node_id
    for _ in range(max_depth):
        parents = df_relations[df_relations['child_id'] == curr]
        if len(parents) == 0: break
        curr = parents.iloc[0]['parent_id']
        chain.append(curr)
    return chain
# def get_wiktionary_def(word):
#     try:
#         url = f"https://ru.wiktionary.org/wiki/{word}"
#         # Timeout to prevent hanging
#         response = requests.get(url, timeout=2)
#         if response.status_code == 200:
#             text = response.text
#             if "Значение</h3>" in text:
#                 start = text.find("Значение</h3>")
#                 end = text.find("</ol>", start)
#                 snippet = text[start:end]
#                 clean = re.sub('<[^<]+?>', '', snippet).strip()
#                 lines = [l.strip() for l in clean.split('\n') if len(l.strip()) > 5]
#                 # Return the first real definition found
#                 for l in lines:
#                     if "Значение" not in l: return l
#     except:
#         pass
#     return None
def expand_candidates_with_parents(candidates_batch, df_relations, df_search):
    id_to_name = df_search.set_index('id')['name'].to_dict()
    batch_results = []
    
    for candidates in candidates_batch:
        parent_counts = {}
        unique_map = {}
        
        for sid, name in candidates:
            unique_map[sid] = name
            parents = df_relations[df_relations['child_id'] == sid]
            for pid in parents['parent_id'].values:
                if pid in id_to_name:
                    parent_counts[pid] = parent_counts.get(pid, 0) + 1
                    unique_map[pid] = id_to_name[pid]
        
        ranked_ids = sorted(unique_map.keys(), key=lambda x: parent_counts.get(x, 0), reverse=True)
        final_list = [(pid, unique_map[pid]) for pid in ranked_ids]
        batch_results.append(final_list[:50])
        
    return batch_results

def run_sanity_check(df_test, judge, retriever, reranker, df_search, df_relations):
    print("\n" + "="*60)
    print("SANITY CHECK")
    
    sample = df_test.head(3)
    orphans = sample['text'].tolist()
    contexts = sample['context'].tolist()
    
    print("Generating Definitions...")
    defs = judge.generate_definitions_batch(orphans, contexts)
    
    print("Retrieving Candidates...")
    queries = [f"query: {d}" for d in defs]
    cands_wide = retriever.get_candidates_batch(queries, k=50)
    
    print("Expanding Graph...")
    cands_expanded = expand_candidates_with_parents(cands_wide, df_relations, df_search)

    print("Reranking...")
    cands_refined = reranker.rerank_batch(defs, cands_expanded, top_k=5)
    
    print("Selecting Best...")
    best_ids = judge.select_best_candidate_batch(orphans, cands_refined)
    
    for i, orphan in enumerate(orphans):
        best_id = best_ids[i]
        try:
            best_name = df_search[df_search['id'] == best_id].iloc[0]['name']
        except:
            best_name = "Unknown ID"
            
        print(f"\nWORD: '{orphan}'")
        print(f"Context: {str(contexts[i])[:60]}...")
        print(f"Def: {defs[i]}")
        print(f"Winner: {best_name} ({best_id})")
        print(f"Top Options: {[c[1] for c in cands_refined[i]]}")

    print("\n" + "="*60)
    try:
        input("Press ENTER to continue...")
    except EOFError:
        time.sleep(5)

def main():
    args = get_args()
    
    with TelegramLogger():
        print(f"\nCONFIG: {args.mode.upper()} | {args.phase.upper()}")

        zip_full_path = os.path.join(args.data_dir, args.zip_name)
        test_filename = f"{args.mode}_{args.phase}.tsv"
        test_folder = f"{args.phase}_test"
        test_file_path = os.path.join(args.data_dir, test_folder, test_filename)

        if not os.path.exists(zip_full_path):
            raise FileNotFoundError(f"ZIP not found: {zip_full_path}")
        if not os.path.exists(test_file_path):
            raise FileNotFoundError(f"Test file not found: {test_file_path}")

        xml_filter = ".N.xml" if args.mode == "nouns" else ".V.xml"
        print(f"Loading Graph with filter: {xml_filter}")
        
        df_search, df_relations = load_graph_from_zip(zip_full_path, xml_filter=xml_filter)
        
        target_suffix = "N" if args.mode == "nouns" else "V"
        if not df_search.iloc[0]['id'].endswith(target_suffix):
            print(f"Filtering database for {target_suffix}...")
            df_search = df_search[df_search['id'].str.endswith(target_suffix)].copy()

        if len(df_search) == 0:
            raise ValueError("Database is empty after filtering!")

        print(f"Loading Test File: {test_file_path}")
        
        try:
            df_test = pd.read_csv(test_file_path, sep='\t', header=None)
            if len(df_test.columns) == 1:
                df_test.columns = ['text']
                df_test['context'] = ""
            elif len(df_test.columns) >= 2:
                df_test = df_test.iloc[:, :2]
                df_test.columns = ['text', 'context']
        except Exception as e:
            raise ValueError(f"Failed to read test file: {e}")

        df_test['context'] = df_test['context'].fillna("").astype(str)
        
        print("Loading Models...")
        judge = QwenJudge() 
        retriever = EmbeddingRetriever(df_search) 
        reranker = CrossEncoderReranker() 

        if not args.skip_check:
            run_sanity_check(df_test, judge, retriever, reranker, df_search, df_relations)

        submission = {}
        all_orphans = df_test['text'].tolist()
        all_contexts = df_test['context'].tolist()
        
        print(f"Starting Inference on {len(all_orphans)} items...")
        
        for i in tqdm(range(0, len(all_orphans), args.batch_size)):
            batch_orphans = all_orphans[i : i + args.batch_size]
            batch_contexts = all_contexts[i : i + args.batch_size]
            
            batch_definitions = judge.generate_definitions_batch(batch_orphans, batch_contexts)
            
            search_queries = [f"query: {d}" for d in batch_definitions]
            candidates_wide = retriever.get_candidates_batch(search_queries, k=args.top_k_retrieve)
            
            candidates_expanded = expand_candidates_with_parents(candidates_wide, df_relations, df_search)
            
            candidates_refined = reranker.rerank_batch(batch_definitions, candidates_expanded, top_k=args.top_k_rerank)
            
            best_ids = judge.select_best_candidate_batch(batch_orphans, candidates_refined)
            
            for orphan, best_id, candidates in zip(batch_orphans, best_ids, candidates_refined):
                ancestors = get_ancestors(best_id, df_relations)
                final_list = [best_id] + ancestors
                
                for cid, _ in candidates:
                    if cid not in final_list: final_list.append(cid)
                    if len(final_list) >= 10: break
                
                submission[orphan] = final_list[:10]

            if i > 0 and i % (args.batch_size * 50) == 0:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_{args.mode}.json")
                with open(ckpt_path, 'w', encoding='utf-8') as f:
                    json.dump(submission, f, ensure_ascii=False)

        output_filename = f"submission_{args.mode}_{args.phase}.json"
        save_path = os.path.join(args.output_dir, output_filename)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(submission, f, ensure_ascii=False, indent=4)
            
        print(f"Finished! Saved to: {save_path}")

if __name__ == "__main__":
    main()
