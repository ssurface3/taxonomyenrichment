import pandas as pd
import numpy as np
from dataloader import load_graph_from_zip
from retriever import EmbeddingRetriever
from llm_judge import QwenJudge
from retriever import EmbeddingRetriever
# MAP Calculation Helper
def calculate_map(predictions, truths):
    """
    predictions: List of list of IDs [['1-N', '2-N'], ...]
    truths: List of sets of valid IDs [{'1-N'}, ...]
    """
    aps = []
    for pred, true_set in zip(predictions, truths):
        relevant_cnt = 0
        precision_sum = 0.0
        
        # Clean truth (remove quotes if any)
        clean_true = {t.replace('"', '').replace("'", "").strip() for t in true_set}
        
        for k, p_id in enumerate(pred):
            if p_id in clean_true:
                relevant_cnt += 1
                precision_at_k = relevant_cnt / (k + 1)
                precision_sum += precision_at_k
        
        if len(clean_true) > 0:
            aps.append(precision_sum / len(clean_true))
        else:
            aps.append(0.0)
            
    return np.mean(aps)

def test_batch_quality():
    print("🔬 INITIALIZING DIAGNOSTIC TEST...")
    
    # 1. Load Graph (Verbs for now based on your last run)
    df_search, df_relations = load_graph_from_zip('/kaggle/working/taxonomy-enrichment/data/ruwordnet.zip')
    
    # 2. Load TRAINING Data (It has answers!)
    # We use 'training_verbs.tsv' or 'training_nouns.tsv' depending on your focus
    # Let's assume Verbs since that was your last issue
    try:
        df_train = pd.read_csv('/kaggle/working/taxonomy-enrichment/data/training_data/training_verbs.tsv', 
                               sep='\t', header=None, names=['text', 'truth'])
    except:
        print("⚠️ Could not find training_verbs.tsv, trying nouns...")
        df_train = pd.read_csv('/kaggle/working/taxonomy-enrichment/data/training_data/training_nouns.tsv', 
                               sep='\t', header=None, names=['text', 'truth'])

    # 3. Initialize Models
    # CHANGE THIS: Use a better embedding model if you can
    retriever = EmbeddingRetriever()
    judge = QwenJudge(model_id="unsloth/Qwen2.5-7B-Instruct-bnb-4bit")

    # 4. Select a small batch (e.g., 5 difficult items)
    batch = df_train.sample(5, random_state=42)
    
    orphans = batch['text'].tolist()
    # We don't have context in training_verbs.tsv usually, so we use the word itself as context
    # If you have contexts for training, load them here
    queries = [f"{o} ." for o in orphans] 
    
    # Parse Truth strings "['123-V', '456-V']" -> {'123-V', '456-V'}
    import ast
    truths = []
    for t in batch['truth']:
        try:
            truths.append(set(ast.literal_eval(t)))
        except:
            # Fallback cleanup
            clean = t.replace('[','').replace(']','').replace('"','').replace("'",'').split(',')
            truths.append({x.strip() for x in clean})

    print(f"\n🧪 Testing on {len(orphans)} samples...")

    # --- STEP A: RETRIEVAL CHECK ---
    candidates_batch = retriever.get_candidates_batch(queries, k=25)
    
    recall_hits = 0
    for i, candidates in enumerate(candidates_batch):
        # specific candidates IDs
        cand_ids = {c[0] for c in candidates}
        # Check if ANY true parent is in the top 25
        intersection = cand_ids.intersection(truths[i])
        
        print(f"\n🔸 Orphan: {orphans[i]}")
        print(f"   True Parents: {truths[i]}")
        
        if len(intersection) > 0:
            print(f"   ✅ Retriever found: {intersection}")
            recall_hits += 1
        else:
            print(f"   ❌ Retriever FAILED. Top 5 guesses: {[c[1] for c in candidates[:5]]}")

    print(f"\n📊 RETRIEVER RECALL: {recall_hits}/{len(orphans)}")
    
    if recall_hits == 0:
        print("🛑 STOP. Your Embedding Model is the problem. Qwen has no chance.")
        return

    # --- STEP B: QWEN CHECK ---
    print("\n🤖 Asking Qwen to Judge...")
    best_ids = judge.select_best_candidate_batch(orphans, candidates_batch)
    
    # Calculate simple accuracy on top 1
    qwen_hits = 0
    final_lists = []
    
    for i, best_id in enumerate(best_ids):
        # Expand Ancestors (MAP Strategy)
        curr = best_id
        chain = []
        for _ in range(5):
            parents = df_relations[df_relations['child_id'] == curr]
            if len(parents) == 0: break
            curr = parents.iloc[0]['parent_id']
            chain.append(curr)
        
        final_list = [best_id] + chain
        final_lists.append(final_list)
        
        # Check if best_id is correct
        if best_id in truths[i]:
            print(f"   ✅ Qwen picked Correctly: {best_id}")
            qwen_hits += 1
        else:
            print(f"   ❌ Qwen picked Wrong: {best_id} (Expected one of {truths[i]})")

    # --- STEP C: CALCULATE MAP ---
    map_score = calculate_map(final_lists, truths)
    print(f"\n📈 ESTIMATED MAP SCORE: {map_score:.4f}")

if __name__ == "__main__":
    test_batch_quality()
