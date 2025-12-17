import json
import pandas as pd
from dataloader import load_graph_from_zip

def diagnose_model_output(json_path, zip_path='/kaggle/working/taxonomy-enrichment/data/ruwordnet.zip'):
    print(f"🕵️‍♂️ Diagnosing {json_path}...")
    
    # 1. Load the Truth (Graph)
    df_search, _ = load_graph_from_zip(zip_path)
    valid_ids = set(df_search['id'].values)
    id_to_name = df_search.set_index('id')['name'].to_dict()
    
    # 2. Load Your Submission
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"📊 Total Predictions: {len(data)}")
    
    # 3. Check Public vs Private Lengths
    try:
        public_len = len(pd.read_csv('data/public_test/nouns_public.tsv', sep='\t', header=None))
        private_len = len(pd.read_csv('data/private_test/nouns_private.tsv', sep='\t', header=None))
        print(f"   - Public Test Set Size:  {public_len}")
        print(f"   - Private Test Set Size: {private_len}")
        
        if len(data) == private_len:
            print("🚨 WARNING: You generated predictions for the PRIVATE set.")
            print("   If you submit this to the 'Practice' phase (Public), you will get 0.0 score.")
        elif len(data) == public_len:
            print("✅ Size matches Public set.")
    except:
        print("   (Could not check file lengths automatically)")

    # 4. Check ID Validity
    print("\n🔍 Checking ID Quality (First 5)...")
    invalid_count = 0
    
    for word, ids in list(data.items())[:5]:
        print(f"\nOrphan: {word}")
        decoded_path = []
        for pid in ids[:3]: # Check top 3
            if pid in valid_ids:
                decoded_path.append(f"{id_to_name[pid]} ({pid})")
            else:
                decoded_path.append(f"❌ INVALID_ID ({pid})")
                invalid_count += 1
                
        print(f"   -> Model Predicted: {', '.join(decoded_path)}")

    if invalid_count > 0:
        print("\n❌ CRITICAL: Your model is inventing fake IDs!")
    else:
        print("\n✅ IDs look valid. The model is predicting real concepts.")

# === RUN ===
diagnose_model_output('submission.json')
