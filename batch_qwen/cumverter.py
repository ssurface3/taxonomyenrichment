import json
import pandas as pd
from dataloader import load_graph_from_zip

def convert_to_baseline_format(json_path, output_name='verbs.tsv', zip_name='submission.zip'):
    print("Converting to Baseline Format (Multi-line)...")
    
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)


    df_search, _ = load_graph_from_zip('/kaggle/working/taxonomy-enrichment/data/ruwordnet.zip')
    id_to_name = df_search.set_index('id')['name'].to_dict()
    
    with open(output_name, 'w', encoding='utf-8') as f:
        for word, ids in data.items():
    
            for pid in ids[:10]:            
                name = id_to_name.get(pid, "UNKNOWN")
                
                # Format: WORD <TAB> ID <TAB> NAME
                f.write(f"{word}\t{pid}\t{name}\n")
                
    print(f"Created {output_name} in Baseline format.")
    
    import zipfile
    print(f" Zipping...")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_name)
    print("🚀 Ready to submit!")

convert_to_baseline_format('/kaggle/working/batch_qwen/sub_nouns.json', 'verbs.tsv', 'submission_verbs_BASELINE.zip')
