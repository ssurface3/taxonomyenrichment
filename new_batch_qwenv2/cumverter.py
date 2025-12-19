import json
import zipfile
import argparse
import os

def convert_and_zip(json_path, tsv_name, zip_name):
    print(f"🔄 Converting {json_path}...")
    
    if not os.path.exists(json_path):
        print(f"❌ Error: {json_path} not found.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    with open(tsv_name, 'w', encoding='utf-8') as f:
        for word, ids in data.items():
            top_10 = ids[:10]
            ids_string = json.dumps(top_10, ensure_ascii=False)
            f.write(f"{word}\t{ids_string}\n")
            
    print(f"Created {tsv_name}")

    print(f"📦 Zipping into {zip_name}...")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(tsv_name)
    print(f"🚀 READY! Download '{zip_name}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output", type=str, default="verbs.tsv", help="Internal TSV name (verbs.tsv/nouns.tsv)")
    args = parser.parse_args()
    
    # Auto-generate zip name based on input
    zip_filename = args.input.replace(".json", ".zip")
    
    convert_and_zip(args.input, args.output, zip_filename)