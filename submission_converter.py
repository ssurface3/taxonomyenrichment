import json
import zipfile
import os

def convert_and_zip(json_path, tsv_name='nouns.tsv', zip_name='submission.zip'):
    print(f"🔄 Converting {json_path}...")
    
    # 1. Load Data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 2. Write TSV Manually (No CSV library to avoid quoting issues)
    with open(tsv_name, 'w', encoding='utf-8') as f:
        for word, ids in data.items():
            top_10 = ids[:10]
            
            # Create the JSON string: ["123-N", "456-N"]
            ids_string = json.dumps(top_10, ensure_ascii=False)
            
            # STRICT FORMAT: Word + Tab + String + Newline
            # We explicitly write the \t character
            line = f"{word}\t{ids_string}\n"
            f.write(line)
            
    print(f"✅ Created {tsv_name}")

    # 3. Sanity Check (Read the first line back)
    print("\n🔍 SANITY CHECK (First Line):")
    with open(tsv_name, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        parts = first_line.split('\t')
        print(f"   Raw Line: {first_line}")
        print(f"   Split by Tab: {parts}")
        
        if len(parts) != 2:
            print("❌ ERROR: File is NOT 2 columns! Check your code.")
        else:
            print(f"   Column 1 (Word): {parts[0]}")
            print(f"   Column 2 (IDs):  {parts[1]}")
            print("✅ SUCCESS: File format is correct.")

    # 4. Zip it
    print(f"\n📦 Zipping into {zip_name}...")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(tsv_name)
    print(f"🚀 READY! Download '{zip_name}' and upload to Codalab.")

if __name__ == "__main__":
    # Ensure you are pointing to the right file!
    convert_and_zip('submission.json', 'nouns.tsv', 'submission.zip')