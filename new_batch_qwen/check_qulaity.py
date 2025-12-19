import pandas as pd
import argparse
import os

def check_file_raw(file_path):
    print(f"🔬 INSPECTING: {file_path}")
    
    if not os.path.exists(file_path):
        print("❌ File not found.")
        return

    # 1. Check Raw Text (First 5 lines)
    print("\n[1] Raw Text Preview:")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:5]):
            print(f"   Line {i}: {repr(line)}") # repr shows hidden characters like \t or \n

    # 2. Find "АДСОРБИРОВАТЬ" specifically
    print("\n[2] Hunting for 'АДСОРБИРОВАТЬ'...")
    found = False
    for i, line in enumerate(lines):
        if "АДСОРБИРОВАТЬ" in line:
            print(f"   ✅ Found on Line {i}: {repr(line)}")
            parts = line.strip().split('\t')
            print(f"      -> Columns found: {len(parts)}")
            if len(parts) > 1:
                print(f"      -> Context part: '{parts[1]}'")
            else:
                print("      ❌ NO CONTEXT DETECTED (Split by tab failed)")
            found = True
            break
    
    if not found:
        print("   ⚠️ Word not found in this file.")

    # 3. Check Pandas Parsing
    print("\n[3] Pandas Parse Check:")
    try:
        df = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'context'])
        
        # Check for NaNs
        nulls = df[df['context'].isna()]
        print(f"   Total Rows: {len(df)}")
        print(f"   Rows with NaN Context: {len(nulls)}")
        
        if len(nulls) > 0:
            print("   ⚠️ SAMPLE OF BROKEN ROWS:")
            print(nulls.head())
    except Exception as e:
        print(f"   Pandas crashed: {e}")

if __name__ == "__main__":
    # Adjust path if needed (pointing to verbs_public based on your recent work)
    target_file = "taxonomy-enrichment/data/private_test/verbs_private.tsv"
    
    # If using absolute path on Kaggle:
    if not os.path.exists(target_file):
        target_file = "/kaggle/working/taxonomy-enrichment/data/private_test/verbs_private.tsv"
        
    check_file_raw(target_file)