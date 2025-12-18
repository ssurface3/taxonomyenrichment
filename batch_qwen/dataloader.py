import pandas as pd
import zipfile
import xml.etree.ElementTree as ET

def load_graph_from_zip(zip_path):
    synset_map = {}
    senses_map = []
    relations = []

    with zipfile.ZipFile(zip_path, 'r') as z:
        for filename in z.namelist():
            if 'synsets.N.xml' in filename:
                with z.open(filename) as f:
                    root = ET.parse(f).getroot()
                    for synset in root.findall('synset'):
                        sid = synset.get('id')
                        name = synset.get('ruthes_name')
                        synset_map[sid] = name
                        for sense in synset.findall('sense'):
                            senses_map.append({'id': sid, 'text': sense.text})

            elif 'synset_relations.N.xml' in filename:
                with z.open(filename) as f:
                    root = ET.parse(f).getroot()
                    for rel in root.findall('relation'):
                        if rel.get('name') in ('hypernym', 'instance hypernym'):
                            relations.append({
                                'parent_id': rel.get('parent_id'),
                                'child_id': rel.get('child_id')
                            })

    df_relations = pd.DataFrame(relations)
    
    df_senses = pd.DataFrame(senses_map)
    df_search = df_senses.groupby('id')['text'].apply(lambda x: ", ".join(x)).reset_index()
    df_search.rename(columns={'text': 'lemmas'}, inplace=True)
    df_search['name'] = df_search['id'].map(synset_map)
    df_search['full_text'] = df_search['name'] + ": " + df_search['lemmas']
    
    return df_search, df_relations