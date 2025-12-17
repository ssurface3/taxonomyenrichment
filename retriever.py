from sentence_transformers import SentenceTransformer, util
import torch

class EmbeddingRetriever:
    def __init__(self, df_search, model_name='ai-forever/sbert_large_nlu_ru'):
        self.df_search = df_search
        self.ids = df_search['id'].tolist()
        self.texts = df_search['full_text'].tolist()
        
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = self.model.encode(self.texts, convert_to_tensor=True, show_progress_bar=True)

    def get_candidates(self, orphan_word, k=20):
        query_embedding = self.model.encode(orphan_word, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=k)[0]
        
        candidates = []
        for hit in hits:
            idx = hit['corpus_id']
            sid = self.ids[idx]
            name = self.df_search.iloc[idx]['name']
            candidates.append((sid, name))
            
        return candidates