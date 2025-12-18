from sentence_transformers import SentenceTransformer, util
import torch

class EmbeddingRetriever:
    def __init__(self, df_search, model_name='intfloat/multilingual-e5-large'):
        self.df_search = df_search
        self.ids = df_search['id'].tolist()
        
        
        self.texts = [f"passage: {t}" for t in df_search['full_text'].tolist()]
        
        print(f"Loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        
        
        self.corpus_embeddings = self.model.encode(
            self.texts, 
            convert_to_tensor=True, 
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=16 l
        )

    def get_candidates_batch(self, queries, k=50):
        """
        queries: List of strings ['run context...', 'eat context...']
        """
       
        formatted_queries = [f"query: {q}" for q in queries]
        
       
        query_embeddings = self.model.encode(
            formatted_queries, 
            convert_to_tensor=True, 
            normalize_embeddings=True
        )
        
       
        hits_batch = util.semantic_search(query_embeddings, self.corpus_embeddings, top_k=k)
        
        batch_results = []
        for hits in hits_batch:
            candidates = []
            for hit in hits:
                idx = hit['corpus_id']
                sid = self.ids[idx]
                name = self.df_search.iloc[idx]['name']
                candidates.append((sid, name))
            batch_results.append(candidates)
            
        return batch_results