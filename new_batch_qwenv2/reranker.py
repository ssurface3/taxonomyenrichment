from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class CrossEncoderReranker:
    def __init__(self, model_name='BAAI/bge-reranker-v2-m3'):
        print(f"Loading Reranker: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.model.to("cuda")

    def rerank_batch(self, query_batch, candidates_batch, top_k=5):
        """
        query_batch: List of strings (Definitions or Words)
        candidates_batch: List of Lists of (id, name) tuples
        """
        final_results = []

        for query, candidates in zip(query_batch, candidates_batch):
            if not candidates:
                final_results.append([])
                continue

            # Prepare pairs: [ [Query, Cand1], [Query, Cand2] ... ]
            pairs = [[query, cand_name] for _, cand_name in candidates]
            
            with torch.no_grad():
                inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to("cuda")
                scores = self.model(**inputs, return_dict=True).logits.view(-1,).float()
            
            scores = scores.cpu().numpy()
            combined = list(zip(candidates, scores))
            ranked = sorted(combined, key=lambda x: x[1], reverse=True)
            top_candidates = [item[0] for item in ranked[:top_k]]
            final_results.append(top_candidates)
            
        return final_results