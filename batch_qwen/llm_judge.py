import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class QwenJudge:
    def __init__(self, model_id="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # CRITICAL FOR BATCH GENERATION:
        self.tokenizer.padding_side = "left" 
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )

    def select_best_candidate_batch(self, orphans, candidates_batch):
        prompts = []
        valid_ids_batch = []

        # Prepare all prompts
        for orphan, candidates in zip(orphans, candidates_batch):
            options_text = ""
            current_valid_ids = []
            for i, (sid, name) in enumerate(candidates):
                options_text += f"{i+1}. {name} (ID: {sid})\n"
                current_valid_ids.append(sid)
            
            valid_ids_batch.append(current_valid_ids)

            messages = [
                {"role": "system", "content": "Select the most specific hypernym (parent) ID. Output ONLY the ID."},
                {"role": "user", "content": f"Word: {orphan}\nOptions:\n{options_text}\nAnswer with ID only."}
            ]
            
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)

        # Batch Tokenize
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

        # Batch Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=20,
                do_sample=False
            )

        # Decode all
        # We slice [input_len:] to get only new tokens
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        responses = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        final_ids = []
        for response, valid_ids in zip(responses, valid_ids_batch):
            found = False
            for vid in valid_ids:
                if vid in response:
                    final_ids.append(vid)
                    found = True
                    break
            if not found:
                final_ids.append(valid_ids[0]) # Fallback
                
        return final_ids