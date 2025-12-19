import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
#Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24
#unsloth/Qwen2.5-14B-Instruct-bnb-4bit
class QwenJudge:
    def __init__(self, model_id="Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        self.tokenizer.padding_side = "left" 
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
    def generate_definitions_batch(self, orphans, contexts):
            prompts = []
            for orphan, ctx in zip(orphans, contexts):
                
                if not ctx or str(ctx) == "nan" or str(ctx) == "":
                    prompt = (
                        f"Дай точное лингвистическое определение слову '{orphan}'. "
                        f"Не путай с похожими словами (паронимами). "
                        f"Если слово имеет несколько значений, укажи основное. "
                        f"Ответ на русском."
                        f"Без вводных фраз типа 'это глагол', 'означает'. "
                        f"Сразу суть."
                    )
                else:
                    prompt = f"Дай определение слову '{orphan}' в контексте: '{ctx}'. Отвечай на русском."

                messages = [
                    {"role": "system", "content": "Ты толковый словарь русского языка."},
                    {"role": "user", "content": prompt}
                ]
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompts.append(text)
                
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=60,
                    do_sample=False
                )
                
            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_len:]
            responses = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            return responses
    def generate_hypernym_batch(self, orphans, contexts):
        prompts = []
        for orphan, ctx in zip(orphans, contexts):
            # Explicitly ask for the broad category
            prompt = f"Identify the hypernym (broad category) for the Russian word '{orphan}'. Context: '{ctx}'. Output ONLY the hypernym word."
            
            messages = [
                {"role": "system", "content": "You are a linguist."},
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)
            
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=20, # Short answer (just the word)
                do_sample=False
            )
            
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        responses = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        return responses
    def select_best_candidate_batch(self, orphans, candidates_batch):
        prompts = []
        valid_ids_batch = []

        
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

        
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=20,
                do_sample=False
            )

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
                final_ids.append(valid_ids[0]) 
                
        return final_ids