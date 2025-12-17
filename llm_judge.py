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
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )

    def select_best_candidate(self, orphan, candidates):
        options_text = ""
        valid_ids = []
        for i, (sid, name) in enumerate(candidates):
            options_text += f"{i+1}. {name} (ID: {sid})\n"
            valid_ids.append(sid)

        messages = [
            {"role": "system", "content": "Select the most specific hypernym (parent) ID for the given word."},
            {"role": "user", "content": f"Word: {orphan}\nOptions:\n{options_text}\nAnswer with ID only."}
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(inputs.input_ids, max_new_tokens=20, do_sample=False)
            
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        for vid in valid_ids:
            if vid in response:
                return vid
        
        return valid_ids[0]