import os
import json
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(
        "/root/czh/DeepSeek-R1/", trust_remote_code=True
    )

res = []
# load dataset from folder
dataset_path = "/root/czh/geogpt_rag_res/"
for filename in os.listdir(dataset_path):
    if filename.startswith("OALQA"):
        continue
    file_path = os.path.join(dataset_path, filename)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": data["question"]}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt = prompt.replace(tokenizer.bos_token, "")

            prompt_token_ids = tokenizer.encode(prompt)
            prompt_len = len(prompt_token_ids)
            
            retokenized_output_len = len(
                tokenizer.encode(data["answer"], add_special_tokens=False)
            )
            retokenized_reasoning_len = len(
                tokenizer.encode(data["reasoning"], add_special_tokens=False)
            )
            res.append(prompt_len + retokenized_output_len + retokenized_reasoning_len)
            

print(np.mean(res), np.median(res), np.max(res), np.std(res), np.percentile(res, 80))
