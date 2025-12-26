import torch
from torch.utils.data import Dataset
from lxml import etree
import json

class MamsClaraDataset(Dataset):
    def __init__(self, xml_path, tokenizer, num_mem_tokens=4, max_enc_len=128, max_dec_len=128):
        self.tokenizer = tokenizer
        self.num_mem_tokens = num_mem_tokens
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.mem_tokens_str = " ".join([f"[M{i}]" for i in range(num_mem_tokens)])
        
        # parse xml
        self.raw_data = self._parse_xml(xml_path)
        
        # flattening of tasks (rec, ext, reason)
        self.samples = [] # result items of dataset
        
        for entry in self.raw_data:
            text = entry['text']
            aspects = entry['aspects']
            
            # a: Reconstruction
            self.samples.append({
                "text": text, "task": "rec", 
                "prompt": "Restore text:", "target": text
            })
            
            # b: Extraction
            ext_target = ", ".join([f"{a['term']}: {a['polarity']}" for a in aspects])
            self.samples.append({
                "text": text, "task": "ext", 
                "prompt": "List all aspects:", "target": ext_target
            })
            
            # c: Reasoning
            for aspect in aspects:
                self.samples.append({
                    "text": text, "task": "reason", 
                    "prompt": f"Sentiment of {aspect['term']}?", 
                    "target": aspect['polarity']
                })

    def _parse_xml(self, path):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def _apply_labels_mask(self, input_ids, target_ids):
        """
        create labels, where everything except target is replaced by -100 value
        answer is always at the end of sequence
        """
        labels = torch.full_like(input_ids, -100)
        target_len = len(target_ids)
        
        # find real len (without padding)
        padding_mask = (input_ids != self.tokenizer.pad_token_id)
        actual_len = padding_mask.sum().item()
        
        # fill only last tokens that are target
        # (actual_len - target_len : actual_len)
        labels[actual_len - target_len : actual_len] = input_ids[actual_len - target_len : actual_len]
        return labels

    def process_step(self, sample):
        # --- STEP 1: GET DATA FOR ENCODER ---
        # format: "Text of review [M1] [M2] [M3] [M4]"
        enc_text = f"{sample['text']} {self.mem_tokens_str}"
        enc_res = self.tokenizer(
            enc_text, max_length=self.max_enc_len, 
            padding='max_length', truncation=True, return_tensors="pt"
        )

        # --- STEP 2: GET DATA FOR DECODER (GENERATION) ---
        # format: "[M1] [M2] [M3] [M4] Prompt: Target"
        # model will learn to generate 'target' after 'prompt'
        dec_prompt = f"{self.mem_tokens_str} {sample['prompt']} "
        dec_target = sample['target']
        full_dec_text = dec_prompt + dec_target
        
        dec_res = self.tokenizer(
            full_dec_text, max_length=self.max_dec_len, 
            padding='max_length', truncation=True, return_tensors="pt"
        )
        
        # tokenise target to know its len of mask
        target_ids = self.tokenizer(dec_target, add_special_tokens=False).input_ids
        
        # --- STEP 3: MASK LABELS ---
        labels = self._apply_labels_mask(dec_res['input_ids'][0], target_ids)

        return {
            "enc_input_ids": enc_res['input_ids'][0],
            "enc_mask": enc_res['attention_mask'][0],
            "dec_input_ids": dec_res['input_ids'][0],
            "dec_mask": dec_res['attention_mask'][0],
            "labels": labels,
            "task": sample['task']
        }

    def __getitem__(self, idx):
        return self.process_step(self.samples[idx])