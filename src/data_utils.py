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
        """
        Parse MAMS/SemEval-style XML into a list of dicts:
        [
        {
            "text": "...",
            "aspects": [
                {"term": "...", "polarity": "...", "from": 0, "to": 7},
                ...
            ]
        },
        ...
        ]

        - skip aspectTerm with polarity="conflict" (if present)
        - skip sentences that end up with no valid aspect terms
        """
        tree = etree.parse(path)
        root = tree.getroot()

        data = []
        # <sentences> -> many <sentence>
        for sent_el in root.findall("sentence"):
            text_el = sent_el.find("text")
            if text_el is None or text_el.text is None:
                continue

            text = text_el.text.strip()
            if not text:
                continue

            aspects = []
            aspect_terms_el = sent_el.find("aspectTerms")
            if aspect_terms_el is not None:
                for at_el in aspect_terms_el.findall("aspectTerm"):
                    term = at_el.get("term")
                    polarity = at_el.get("polarity")
                    frm = at_el.get("from")
                    to = at_el.get("to")

                    if term is None or polarity is None:
                        continue

                    term = term.strip()
                    polarity = polarity.strip().lower()

                    # labels: positive/negative/neutral
                    # conflict -> skip 
                    if polarity not in {"positive", "negative", "neutral"}:
                        continue

                    # from/to can be missing or non-int in some files 
                    try:
                        frm_i = int(frm) if frm is not None else -1
                        to_i = int(to) if to is not None else -1
                    except ValueError:
                        frm_i, to_i = -1, -1

                    aspects.append({
                        "term": term,
                        "polarity": polarity,
                        "from": frm_i,
                        "to": to_i
                    })

            # only keep sentences with at least one valid aspect
            if len(aspects) == 0:
                continue

            data.append({"text": text, "aspects": aspects})

        return data


    def __len__(self):
        return len(self.samples)

    
    def process_step(self, sample):
        # --- ШАГ 1: ЭНКОДЕР (без изменений) ---
        enc_text = sample['text'].strip() + " " + self.mem_tokens_str.strip()
        enc_res = self.tokenizer(enc_text, max_length=self.max_enc_len, 
                                padding='max_length', truncation=True, return_tensors="pt")

        # --- ШАГ 2: ПОДГОТОВКА ДЕКОДЕРА (Ручная сборка IDs) ---
        prompt_part = f"{self.mem_tokens_str} {sample['prompt']}".strip() + " "
        target_part = sample['target'].strip()

        # 1. Получаем ID промпта
        prompt_ids = self.tokenizer.encode(prompt_part, add_special_tokens=False)
        
        # 2. Получаем ID таргета и добавляем EOS
        target_ids = self.tokenizer.encode(target_part, add_special_tokens=False)
        target_ids.append(self.tokenizer.eos_token_id)
        
        # 3. Собираем full_ids и labels синхронно
        # В labels промпт закрываем -100
        full_ids = prompt_ids + target_ids
        labels_list = ([-100] * len(prompt_ids)) + target_ids
        
        # 4. Обрезаем по max_dec_len, если текст слишком длинный
        full_ids = full_ids[:self.max_dec_len]
        labels_list = labels_list[:self.max_dec_len]
        
        # 5. Ручной паддинг до фиксированной длины
        actual_len = len(full_ids)
        padding_len = self.max_dec_len - actual_len
        
        if padding_len > 0:
            full_ids += [self.tokenizer.pad_token_id] * padding_len
            # Паддинг в labels ВСЕГДА -100
            labels_list += [-100] * padding_len
        
        dec_input_ids = torch.tensor(full_ids)
        labels = torch.tensor(labels_list)

        # --- ШАГ 3: СОЗДАНИЕ МАСКИ ВНИМАНИЯ ---
        dec_mask = torch.zeros(self.max_dec_len, dtype=torch.bool)
        dec_mask[:actual_len] = True

        return {
            "enc_input_ids": enc_res['input_ids'][0],
            "enc_mask": enc_res['attention_mask'][0],
            "dec_input_ids": dec_input_ids,
            "dec_mask": dec_mask,
            "labels": labels,
            "task": sample['task']
        }

    def __getitem__(self, idx):
        return self.process_step(self.samples[idx])
    


def debug_clara_batch(batch, tokenizer, num_samples=3):
    """
    Детальная диагностика батча: декодирует тензоры обратно в текст, 
    проверяет маски и правильность расстановки -100 в labels.
    """
    print("\n" + "="*80)
    print(f"DIAGNOSTIC REPORT FOR BATCH (Batch Size: {len(batch['task'])})")
    print("="*80)

    for i in range(min(num_samples, len(batch['task']))):
        task = batch['task'][i]
        print(f"\n--- SAMPLE {i} | TASK: {task.upper()} ---")

        # 1. Диагностика Энкодера (Compression)
        enc_ids = batch['enc_input_ids'][i]
        enc_text = tokenizer.decode(enc_ids, skip_special_tokens=False)
        # Проверяем наличие токенов памяти
        mem_tokens_detected = [t for t in tokenizer.additional_special_tokens if t in enc_text]
        
        print(f"[Encoder Input]: {enc_text[:100]}... {enc_text[-50:]}")
        print(f"  > Memory Tokens Detected: {mem_tokens_detected}")
        print(f"  > Padding tokens count: {(enc_ids == tokenizer.pad_token_id).sum().item()}")

        # 2. Диагностика Декодера (Reasoning/Generation)
        dec_ids = batch['dec_input_ids'][i]
        dec_text = tokenizer.decode(dec_ids, skip_special_tokens=False)
        print(f"[Decoder Input]: {dec_text}")

        # 3. Диагностика Labels (ОБУЧЕНИЕ)
        labels = batch['labels'][i]
        
        # Извлекаем то, на чем реально учится модель (где не -100)
        active_label_ids = labels[labels != -100]
        active_label_text = tokenizer.decode(active_label_ids, skip_special_tokens=False) if len(active_label_ids) > 0 else "EMPTY!"
        
        print(f"[Target Answer]: {active_label_text}")

        # 4. Логические проверки (Sanity Checks)
        # Проверка 1: labels должны совпадать с концом dec_input_ids
        actual_answer_ids = dec_ids[labels != -100]
        if not torch.equal(actual_answer_ids, active_label_ids):
            print("  [!] ERROR: Labels do not align with dec_input_ids!")
        
        # Проверка 2: токены памяти [M] НЕ должны быть в labels (там должно быть -100)
        # Мы предполагаем, что [M] токены - это спецтокены с большими ID
        for m_token in tokenizer.additional_special_tokens:
            m_id = tokenizer.convert_tokens_to_ids(m_token)
            if m_id in labels:
                print(f"  [!] WARNING: Memory token {m_token} leaked into Labels! (Model will try to predict it)")

        # Проверка 3: Attention Mask
        if batch['enc_mask'][i][0] == 0:
            print("  [!] WARNING: Encoder mask starts with 0. Check your padding (should be right-padded).")

    print("\n" + "="*80)


def stack_batch(samples):
    return {
        "enc_input_ids": torch.stack([s["enc_input_ids"] for s in samples]),
        "enc_mask": torch.stack([s["enc_mask"] for s in samples]),
        "dec_input_ids": torch.stack([s["dec_input_ids"] for s in samples]),
        "dec_mask": torch.stack([s["dec_mask"] for s in samples]),
        "labels": torch.stack([s["labels"] for s in samples]),
        "task": [s["task"] for s in samples],
    }