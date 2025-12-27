import torch
import torch.nn as nn
from transformers import PreTrainedModel
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup
from configs.base_config import ClaraConfig
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.data_utils import MamsClaraDataset
from src.logger import LocalLogger


class FastClaraModel(nn.Module):
    def __init__(self, base_model, tokenizer, mem_token_ids, alpha=1.0, beta=0.1, gamma=0.1):
        """
        :param alpha: Начальный вес Reconstruction (ставим высоким)
        :param beta: Начальный вес ABSA (ставим низким)
        :param gamma: Вес MSE (обычно константа)
        """
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        self.mem_token_ids = mem_token_ids
        self.num_mem_tokens = len(mem_token_ids)
        
        # Текущие веса лоссов
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def set_weights(self, alpha=None, beta=None, gamma=None):
        """Метод для динамического обновления весов лосса во время обучения"""
        if alpha is not None: self.alpha = alpha
        if beta is not None: self.beta = beta
        if gamma is not None: self.gamma = gamma

    def calculate_mse_alignment(self, input_embeddings, memory_states, enc_input_ids, enc_mask):
        """
        Реализация Equation 2.2 из статьи CLaRa.
        Выравнивает среднее латентное состояние памяти со средним эмбеддингом исходного текста.
        """
        # 1. Создаем маску только для текстовых токенов (исключая [M] и Padding)
        text_only_mask = enc_mask.clone().bool()
        for m_id in self.mem_token_ids:
            text_only_mask &= (enc_input_ids != m_id)
        
        # 2. Средний эмбеддинг текста (берем из первого слоя - входные эмбеддинги)
        # [B, L_enc, D] * [B, L_enc, 1] -> [B, D]
        text_sum = (input_embeddings * text_only_mask.unsqueeze(-1)).sum(dim=1)
        text_count = text_only_mask.sum(dim=1, keepdim=True).clamp(min=1)
        avg_text_embed = text_sum / text_count
        
        # 3. Средний вектор памяти (выход последнего слоя энкодера)
        # memory_states имеет размер [B, num_mem_tokens, D]
        avg_mem_embed = memory_states.mean(dim=1)
        
        # 4. MSE Loss
        return nn.functional.mse_loss(avg_mem_embed, avg_text_embed)

    def get_encoder_memory_states(self, enc_input_ids, enc_mask):
        """
        Проход энкодера для получения сжатых векторов.
        """
        outputs = self.model.model( # Обращаемся к базовому Transformer
            input_ids=enc_input_ids,
            attention_mask=enc_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Hidden states последнего слоя
        last_hidden_state = outputs.hidden_states[-1]
        
        # Находим позиции [M] токенов в enc_input_ids
        # Твой формат: [Text] + [M1..Mk] + [Pad]. Токены памяти всегда ПЕРЕД паддингом.
        # Создаем маску для извлечения
        mem_mask = torch.zeros_like(enc_input_ids, dtype=torch.bool)
        for m_id in self.mem_token_ids:
            mem_mask |= (enc_input_ids == m_id)
            
        # Извлекаем векторы памяти: [B, num_mem_tokens, D]
        memory_states = last_hidden_state[mem_mask].view(
            enc_input_ids.size(0), self.num_mem_tokens, -1
        )
        
        # Также возвращаем входные эмбеддинги для MSE (hidden_states[0])
        return memory_states, outputs.hidden_states[0]

    def forward(self, enc_input_ids, enc_mask, dec_input_ids, dec_mask, labels, task):
        # --- 1. ENCODING ---
        memory_states, input_embeddings = self.get_encoder_memory_states(enc_input_ids, enc_mask)
        
        # --- 2. PREPARE DECODER INPUTS ---
        # Вместо замены на месте (in-place), мы делаем конкатенацию
        
        # Получаем эмбеддинги только для той части декодера, которая ИДЕТ ПОСЛЕ [M] токенов
        # Твой формат: [M1..Mk] + [Prompt + Target]
        # Мы берем эмбеддинги всего, кроме первых k токенов
        rest_of_dec_ids = dec_input_ids[:, self.num_mem_tokens:]
        rest_of_dec_embeds = self.model.get_input_embeddings()(rest_of_dec_ids)
        
        # Теперь склеиваем: [Memory States] + [Rest of Embeds]
        # Градиент теперь будет течь плавно через конкатенацию в оба источника
        inputs_embeds = torch.cat([memory_states, rest_of_dec_embeds], dim=1)
        
        # --- 3. DECODING (Generation) ---
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_mask,
            return_dict=True
        )

        logits = outputs.logits

        # --- 4. MULTI-TASK LOSS CALCULATION ---
        # Вычисляем CE Loss для каждой строки батча без усреднения
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        
        # Традиционный сдвиг для Causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Плоский лосс: [B * (L_dec-1)]
        flat_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # Возвращаем к размеру [B, L_dec-1]
        token_losses = flat_losses.view(labels.size(0), -1)
        
        # Средний лосс на строку (только по тем токенам, где label != -100)
        sample_losses = token_losses.sum(dim=1) / (shift_labels != -100).sum(dim=1).clamp(min=1)

        # Разделяем на Rec и ABSA (ext/reason)
        rec_mask = torch.tensor([t == "rec" for t in task], device=labels.device)
        absa_mask = torch.tensor([t in ["ext", "reason"] for t in task], device=labels.device)

        l_rec = sample_losses[rec_mask].mean() if rec_mask.any() else torch.tensor(0.0, device=labels.device, requires_grad=True)
        l_absa = sample_losses[absa_mask].mean() if absa_mask.any() else torch.tensor(0.0, device=labels.device, requires_grad=True)

        # --- 5. MSE ALIGNMENT ---
        mse_loss = self.calculate_mse_alignment(input_embeddings, memory_states, enc_input_ids, enc_mask)

        # --- 6. FINAL AGGREGATION ---
        # total_loss = (self.alpha * l_rec) + (self.beta * l_absa) + (self.gamma * mse_loss)
        total_loss = (self.alpha * l_rec) + (self.beta * l_absa) + (self.gamma * mse_loss)

        return {
            "loss": total_loss,
            "l_rec": l_rec,
            "l_absa": l_absa,
            "mse_loss": mse_loss,
            "logits": logits
        }

def setup_model_and_tokenizer(config: ClaraConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    
    # 1. Добавляем токены памяти
    mem_tokens = [f"[M{i}]" for i in range(config.num_mem_tokens)]
    tokenizer.add_special_tokens({"additional_special_tokens": mem_tokens})
    mem_token_ids = tokenizer.convert_tokens_to_ids(mem_tokens)
    

    # 2. Загружаем модель (рекомендуем BF16 для точности градиентов)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False
    )
    
    # 3. Расширяем слой эмбеддингов
    base_model.resize_token_embeddings(len(tokenizer))
        
    # 4. Настройка LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)
    
    return model, tokenizer, mem_token_ids

def run_experiment():
    config = ClaraConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Инициализация модели
    peft_model, tokenizer, mem_token_ids = setup_model_and_tokenizer(config)
    clara_model = FastClaraModel(peft_model, tokenizer, mem_token_ids)
    
    # 2. Датасеты и Лоадеры
    # Предполагаем, что класс ClaraDataset уже определен вами выше
    train_ds = MamsClaraDataset(config.train_xml, tokenizer, num_mem_tokens=config.num_mem_tokens)
    val_ds = MamsClaraDataset(config.val_xml, tokenizer, num_mem_tokens=config.num_mem_tokens)
    
    # Используем ваш WeightedRandomSampler для трейна
    weights = [config.task_weights[s['task']] for s in train_ds.samples]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_ds))
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config.val_batch_size)
    
    # 3. Loger
    logger = LocalLogger(log_dir="research_logs", run_name=config.run_name)

    # 4. Оптимизация
    # optimizer = torch.optim.AdamW(clara_model.parameters(), lr=config.lr)
    optimizer = torch.optim.AdamW(
        clara_model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay,
        eps=1e-8 # Повышенная точность
        )   

    total_steps = len(train_loader) * config.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps*0.1), total_steps)
    
    # 5. Main Loop
    global_step = 0
    for epoch in range(config.epochs):
        clara_model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Curriculum Learning: Обновление альфа и бета
            prog = global_step / total_steps
            a = config.alpha_start + (config.alpha_end - config.alpha_start) * prog
            b = config.beta_start + (config.beta_end - config.beta_start) * prog
            clara_model.set_weights(alpha=a, beta=b)
            
            outputs = clara_model(**batch)
            loss = outputs["loss"]
            loss.backward()
            
            if (global_step + 1) % config.grad_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Логируем метрики обучения
            logger.log({
                "train/total_loss": outputs["loss"].item(),
                "train/l_absa": outputs["l_absa"].item(),
                "train/alpha": a, "train/beta": b,
                "train/lr": lr_scheduler.get_last_lr()[0]
            }, step=global_step)
            
            global_step += 1
            pbar.set_postfix({"L": loss.item()})
            
        # --- ВАЛИДАЦИЯ В КОНЦЕ ЭПОХИ ---
        print(f"\nRunning validation for epoch {epoch}...")
        val_metrics = validate_clara(clara_model, val_loader, device)
        logger.log(val_metrics, step=global_step)
        print(f"Val Loss: {val_metrics['val/total_loss']:.4f}, Val ABSA: {val_metrics['val/l_absa']:.4f}")
        
        # Сохранение лучшей модели
        torch.save(peft_model.state_dict(), f"checkpoints/clara_last.pt")

    logger.save_final()

@torch.no_grad()
def validate_clara(clara_model, val_loader, device):
    clara_model.eval()
    val_metrics = {"total_loss": 0, "l_rec": 0, "l_absa": 0, "mse_loss": 0}
    num_batches = len(val_loader)
    
    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = clara_model(**batch)
        
        val_metrics["total_loss"] += outputs["loss"].item()
        val_metrics["l_rec"] += outputs["l_rec"].item()
        val_metrics["l_absa"] += outputs["l_absa"].item()
        val_metrics["mse_loss"] += outputs["mse_loss"].item()
        
    # Усредняем
    avg_metrics = {f"val/{k}": v / num_batches for k, v in val_metrics.items()}
    clara_model.train()
    return avg_metrics



if __name__ == "__main__":
    run_experiment()
