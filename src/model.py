import torch
import torch.nn as nn
from transformers import PreTrainedModel
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup
from configs.base_config import ClaraConfig
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from src.data_utils import MamsClaraDataset
from src.logger import LocalLogger


class FastClaraModel(nn.Module):
    def __init__(self, base_model, tokenizer, mem_token_ids, alpha=1.0, beta=0.1, gamma=0.1):
        """
        :param alpha: Start weight Reconstruction 
        :param beta: Start weight ABSA 
        :param gamma: Weight MSE (constant)
        """
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        self.mem_token_ids = mem_token_ids
        self.num_mem_tokens = len(mem_token_ids)
        
        # Current loss weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def set_weights(self, alpha=None, beta=None, gamma=None):
        """Dynamic loss weight upgrade during training"""
        if alpha is not None: self.alpha = alpha
        if beta is not None: self.beta = beta
        if gamma is not None: self.gamma = gamma

    def calculate_mse_alignment(self, input_embeddings, memory_states, enc_input_ids, enc_mask):
        """
        Equation 2.2 from CLaRa article.
        Aligns average continuous latent memory with average embedding of training text
        """
        # 1. Get mask only for text tokens (excl. [M] and Padding)
        text_only_mask = enc_mask.clone().bool()
        for m_id in self.mem_token_ids:
            text_only_mask &= (enc_input_ids != m_id)
        
        # 2. Average embedding of text (get from the 1st layer – input embeddings)
        # [B, L_enc, D] * [B, L_enc, 1] -> [B, D]
        text_sum = (input_embeddings * text_only_mask.unsqueeze(-1)).sum(dim=1)
        text_count = text_only_mask.sum(dim=1, keepdim=True).clamp(min=1)
        avg_text_embed = text_sum / text_count
        
        # 3. Average embedding of memory (hidden state from last layer of encoder)
        # memory_states имеет размер [B, num_mem_tokens, D]
        avg_mem_embed = memory_states.mean(dim=1)
        
        # 4. MSE Loss
        return nn.functional.mse_loss(avg_mem_embed, avg_text_embed)

    def get_encoder_memory_states(self, enc_input_ids, enc_mask):
        outputs = self.model.model(
            input_ids=enc_input_ids,
            attention_mask=enc_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        last_hidden_state = outputs.hidden_states[-1] # [B, L, D]
        
        # Find memory tokens position for every batch separately
        memory_indices = []
        for i in range(enc_input_ids.size(0)):
            # Find idx tokens [M] in current string i
            row_indices = []
            for m_id in self.mem_token_ids:
                idx = (enc_input_ids[i] == m_id).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    row_indices.append(idx[0].item())
            
            # If tokens are not enough (truncation), raise error 
            if len(row_indices) != self.num_mem_tokens:
                raise RuntimeError(
                    f"Row {i} has {len(row_indices)} memory tokens, expected {self.num_mem_tokens}. "
                    f"Check max_enc_len or text length!"
                )
            memory_indices.append(row_indices)
            
        # Get tensor with memory tokens
        batch_indices = torch.tensor(memory_indices, device=enc_input_ids.device)
        # use gather for safety unsqueeze
        # expand idx to [B, num_mem_tokens, D]
        gather_indices = batch_indices.unsqueeze(-1).expand(-1, -1, last_hidden_state.size(-1))
        memory_states = torch.gather(last_hidden_state, 1, gather_indices)
        
        return memory_states, outputs.hidden_states[0]

    def forward(self, enc_input_ids, enc_mask, dec_input_ids, dec_mask, labels, task):
        # --- 1. ENCODING ---
        memory_states, input_embeddings = self.get_encoder_memory_states(enc_input_ids, enc_mask)
        
        # --- 2. PREPARE DECODER INPUTS ---
        
        # [M1..Mk] + [Prompt + Target]
        # get embeddings except first k tokens
        rest_of_dec_ids = dec_input_ids[:, self.num_mem_tokens:]
        rest_of_dec_embeds = self.model.get_input_embeddings()(rest_of_dec_ids)
        
        # concat: [Memory States] + [Rest of Embeds]
        inputs_embeds = torch.cat([memory_states, rest_of_dec_embeds], dim=1)
        
        # --- 3. DECODING (Generation) ---
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_mask,
            return_dict=True
        )

        logits = outputs.logits

        # --- 4. MULTI-TASK LOSS CALCULATION ---
        # get CE Loss for every batch string without average
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        
        # shift for Causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # flat losses: [B * (L_dec-1)]
        flat_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # back to [B, L_dec-1]
        token_losses = flat_losses.view(labels.size(0), -1)
        
        # average loss for sample from batch (only by tokens where label != -100)
        sample_losses = token_losses.sum(dim=1) / (shift_labels != -100).sum(dim=1).clamp(min=1)

        # divide to Rec and ABSA (ext/reason tasks)
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
    
    # 1. Add memory tokens
    mem_tokens = [f"[M{i}]" for i in range(config.num_mem_tokens)]
    tokenizer.add_special_tokens({"additional_special_tokens": mem_tokens})
    mem_token_ids = tokenizer.convert_tokens_to_ids(mem_tokens)
    

    # 2. Get model (with BF16)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        use_cache=False
    )
    
    # 3. Resize token embeddings as mem tokens added
    base_model.resize_token_embeddings(len(tokenizer))
        
    # 4. LoRA config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    return model, tokenizer, mem_token_ids

def run_experiment():
    config = ClaraConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Model initialise
    peft_model, tokenizer, mem_token_ids = setup_model_and_tokenizer(config)
    
    # we don't freeze embeddings! because we need to upgrade [M] mem tokens
    peft_model.get_input_embeddings().weight.requires_grad = True
    
    clara_model = FastClaraModel(peft_model, tokenizer, mem_token_ids)
    
    clara_model.to(device)
    
    # 2. Datasets and dataloaders
    train_ds = MamsClaraDataset(
        config.train_xml, tokenizer, 
        num_mem_tokens=config.num_mem_tokens, 
        max_enc_len=config.max_enc_len, 
        max_dec_len=config.max_dec_len
    )
    val_ds = MamsClaraDataset(
        config.val_xml, tokenizer, 
        num_mem_tokens=config.num_mem_tokens, 
        max_enc_len=config.max_enc_len, 
        max_dec_len=config.max_dec_len
    )
    
    # validation of task weights for sampler
    weights = [config.task_weights.get(s['task'], 1.0) for s in train_ds.samples]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_ds))
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        sampler=sampler, 
        num_workers=8,
        pin_memory=True # Ускоряет передачу данных на GPU
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, 
        batch_size=config.val_batch_size, 
        num_workers=4
    )
    
    # 3. Logger
    logger = LocalLogger(log_dir="research_logs", run_name=config.run_name)

    # 4. Optimiser config 
    # get trainable params (LoRA + Embeddings)
    trainable_params = [p for p in clara_model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(
        trainable_params, 
        lr=config.lr, 
        weight_decay=config.weight_decay,
        eps=1e-8
    )   

    # STEP COUNTING
    # (updates) = (all epoch batch * epochs)
    num_update_steps_per_epoch = len(train_loader) // config.grad_accumulation_steps
    # take into account probable left batches in epoch end
    if len(train_loader) % config.grad_accumulation_steps != 0:
        num_update_steps_per_epoch += 1
        
    total_optimization_steps = num_update_steps_per_epoch * config.epochs
    warmup_steps = int(total_optimization_steps * 0.1)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_optimization_steps
    )
    
    # save total_steps to config in training (Curriculum Learning CL)
    # for alpha/beta we need batches, for LR scheduler – optimiser steps
    total_batch_steps = len(train_loader) * config.epochs
    
    # 5. Main Loop
    global_step = 0
    total_batches = len(train_loader)
    
    for epoch in range(config.epochs):
        clara_model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # CL: update alpha and beta
            prog = global_step / total_batch_steps
            a = config.alpha_start + (config.alpha_end - config.alpha_start) * prog
            b = config.beta_start + (config.beta_end - config.beta_start) * prog
            clara_model.set_weights(alpha=a, beta=b)
            
            # --- FORWARD PASS ---
            outputs = clara_model(**batch)
            
            loss = outputs["loss"] / config.grad_accumulation_steps
            
            # --- BACKWARD PASS ---
            loss.backward()
            
            # loss change only after accumulation
            if (batch_idx + 1) % config.grad_accumulation_steps == 0 or (batch_idx + 1) == total_batches:
                # Gradient Clipping 
                torch.nn.utils.clip_grad_norm_(clara_model.parameters(), max_norm=1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # logg training metrics
            if global_step % 5 == 0:
                logger.log({
                    "train/total_loss": outputs["loss"].item(),
                    "train/l_absa": outputs["l_absa"].item(),
                    "train/l_rec": outputs["l_rec"].item(),
                    "train/mse_loss": outputs["mse_loss"].item(),
                    "train/alpha": a, 
                    "train/beta": b,
                    "train/lr": lr_scheduler.get_last_lr()[0]
                }, step=global_step)
            
            global_step += 1
            pbar.set_postfix({"L": outputs["loss"].item(), "A": round(a,2), "B": round(b,2)})
            
        # --- VALIDATION ---
        print(f"\nRunning validation for epoch {epoch}...")
        val_metrics = validate_clara(clara_model, val_loader, device)
        logger.log(val_metrics, step=global_step)
        print(f"Val Loss: {val_metrics['val/total_loss']:.4f}, Val ABSA: {val_metrics['val/l_absa']:.4f}")
        
        # --- SAVE LOCALLY ---
        save_path = f"checkpoints/clara_epoch_{epoch}"
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # 1. Save LoRA adapter
            clara_model.model.save_pretrained(save_path)
            
            # 2. Save tokeniser
            tokenizer.save_pretrained(save_path)
            
            # 3. Save embedding weights
            embed_weights = {
                "embed_tokens": clara_model.model.get_input_embeddings().state_dict(),
                "lm_head": clara_model.model.get_output_embeddings().state_dict(),
            }
            torch.save(embed_weights, os.path.join(save_path, "extra_weights.pt"))
            
            print(f"Epoch {epoch}: checkpoint is saved in {save_path}")
        except Exception as e:
            print(f"Error while trying to save: {e}")

    logger.save_final()
    return clara_model, tokenizer

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
        
    # get average metrics
    avg_metrics = {f"val/{k}": v / num_batches for k, v in val_metrics.items()}
    clara_model.train()
    return avg_metrics



if __name__ == "__main__":
    run_experiment()
