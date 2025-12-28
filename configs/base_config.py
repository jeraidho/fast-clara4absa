from dataclasses import dataclass, field

@dataclass
class ClaraConfig:
    # Base model
    model_id: str = "microsoft/Phi-3.5-mini-instruct"

    # Data path
    train_xml: str = "../data/train.xml"
    val_xml: str = "../data/val.xml"
    
    # LoRA params.
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Memory params.
    num_mem_tokens: int = 8
    max_enc_len: int = 256
    max_dec_len: int = 256
    
    # Hyperparams.: training
    batch_size: int = 4
    grad_accumulation_steps: int = 8
    epochs: int = 5
    lr: float = 1e-4
    warmup_steps_ratio: float = 0.1  
    weight_decay: float = 0.01
    grad_accumulation_steps: int = 2
    val_batch_size: int = 16
    eval_every_epoch: bool = True
    task_weights = {"rec": 1.0, "ext": 1.0, "reason": 0.5}

    
    # Curriculum Learning
    # Alpha (Rec): Start -> End
    alpha_start: float = 1.0
    alpha_end: float = 0.2
    
    # Beta (ABSA): Start -> End
    beta_start: float = 0.1
    beta_end: float = 1.0
    
    # Gamma (MSE Alignment): Constant
    gamma: float = 0.1
    
    # Logging
    project_name: str = "Fast-CLaRa-ABSA"
    run_name: str = "Phi3.5-MAMS-Joint-Training"
