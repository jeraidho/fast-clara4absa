# ==============================================================================
# FAST CLARA ABSA: BATCH STRUCTURE SPECIFICATION
# ==============================================================================
# B: Batch Size (e.g, 8, 16 or 32)
# L_enc: Max Encoder Length (config: 256)
# L_dec: Max Decoder Length (config: 256)
# ==============================================================================

BATCH_SCHEMA = {
    # --- INPUT FOR ENCODER (TEXT COMPRESSION) ---
    "enc_input_ids":  torch.LongTensor,   # Size: [B, L_enc]
    # Includes: [Original text tokens] + [Memory tokens M0..Mk] + [Padding]

    "enc_mask":       torch.BoolTensor,   # Size: [B, L_enc]
    # Includes: 1 for real tokens, 0 for padding

    # --- INPUT FOR DECODER (DISCUSSION/GENERATION) ---
    "dec_input_ids":  torch.LongTensor,   # Size: [B, L_dec]
    # Includes: [Memory tokens M0..Mk] + [Tokens of task prompt] + [Target answer token] + [Padding]

    "dec_mask":       torch.BoolTensor,   # Size: [B, L_dec]
    # Includes: 1 for memory, prompt and answer. 0 for padding

    # --- TARGETS FOR COUNTING LOSS ---
    "labels":         torch.LongTensor,   # Size: [B, L_dec]
    # Includes: -100 for all positions, except answer tokens (Sentiment/Text/JSON)
    # Allows counting CrossEntropy only for correct part of generation.

    # --- METADATA ---
    "task":           list[str],          # Size: [B]
    # Includes: String identifiers ("rec", "ext", "reason") for each task in training
}