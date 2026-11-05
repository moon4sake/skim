MODEL_REGISTRY = {
    "qwen35_0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen35_2b": "Qwen/Qwen3.5-2B",
    "qwen35_4b": "Qwen/Qwen3.5-4B",
    "qwen35_9b": "Qwen/Qwen3.5-9B",
}

ALPHA = (1.0, 1.0, 0.2)
FAMILY_RATIO = {"mlp": 1.0, "linear_attn": 0.6, "attention": 0.0}
LAYER_SCHEDULE = "adaptive"

N_XLAM = 4096
XLAM_SEED = 0
MAX_SEQ_LEN = 1300

EPOCHS = 2
LR = 1e-5
BATCH = 1
GRAD_ACCUM = 4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
LR_WARMUP_FRAC = 0.1
SEED = 0

LORA_LR = 1e-4
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
)

TEMPERATURE = 2.0
LAMBDA_HAT_H = 1.1
LAMBDA_KD = 0.0
OBJECTIVE = "skim"

KD_SCHEDULE = "hybrid"
OPD_WARMUP_FRAC = 0.25
ROLLOUT_MAX_NEW_TOKENS = 300

EVAL_MAX_NEW_TOKENS = 1024
EVAL_DECODING_MODES = ("grammar", "free")

RECORD_CACHE_SHARDS = 4
MAX_SPARSITY_DRIFT = 0.02
