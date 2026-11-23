"""
configuration_minilm_v8 — Tuesday V8 的 HuggingFace 配置类。

V8 设计原则：**用 V6 的体型（~0.13B）装下 V7 的全部架构红利**。

    | 项目       | V6 / V6_2                | V7                    | V8（本类）            |
    | ---------- | ------------------------ | --------------------- | --------------------- |
    | 规模       | 768d / 12L / 110.9M      | 1024d / 24L / 304M    | 768d / 16L / ~126M    |
    | FFN        | GELU 3072                | SwiGLU 2816           | SwiGLU 2048           |
    | 归一化     | LayerNorm                | RMSNorm               | RMSNorm（eps=1e-6）   |
    | 位置编码   | 可学习绝对位置（1024 硬上限）| RoPE（外推 2048+） | RoPE（训练 2048/上限 4096，可 YaRN 到 8k+） |
    | 注意力     | MHA（12Q/12KV）          | GQA（16Q/4KV）        | GQA（12Q/4KV）        |
    | QK 归一化  | 无                       | QK-Norm               | QK-Norm               |
    | 权重共享   | 不共享                   | tied                  | tied（省 ~25M）       |
    | KV cache   | 36 KB/token              | 24 KB/token           | 16 KB/token           |
    | EOT        | 无                       | SFT 可选 <|eot|>      | 出厂内置 <|eot|>（Qwen 式预留制词表） |

词表（Qwen 式预留制，参考 Qwen2.5 / Llama-3 的控制符预留方案）：

    0 ~ 32767    BPE 词 + 原有 6 个特殊符（与 V6/V7 tokenizer 完全一致，向下兼容）
    32768        <|eot|>  —— 出厂即有、id 固定，pretrain/SFT/base/instruct 词表永不变化
    32769~33023  255 个预留空槽（将来加 token 不再 resize 矩阵，同 Qwen 151936 留 272 槽）

预训练阶段预留行与 EOT 行不被语料命中、embedding 保持初始化值（EOT 行拷贝 EOS
作暖启动）；SFT 直接在 33024 上训练，base 与 SFT 词表逐字节一致。

model_type 用 "minilm_v8" 与 V6/V7 区分，避免 AutoConfig 串味。
"""
from __future__ import annotations

from transformers import PretrainedConfig

# ---- V8 词表规范（改动须同步 smoke_v8.sh 断言与模型卡） ----
V8_EOT_ID = 32768        # <|eot|> 固定 id（词表末尾第一个预留位）
V8_VOCAB_SIZE = 33024    # = 32768（BPE+specials）+ 1（EOT）+ 255（预留），64 对齐
V8_RESERVED_SLOTS = V8_VOCAB_SIZE - V8_EOT_ID - 1   # = 255


class MiniLMV8Config(PretrainedConfig):
    model_type = "minilm_v8"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = V8_VOCAB_SIZE,   # Qwen 式预留制：含 <|eot|>(32768) 与 255 空槽
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        head_dim: int = 64,
        max_position_embeddings: int = 4096,   # RoPE 预计算上限（训练长度由 seq-len 决定）
        rope_theta: float = 10000.0,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        use_qk_norm: bool = True,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.use_qk_norm = use_qk_norm
        self.use_cache = use_cache

        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads({num_attention_heads}) 必须能被 "
                f"num_key_value_heads({num_key_value_heads}) 整除（GQA）")
        if head_dim * num_attention_heads != hidden_size:
            raise ValueError(
                f"head_dim({head_dim}) × heads({num_attention_heads}) "
                f"= {head_dim * num_attention_heads} 必须等于 hidden_size({hidden_size})")

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
