"""
modeling_minilm_v8 — Tuesday V8 的 HuggingFace 实现（小而强版）。

V8 设计原则：**用 V6 的体型（~0.13B）装下 V7 的全部架构红利**。
在 V6（768d×12L、可学习绝对位置、MHA、GELU、LayerNorm）基础上修改：

  1. RoPE 旋转位置编码取代可学习绝对位置嵌入（V6 最大的结构短板：
     位置表 1024 行是硬上限；RoPE 无参数、相对、可外推+YaRN 扩展）
  2. RMSNorm 取代 LayerNorm；SwiGLU 取代 GELU（V7 验证过的收益）
  3. GQA（12 Q / 4 KV）+ QK-Norm：KV cache 16KB/token，V6 是 36KB（省 56%）
  4. tied embedding（V7 借鉴）：省 ~25M 参数
  5. 上下文：训练 2048（V6/V7 都是 1024），RoPE theta=10000，max-pos 4096，
     后续可 YaRN 外推到 8k+

规模：768d × 16L × 12Q/4KV（head_dim 64）× FFN 2048，约 126M 参数。
（12L 轻量版 ≈ 101M，比 V6_2 的 110.9M 还小，用 --num-hidden-layers 12 切换）

权重命名与 HF 惯例一致（Llama 风格），`trust_remote_code=True` 直接加载。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache

try:                                            # transformers>=4.50 需要显式继承
    from transformers.generation.utils import GenerationMixin
except Exception:                               # 老版本里 PreTrainedModel 已包含
    GenerationMixin = object
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

try:                                            # 本地包内相对导入
    from .configuration_minilm_v8 import MiniLMV8Config
except ImportError:                             # trust_remote_code / 单文件加载
    from configuration_minilm_v8 import MiniLMV8Config


# ------------------------------------------------------------------ 基础组件
class MiniLMV8RMSNorm(nn.Module):
    """RMSNorm：只用均方根归一化，无均值中心化、无 bias。fp32 计算保稳定。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class MiniLMV8RotaryEmbedding(nn.Module):
    """RoPE：预先算好 cos/sin 查表，position_ids 超表时自动扩展（支持外推）。

    为什么 forward 里要重建缓存（踩过的坑）：cos/sin/inv_freq 都是
    `persistent=False` 的 buffer，不进 state_dict；而 transformers 5 加载模型时
    先在 **meta device** 上初始化再灌权重，非持久 buffer 不会被覆盖，会一直停在
    meta 上，于是首帧前向就报 `Cannot copy out of meta tensor; no data!`。
    所以这里检测「空 / meta / 设备不符 / 需要更长 / 构建设备不符」任一情况就地重建。
    （to_empty 会把 meta buffer 变成真张量但内容是垃圾，必须靠 _built_on 记录来识别。）
    """

    def __init__(self, head_dim: int, max_position_embeddings: int = 2048,
                 base: float = 10000.0):
        super().__init__()
        self.head_dim = int(head_dim)
        self.base = float(base)
        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", torch.zeros(0), persistent=False)
        self.register_buffer("sin_cached", torch.zeros(0), persistent=False)
        self._build_cache(int(max_position_embeddings), torch.device("cpu"))
        self._built_on = "cpu"

    @torch.no_grad()
    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        seq_len = max(int(seq_len), 1)
        inv_freq = 1.0 / (self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32,
                         device=device) / self.head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)                            # (S, D/2)
        emb = torch.cat((freqs, freqs), dim=-1)                     # (S, D)
        self.cos_cached = emb.cos()                                 # 仍是 buffer
        self.sin_cached = emb.sin()
        self.max_seq_len_cached = seq_len
        self._built_on = str(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        need = int(position_ids.max().item()) + 1
        cos = self.cos_cached
        # 内容抽查：to_empty 物化的 buffer 可能是 NaN/垃圾，也可能是恰好清零的
        # 内存页——全零缓存同样会让 RoPE 失效（位置盲→复读退化），且能骗过
        # 「isfinite + |cos|<=1」这类值域检查。所以直接对照 RoPE 数学定义
        # 抽查第 1 行：cos[p,k]=cos(p*inv_freq[k])，p=1 时
        # cos[1,0]=cos(1)≈0.540302、sin[1,0]=sin(1)≈0.841471，
        # cos[1,1]=cos(inv_freq[1])=cos(base^(-2/head_dim))。不符即重建。
        cache_ok = False
        if (cos.numel() and not cos.is_meta and cos.device == x.device
                and need <= self.max_seq_len_cached
                and self.sin_cached.numel() == cos.numel()):
            f1 = self.base ** (-2.0 / self.head_dim)  # inv_freq[1]
            cache_ok = (
                abs(float(cos[1, 0]) - math.cos(1.0)) < 1e-3
                and abs(float(self.sin_cached[1, 0]) - math.sin(1.0)) < 1e-3
                and abs(float(cos[1, 1]) - math.cos(f1)) < 1e-3
            )
        if (not cache_ok or cos.device != x.device
                or need > self.max_seq_len_cached):  # 外推 / meta / 物化垃圾：重建
            self._build_cache(max(need, self.max_seq_len_cached), x.device)
        cos = self.cos_cached[position_ids].to(x.dtype)
        sin = self.sin_cached[position_ids].to(x.dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor):
    """q/k: (B, H, S, D)；cos/sin: (B, S, D) -> 广播到 (B, 1, S, D)。"""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


ACT2FN = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
}


# ------------------------------------------------------------------ 模块
class MiniLMV8Attention(nn.Module):
    """GQA + QK-Norm + RoPE 的因果自注意力（SDPA 后端）。"""

    def __init__(self, config: MiniLMV8Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size,
                                self.num_heads * self.head_dim,
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size,
                                self.num_kv_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size,
                                self.num_kv_heads * self.head_dim,
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                config.hidden_size,
                                bias=config.attention_bias)
        self.o_proj._residual_scaled = True      # 残差投影：初始化时缩小 std

        if config.use_qk_norm:
            self.q_norm = MiniLMV8RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = MiniLMV8RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,   # (B, KV) 1/0
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:              # QK-Norm：按 head_dim 归一化
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        # GQA：KV 头复制到 Q 头数（SDPA 统一走 MHA 形状）
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        kv_len = k.shape[2]
        past_len = kv_len - q_len

        attn_mask = None
        need_causal = q_len > 1
        need_pad = attention_mask is not None and bool((attention_mask == 0).any())
        if need_causal or need_pad or past_len > 0:
            attn_mask = torch.zeros(q_len, kv_len, dtype=q.dtype, device=q.device)
            if need_causal:
                # 位置 i（绝对位置 past_len+i）只能看到 <= 它的 key
                future = torch.ones(q_len, kv_len, dtype=torch.bool,
                                    device=q.device).triu(diagonal=past_len + 1)
                attn_mask = attn_mask.masked_fill(future, torch.finfo(q.dtype).min)
            if need_pad:
                pad = (1.0 - attention_mask.to(q.dtype)).unsqueeze(1).unsqueeze(1)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0) + pad * torch.finfo(q.dtype).min
            else:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            scale=self.scaling,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(
            bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(out)


class MiniLMV8MLP(nn.Module):
    """SwiGLU：down(silu(gate(x)) * up(x))，三矩阵同参数下强于两矩阵 GELU。"""

    def __init__(self, config: MiniLMV8Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size,
                                   bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size,
                                 bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size,
                                   bias=config.mlp_bias)
        self.down_proj._residual_scaled = True   # 残差投影：初始化时缩小 std
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MiniLMV8DecoderLayer(nn.Module):
    def __init__(self, config: MiniLMV8Config, layer_idx: int):
        super().__init__()
        self.self_attn = MiniLMV8Attention(config, layer_idx)
        self.mlp = MiniLMV8MLP(config)
        self.input_layernorm = MiniLMV8RMSNorm(config.hidden_size,
                                               eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniLMV8RMSNorm(config.hidden_size,
                                                        eps=config.rms_norm_eps)

    def forward(self, hidden_states, cos, sin, attention_mask=None,
                past_key_values=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(
            hidden_states, cos, sin, attention_mask, past_key_values, use_cache)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


# ------------------------------------------------------------------ 模型
class MiniLMV8PreTrainedModel(PreTrainedModel):
    config_class = MiniLMV8Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _no_split_modules = ["MiniLMV8DecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            # 残差输出投影（attn.o_proj / mlp.down_proj）按 GPT-2 惯例缩放，
            # 否则 24 层残差初值过大，训练早期 loss 明显偏高。
            if getattr(module, "_residual_scaled", False):
                std = std / math.sqrt(2 * self.config.num_hidden_layers)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, MiniLMV8RMSNorm):
            module.weight.data.fill_(1.0)       # 归一化权重必须回到 1


class MiniLMV8Model(MiniLMV8PreTrainedModel):
    """embedding + N×decoder layer + final RMSNorm（无位置嵌入，位置来自 RoPE）。"""

    def __init__(self, config: MiniLMV8Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         padding_idx=None)
        self.rotary_emb = MiniLMV8RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000.0))
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList(
            [MiniLMV8DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = MiniLMV8RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_hidden_states = bool(output_hidden_states)

        bsz, seq_len = input_ids.shape
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = past_key_values.get_seq_length()
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len,
                                        dtype=torch.long, device=input_ids.device
                                        ).unsqueeze(0).expand(bsz, -1)

        hidden_states = self.dropout(self.embed_tokens(input_ids))
        cos, sin = self.rotary_emb(hidden_states, position_ids)   # 全局算一次

        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = layer(hidden_states, cos, sin, attention_mask,
                                  past_key_values, use_cache)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )


class MiniLMV8ForCausalLM(MiniLMV8PreTrainedModel, GenerationMixin):
    """因果语言模型：`model` + 输出头，与 embed_tokens 共享权重。"""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniLMV8Config):
        super().__init__(config)
        self.model = MiniLMV8Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = outputs.last_hidden_state
        slice_len = hidden_states.shape[1] if logits_to_keep == 0 else logits_to_keep
        logits = self.lm_head(hidden_states[:, -slice_len:, :]).float()

        loss = None
        if labels is not None:
            # HF 语义：logits[t] 预测 labels[t+1]（内部右移一位）
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = past_key_values.get_seq_length()
        if past_len > 0:
            input_ids = input_ids[:, past_len:]

        position_ids = None
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids[:, -input_ids.shape[1]:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return (past_key_values.reorder_cache(beam_idx)
                if hasattr(past_key_values, "reorder_cache") else past_key_values)
