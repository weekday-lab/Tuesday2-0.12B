# Tuesday2-0.12B

**Tuesday2-0.12B**（内部代号 V8.5_DPO_v3）是 Tuesday 系列自研中文对话模型的第二代旗舰版本：从零预训练（零依赖纯 PyTorch 训练栈，非微调任何第三方模型）+ 三阶段 SFT 配方 + 自采样偏好 DPO 精修，在 200 题标准评测中取得 **55.5%** 的系列历史最高分。

[v7版本](https://github.com/weekday-lab/Tuesday-0.1B-v7)

[模型地址](https://modelscope.cn/models/Weekday-Lab/Tuesday2-0.12B),提申请下载。

[开源github地址](https://github.com/weekday-lab/Tuesday2-0.12B)不包含模型。


---
## 0. 声明
开源大模型V8版本，采用能获取到的开源数据预训练及SFT，DPO采用大模型打分数据进行训练，遵循数据的协议，严禁商用，否则发布者及关联方不承担任何不利后果及连带责任。

## 1. 模型概况

| 项目 | 值 |
|------|-----|
| 参数量 | 126.05M（0.12B） |
| 结构 | GPT-2 式 decoder-only，16 层 / hidden 768 / 12 头 |
| 词表 | 33024（Qwen 式预留制）|
| 分词器 | 自研 32k BPE（256 字节 + 32,506 条合并规则 + 6 个 special tokens），中位压缩率优于同规模常见方案 |
| 特殊 token | `<|eot|>` 固定 id 32768（对话轮末，出厂内置）；32769~33023 共 255 个预留槽（未来扩展零成本） |
| 上下文长度 | 2048 tokens |
| 权重格式 | safetensors FP32（504MB）；FP16 版 241MB（500 题配对 CE 验证无损，ΔCE=0.0002，p=0.67） |
| 依赖 | transformers ≥ 4.x（trust_remote_code 自定义代码随包分发），无其他依赖 |

## 2. 评测成绩（截至发布日）

| 口径 | Tuesday2-0.12B | 前代最佳 [V7_DPO](https://github.com/weekday-lab/Tuesday-0.1B-v7) | 备注 |
|------|---------------|------------------|------|
| 200 题综合 | **55.5%** | 47.5% | +8.0pp |
| 事实问答 (60题) | 55.0% | 50.0% | +5.0pp |
| 常识推理 (50题) | **80.0%** | 62.0% | +18.0pp |
| 多轮指代 (40题) | 40.0% | 50.0% | -10.0pp（唯一落后项，见 5.2） |
| 指令遵循 (50题) | **44.0%** | 28.0% | +16.0pp |
| DuConv 多轮裁判分 (100段) | **3.23**（≥4分占 53%）| 1.80（≥4分占 1%）| 真实多轮体验碾压 |
| 配对 CE（500题, token加权）| **2.2591** (PPL 9.57) | 2.4227 (PPL 11.28) | 450/500 胜，p≈1e-133 |

## 3. 训练数据声明

### 3.1 使用的数据

| 阶段 | 数据集 | 用途 |
|------|--------|------|
| 预训练 | FineWeb-Edu-Chinese v2.1 | 教育质量中文网页语料（主力） |
| 预训练 | 中文维基百科（Wikipedia zh） | 知识型长文本 |
| 预训练 | MAP-CC（zh_baike 部分） | 中文网页语料 |
| 预训练 | 百度百科 QA 2019（BaikeQA） | 问答形态知识语料 |
| SFT | 自建中文指令精调数据（多源聚合，经质量打分筛选） | 指令与对话能力 |
| SFT | TigerBot 开源数据（规则过滤） | 长问答 |
| SFT | Infinity-Instruct 中文子集 | 指令与多轮 |
| SFT | DuConv（百度） | 知识型多轮对话 |
| DPO | 模型自采样偏好对（裁判模型打分，严格 held-out，不使用任何评测集） | 偏好对齐 |

以上数据仅用于模型训练，模型输出不代表任何数据来源方的观点；预训练语料已做去重与质量过滤（NFKC 归一化、去 HTML/URL、乱码过滤、blake2b 内容哈希去重）。

### 3.2 协议遵循

- 各数据集均来自其发布方的公开发布渠道，本模型的使用遵循各数据集**原始发布协议**；具体授权条款请以各数据集官方发布页为准（如中文维基百科内容遵循 CC BY-SA，百度百科 QA 与 DuConv 面向研究用途发布）。
- 若上游数据协议要求，本模型仅限研究/非商业用途分发；商业使用前请自行核实上述数据集的最新许可条款。
- 本模型不包含训练数据的原始内容副本，亦不提供还原训练数据的能力承诺。
- 请严格遵循上游数据的协议，如有商用需求，请联系获取清洗数据后的商用版本。
- 凡是不遵循上游数据协议，导致的各种不利后果，本发布者及关联方不承担任何责任及连带责任。

## 4. 快速上手

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Tuesday2-0.12B", trust_remote_code=True, dtype=torch.bfloat16
).eval()
tok = AutoTokenizer.from_pretrained("Tuesday2-0.12B", trust_remote_code=True)

messages = [{"role": "user", "content": "用一句话介绍大语言模型的训练流程。"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").to(model.device)

out = model.generate(
    **ids,
    max_new_tokens=160,
    do_sample=False,
    eos_token_id=[tok.eos_token_id, tok.get_vocab()["<|eot|>"]],
)
print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
```

要点：
- **必须 `trust_remote_code=True`**（自定义 `modeling_minilm_v8.py` / `tokenization_minilm.py` 随包分发）。
- 停止符除 EOS 外还需 **`<|eot|>`**（id 32768），否则会生成到 max_new_tokens。
- 预留槽 32769~33023 的 logits 已在代码内屏蔽（-inf），无法被采样到。
- 计算精度建议 BF16（与评测口径一致）；FP16 亦可（CE 验证无损）。

## 5. 已知局限

1. **0.12B 体量的固有限制**：事实性知识有限，长答案细节会出错；不建议用于事实敏感场景的最终答案来源。
2. **多轮指代题型仍弱于长上下文大模型**（40% vs 50% 的前代 V7 指代小样本项），多轮数据继续扩配比是下一代（V8.6）主攻方向。
3. 中文为主，英文能力仅来自字节级 BPE 兜底，未做英文专项。
4. 200 题评测每类仅 40~60 题，±2 题 ≈ 2~4pp，对比小于该差距的结论请以配对 CE 为准。

## 6. 文件清单

| 文件 | 说明 |
|------|------|
| model.safetensors | FP32 权重（504MB） |
| config.json / generation_config.json | 模型与生成配置 |
| configuration_minilm_v8.py / modeling_minilm_v8.py | 自定义模型代码（trust_remote_code 加载） |
| tokenizer.json / vocab.json / merges.txt / tokenizer_config.json / special_tokens_map.json / tokenization_minilm.py | 分词器全套 |
| chat_template.jinja | 对话模板（与训练 encode_dialogue 逐 token 一致，由 verify 脚本保证） |

## 7. 许可与版本

- 模型权重与随包代码仅供研究使用；上游数据集许可见第 3.2 节。
- 版本：Tuesday2-0.12B = V8.5_DPO_v3（checkpoint: sft_v85_dpo3，2026-11-20 定版）。
