#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tuesday2-0.12B (V8.5_DPO_v3) Gradio 体验页面 —— FP16 压缩版
- 加载 Tuesday2-0.12B-fp16（权重 FP32 -> FP16，体积 504MB -> 241MB，
  500 题配对 CE 已验证数学无损）
- 词表 33024，<|eot|>=32768，255 预留槽已屏蔽
- gradio==5.x
- 启动: python app2.py  (默认 http://127.0.0.1:3340)

与 app.py（FP32 版，端口 3358）的差异:
1. 模型目录为同目录下的 FP16 权重（本文件需与 Tuesday2-0.12B-fp16 目录同级部署）
2. 推理 dtype 固定 torch.float16（FP32 版为 bfloat16）
3. 服务端口 3340

其余功能与 app.py 一致:
1. 双 Tab：Tab1 单轮对话（每条消息独立，输出框含输入原文，max_new_tokens 默认 1024）、
   Tab2 多轮对话（保留上下文，可清空重开）
3. 两个 Tab 均有独立参数面板与示例问题
4. 生成停止符同时接受 <eos> 与 <|eot|>(32768)
5. 发送后输入框立即清空（沿用 V6_2 修复）

依赖:
- torch>=2.0, transformers>=5.0 (自定义代码需要 5.x)
- 本地无 transformers 5.x 时自动回退到 ../tf5 目录 (相对本文件的 ../../tf5)
"""
import os
import sys
import threading
import time
from pathlib import Path

import gradio as gr

# ---------- transformers 5.x 定位（本地体验环境可能没有装 5.x） ----------
def _ensure_transformers():
    try:
        import transformers  # noqa
        if int(transformers.__version__.split(".")[0]) >= 5:
            return
    except Exception:
        pass
    # 回退: ~/Downloads/llm01b_models/tf5
    here = Path(__file__).resolve().parent          # .../hf/Tuesday2-0.12B
    for cand in (here.parent.parent / "tf5",        # ~/Downloads/llm01b_models/tf5
                 Path.home() / "Downloads/llm01b_models/tf5"):
        if (cand / "transformers").is_dir():
            sys.path.insert(0, str(cand))
            return
    raise RuntimeError("未找到 transformers>=5，请 pip install 'transformers>=5' "
                       "或将 5.x 安装到 llm01b_models/tf5 目录")

_ensure_transformers()
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers import TextIteratorStreamer  # noqa: E402

# ---------- 模型加载 ----------
# 两种模式:
# 1) app2.py 与 config.json 同级 (本地/服务器直接跑) -> 加载本地 FP16 权重
# 2) ModelScope 创空间: 仓库只放 app2.py, 设置 MODEL_REPO_ID 环境变量
#    (或改下面的默认值), 启动时 snapshot_download 从模型仓库在线拉权重
MODEL_ID = os.environ.get("MODEL_REPO_ID", "你的用户名/Tuesday2-0.12B-fp16")

MODEL_DIR = Path(__file__).resolve().parent
if not (MODEL_DIR / "config.json").is_file():
    from modelscope import snapshot_download
    MODEL_DIR = Path(snapshot_download(MODEL_ID))
    print(f"[load] 从模型仓库下载: {MODEL_ID} -> {MODEL_DIR}")
print(f"[load] {MODEL_DIR}")
_tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
# FP16 压缩版：固定以 float16 加载与推理（CPU 上权重同为 fp16 存储）
_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float16
_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, trust_remote_code=True, torch_dtype=_torch_dtype,
)
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model.to(_device).eval()
CTX_LIMIT = int(getattr(_model.config, "max_position_embeddings", 2048))
# 单轮默认把 max_new_tokens 拉满：留出系统模板与提问的开销
SINGLE_MAX_DEFAULT = max(64, CTX_LIMIT - 64)
print(f"[load] done on {_device}, ctx={CTX_LIMIT}")

# 双停止符：<eos> 收尾单轮/最后一条回答，<|eot|>(32768) 收尾多轮中间回答。
STOP_IDS = [_tok.eos_token_id]
_eot = getattr(_tok, "eot_token_id", None)
if _eot is None and "<|eot|>" in _tok.get_vocab():
    _eot = _tok.get_vocab()["<|eot|>"]
if _eot is not None and _eot != _tok.eos_token_id:
    STOP_IDS.append(_eot)

DEFAULTS = dict(temperature=0.7, top_p=0.95, top_k=50,
                repetition_penalty=1.05, max_new_tokens=1024)


def _build_prompt_ids(history, max_new_tokens):
    """历史 -> chat template ids；超出上下文时从最早一轮开始丢弃。"""
    while True:
        enc = _tok.apply_chat_template(
            history, add_generation_prompt=True, tokenize=True)
        ids = enc["input_ids"] if not isinstance(enc, torch.Tensor) else enc.tolist()
        if len(ids) + max_new_tokens <= CTX_LIMIT or len(history) <= 1:
            return ids
        # 丢弃最早一轮 (user+assistant)
        history = history[2:] if len(history) > 2 else history[1:]


def _stream_generate(history, params):
    """流式生成公共实现，逐段 yield 文本与耗时统计。"""
    ids = _build_prompt_ids(history, int(params["max_new_tokens"]))
    input_ids = torch.tensor([ids], device=_device)

    streamer = TextIteratorStreamer(_tok, skip_prompt=True,
                                    skip_special_tokens=True)
    gen_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        do_sample=True,
        temperature=float(params["temperature"]),
        top_p=float(params["top_p"]),
        top_k=int(params["top_k"]),
        repetition_penalty=float(params["repetition_penalty"]),
        max_new_tokens=int(params["max_new_tokens"]),
        pad_token_id=_tok.eos_token_id,
        eos_token_id=STOP_IDS,           # eos + eot 双停止符
    )
    th = threading.Thread(target=_model.generate, kwargs=gen_kwargs)
    th.start()

    partial = ""
    t0 = time.time()
    for piece in streamer:
        partial += piece
        yield partial
    th.join()
    dt = time.time() - t0
    n = len(partial)
    yield f"{partial}\n\n---\n*{n} 字 · {dt:.1f}s · {n / max(dt, 1e-3):.1f} 字/秒*"


def _make_params(temperature, top_p, top_k, repetition_penalty, max_new_tokens):
    return dict(temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens)


# ---------------- Tab 1：单轮对话（每条消息独立，不携带历史） ----------------
def _respond_single(message, temperature, top_p, top_k,
                    repetition_penalty, max_new_tokens):
    """单轮：每次只回答当前这条消息，回答完即结束。"""
    if not message or not message.strip():
        yield "", ""
        return
    q = message.strip()
    params = _make_params(temperature, top_p, top_k,
                          repetition_penalty, max_new_tokens)
    history = [{"role": "user", "content": q}]
    partial = ""
    for piece in _stream_generate(history, params):
        # 输入原文同步回显到输出框：【问】原文 +【答】流式回答
        yield f"【问】{q}\n\n【答】{piece}", ""


# ---------------- Tab 2：多轮对话（携带完整上下文） ----------------
def _respond_multi(message, chat_history, temperature, top_p, top_k,
                   repetition_penalty, max_new_tokens):
    """多轮：把当前消息追加进历史后流式回答；每次 yield 同时清空输入框。"""
    if not message or not message.strip():
        yield chat_history, ""
        return
    params = _make_params(temperature, top_p, top_k,
                          repetition_penalty, max_new_tokens)
    chat_history = chat_history + [{"role": "user", "content": message}]
    history = list(chat_history)
    chat_history = chat_history + [{"role": "assistant", "content": ""}]
    for partial in _stream_generate(history, params):
        chat_history[-1]["content"] = partial
        yield chat_history, ""          # 第二个输出恒为 "" -> 输入框即刻清空


EXAMPLES_SINGLE = [
    "你好，介绍一下你自己。",
    "用一句话解释什么是机器学习。",
    "帮我写一句鼓励学习英语的话。",
    "标准大气压下水的沸点是多少摄氏度？",
    "写一首关于秋天的五言绝句。",
    "为什么天空是蓝色的？请简要说明。",
]

EXAMPLES_MULTI = [
    "我想养一只小狗，有什么建议吗？",
    "帮我规划一下周末两天的短途旅行。",
    "介绍一下长城的历史，我想了解更多。",
    "推荐三部适合周末看的电影，并说明理由。",
]

PARAM_NOTE = (
    "### 生成参数\n"
    "- 0.12B 小模型，事实性知识有限\n"
    "- 出现复读时调高 repetition_penalty\n"
    "- 想要固定输出把 temperature 调到接近 0.1\n"
    "- 生成会同时被 EOS（`</s>`）与轮末符（`<|eot|>`）停止，"
    "任一出现即结束本轮回复，因此实际输出可能短于 max_new_tokens"
)


def _single_tab():
    with gr.Row():
        with gr.Column(scale=3):
            out_box = gr.Textbox(label="回答（含输入原文）", lines=12, max_lines=28,
                                 show_copy_button=True)
            msg = gr.Textbox(placeholder="输入中文消息，回车发送…",
                             label="消息（单轮，每条独立回答）",
                             autofocus=True, lines=1)
            with gr.Row():
                send = gr.Button("发送", variant="primary")
                clear = gr.Button("清空回答")
            gr.Examples(examples=EXAMPLES_SINGLE, inputs=msg,
                        label="示例问题（点击填入）")
        with gr.Column(scale=1):
            gr.Markdown("### 生成参数（单轮）")
            temperature = gr.Slider(0.1, 1.5, value=DEFAULTS["temperature"],
                                    step=0.05, label="temperature")
            top_p = gr.Slider(0.1, 1.0, value=DEFAULTS["top_p"],
                              step=0.05, label="top_p")
            top_k = gr.Slider(1, 100, value=DEFAULTS["top_k"], step=1, label="top_k")
            repetition_penalty = gr.Slider(1.0, 2.0,
                                           value=DEFAULTS["repetition_penalty"],
                                           step=0.05, label="repetition_penalty")
            max_new_tokens = gr.Slider(
                32, SINGLE_MAX_DEFAULT, value=DEFAULTS["max_new_tokens"], step=32,
                label=f"max_new_tokens（默认 1024，上限 {SINGLE_MAX_DEFAULT}）")
            gr.Markdown(PARAM_NOTE)
    inputs = [msg, temperature, top_p, top_k, repetition_penalty, max_new_tokens]
    outputs = [out_box, msg]
    msg.submit(_respond_single, inputs=inputs, outputs=outputs)
    send.click(_respond_single, inputs=inputs, outputs=outputs)
    clear.click(lambda: ("", None), None, [out_box, msg])


def _multi_tab():
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", height=460,
                                 label="多轮对话", show_copy_button=True)
            msg = gr.Textbox(placeholder="输入中文消息，回车发送…",
                             label="消息（多轮，携带上下文）",
                             autofocus=True, lines=1)
            with gr.Row():
                send = gr.Button("发送", variant="primary")
                clear = gr.Button("清空对话")
            gr.Examples(examples=EXAMPLES_MULTI, inputs=msg,
                        label="示例问题（点击填入）")
        with gr.Column(scale=1):
            gr.Markdown("### 生成参数（多轮）")
            temperature = gr.Slider(0.1, 1.5, value=DEFAULTS["temperature"],
                                    step=0.05, label="temperature")
            top_p = gr.Slider(0.1, 1.0, value=DEFAULTS["top_p"],
                              step=0.05, label="top_p")
            top_k = gr.Slider(1, 100, value=DEFAULTS["top_k"], step=1, label="top_k")
            repetition_penalty = gr.Slider(1.0, 2.0,
                                           value=DEFAULTS["repetition_penalty"],
                                           step=0.05, label="repetition_penalty")
            max_new_tokens = gr.Slider(
                32, SINGLE_MAX_DEFAULT, value=DEFAULTS["max_new_tokens"], step=32,
                label=f"max_new_tokens（默认 1024，上限 {SINGLE_MAX_DEFAULT}）")
            gr.Markdown(PARAM_NOTE)
    inputs = [msg, chatbot, temperature, top_p, top_k,
              repetition_penalty, max_new_tokens]
    outputs = [chatbot, msg]
    msg.submit(_respond_multi, inputs=inputs, outputs=outputs)
    send.click(_respond_multi, inputs=inputs, outputs=outputs)
    clear.click(lambda: (None, ""), None, [chatbot, msg])


with gr.Blocks(title="Tuesday2-0.12B FP16") as demo:
    gr.Markdown(
        "# Tuesday2-0.12B · 中文对话体验（FP16 压缩版）\n"
        "1.26 亿参数中文因果语言模型（V8.5_DPO_v3，从零预训练 19.35B tokens + "
        "三阶段 SFT + 自采样 DPO 偏好对齐），从零训练，`trust_remote_code=True` "
        f"加载。上下文 {CTX_LIMIT} token。本页权重为 **FP16 压缩版**"
        "（504MB → 241MB，配对 CE 验证无损），推理精度 FP16。"
    )
    with gr.Tabs():
        with gr.Tab("单轮对话"):
            _single_tab()
        with gr.Tab("多轮对话"):
            _multi_tab()

if __name__ == "__main__":
    # 本地默认 3340；ModelScope Docker 创空间要求监听 7860（Dockerfile 里设 PORT=7860）
    _port = int(os.environ.get("PORT", 3340))
    demo.queue().launch(server_name="0.0.0.0", server_port=_port,
                        show_error=True)
