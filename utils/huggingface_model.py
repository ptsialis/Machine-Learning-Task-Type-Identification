import os, gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# Best to set before CUDA initializes:
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def load_model(model_id="Qwen/Qwen3-14B", quantization=""):
    """
    Returns a transformers pipeline object (so your caller code stays the same).
    The pipeline already carries `pipe.tokenizer` and `pipe.model`.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device_map = {
    "model.embed_tokens": 0,
    "model.layers": 1,   # all transformer blocks on GPU 1
    "model.norm": 1,
    "lm_head": 1,
}
    kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )

    if quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs.pop("torch_dtype", None)
    elif quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs.pop("torch_dtype", None)

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,   # return only newly generated text
    )


def _normalize_reasoning_flag(reasoning):
    """
    Accepts: "on"/"off", True/False, "true"/"false", None.
    Defaults to True for backward compatibility.
    """
    if reasoning is None:
        return True
    if isinstance(reasoning, bool):
        return reasoning
    s = str(reasoning).strip().lower()
    if s in {"on", "true", "1", "yes", "y"}:
        return True
    if s in {"off", "false", "0", "no", "n"}:
        return False
    return True


def _strip_think(text: str) -> str:
    """
    Qwen3 thinking mode often wraps reasoning in <think>...</think>.
    For your downstream label parsing, returning only the final answer is usually safer.
    """
    if not isinstance(text, str):
        return str(text)
    if "<think>" in text and "</think>" in text:
        # keep only content after </think>
        return text.split("</think>", 1)[1].strip()
    return text.strip()


def ask_model_prompt(
    prompt,
    system_prompt=None,
    pipe=None,
    max_new_tokens=7000,
    reasoning="on",                 # <-- NEW: matches your experiment script
    temperature=None,
    top_p=None,
    top_k=20,
    strip_think=False,               # keep default True so parsing is stable
):
    if pipe is None:
        return "Model was not loaded! Use load_model() before running this function."

    enable_thinking = _normalize_reasoning_flag(reasoning)

    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    tok = getattr(pipe, "tokenizer", None)

    # Build final text prompt (recommended for Qwen3)
    if tok is not None and hasattr(tok, "apply_chat_template"):
        text_prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,   # Qwen3 switch
        )
    else:
        # fallback for non-chat-template models
        text_prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"

    # Qwen3 recommended-ish defaults differ by thinking vs non-thinking; allow override
    if temperature is None:
        temperature = 0.6 if enable_thinking else 0.7
    if top_p is None:
        top_p = 0.95 if enable_thinking else 0.8

    with torch.inference_mode():
        outputs = pipe(
            text_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    # Robust extraction across pipeline variants
    gen = outputs[0].get("generated_text", outputs[0])

    # Some pipelines return list-of-dicts chat-like; handle that too
    if isinstance(gen, list) and gen and isinstance(gen[-1], dict) and "content" in gen[-1]:
        gen_text = gen[-1]["content"]
    else:
        gen_text = gen

    return _strip_think(gen_text) if strip_think else str(gen_text)


def ask_model_mes(messages, pipe=None, max_new_tokens=10000, reasoning="on"):
    """
    Kept for compatibility. Expects `messages` list with role/content dicts.
    """
    if pipe is None:
        return "Model was not loaded! Use load_model() before running this function."

    # Reuse ask_model_prompt logic by converting messages->prompt:
    # Here we pass the whole messages through chat_template directly.
    enable_thinking = _normalize_reasoning_flag(reasoning)
    tok = getattr(pipe, "tokenizer", None)

    if tok is not None and hasattr(tok, "apply_chat_template"):
        text_prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    else:
        text_prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"

    with torch.inference_mode():
        outputs = pipe(text_prompt, max_new_tokens=max_new_tokens)

    gen = outputs[0].get("generated_text", outputs[0])
    if isinstance(gen, list) and gen and isinstance(gen[-1], dict) and "content" in gen[-1]:
        return gen[-1]["content"]
    return str(gen)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()