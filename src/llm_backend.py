"""
llm_backend.py — Unified LLM backend for MeloMatch.

Provides three backends with identical .chat() interface:
  1. LocalLLM  — HuggingFace transformers (AutoModelForCausalLM + BitsAndBytesConfig)
  2. VllmLLM   — vLLM offline inference (PagedAttention, continuous batching)
  3. APILLM    — OpenAI-compatible API (DashScope, vLLM server, etc.)

Factory function create_llm() dispatches based on config.
"""

import logging
import re
from typing import Optional, Union

logger = logging.getLogger("melomatch.llm_backend")


class LocalLLM:
    """Local HuggingFace transformers LLM backend."""

    def __init__(
        self,
        model,
        tokenizer,
        default_temperature: float = 0.0,
        default_max_tokens: int = 1024,
        model_name: str = "local",
        enable_thinking: Optional[bool] = None,
    ):
        """
        Args:
            model: A loaded transformers model (AutoModelForCausalLM).
            tokenizer: Corresponding tokenizer (AutoTokenizer).
            default_temperature: Default sampling temperature.
            default_max_tokens: Default max new tokens to generate.
            model_name: Human-readable name for logging/metadata.
            enable_thinking: Force chat template thinking mode when supported.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self._warned_thinking_unsupported = False

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from chat messages using local model.

        Args:
            messages: List of {"role": str, "content": str} dicts.
            temperature: Sampling temperature (0 = greedy).
            max_tokens: Max new tokens to generate.

        Returns:
            Generated response string.
        """
        import torch

        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        # Apply chat template (with optional hard switch for thinking mode).
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        try:
            text = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            # Some model templates/tokenizers don't accept enable_thinking.
            if "enable_thinking" in template_kwargs:
                template_kwargs.pop("enable_thinking", None)
                if not self._warned_thinking_unsupported:
                    logger.warning(
                        "Tokenizer chat template does not support enable_thinking; "
                        "falling back without hard thinking switch."
                    )
                    self._warned_thinking_unsupported = True
                text = self.tokenizer.apply_chat_template(messages, **template_kwargs)
            else:
                raise
        # For device_map="auto" (multi-GPU), model.device may not exist.
        # Use the device of the first parameter instead.
        try:
            device = self.model.device
        except AttributeError:
            device = next(self.model.parameters()).device
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=4096,
        ).to(device)
        input_len = inputs["input_ids"].shape[1]

        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        if temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the new tokens
        new_tokens = output_ids[0][input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Qwen3 hybrid thinking mode: strip <think>...</think> blocks
        # that precede the actual response content.
        response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()

        return response


class VllmLLM:
    """vLLM offline inference backend — much faster than raw HuggingFace generate()."""

    def __init__(
        self,
        model_name: str,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        default_temperature: float = 0.0,
        default_max_tokens: int = 1024,
        enable_thinking: Optional[bool] = None,
    ):
        from vllm import LLM

        logger.info(f"Loading vLLM model: {model_name} (tp={tensor_parallel_size}, quant={quantization})")
        kwargs = {
            "model": model_name,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
        }
        if quantization:
            kwargs["quantization"] = quantization
        self.llm = LLM(**kwargs)
        self.model_name = model_name
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.enable_thinking = enable_thinking
        self._warned_thinking_unsupported = False
        logger.info(f"vLLM model loaded: {model_name}")

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        from vllm import SamplingParams

        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9 if temperature > 0 else 1.0,
        )
        chat_kwargs = {}
        if self.enable_thinking is not None:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        try:
            outputs = self.llm.chat([messages], sampling_params=sampling_params, **chat_kwargs)
        except TypeError:
            # Older vLLM versions may not support chat_template_kwargs.
            if chat_kwargs:
                if not self._warned_thinking_unsupported:
                    logger.warning(
                        "vLLM chat does not support chat_template_kwargs.enable_thinking; "
                        "falling back without hard thinking switch."
                    )
                    self._warned_thinking_unsupported = True
                outputs = self.llm.chat([messages], sampling_params=sampling_params)
            else:
                raise
        raw = outputs[0].outputs[0].text.strip()
        # Strip Qwen3 <think>...</think> blocks
        raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
        return raw

    def batch_chat(
        self,
        messages_list: list[list[dict]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> list[str]:
        """Batch inference — feed all prompts at once for maximum throughput."""
        from vllm import SamplingParams

        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9 if temperature > 0 else 1.0,
        )
        chat_kwargs = {}
        if self.enable_thinking is not None:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        try:
            outputs = self.llm.chat(messages_list, sampling_params=sampling_params, **chat_kwargs)
        except TypeError:
            if chat_kwargs:
                if not self._warned_thinking_unsupported:
                    logger.warning(
                        "vLLM chat does not support chat_template_kwargs.enable_thinking; "
                        "falling back without hard thinking switch."
                    )
                    self._warned_thinking_unsupported = True
                outputs = self.llm.chat(messages_list, sampling_params=sampling_params)
            else:
                raise
        results = []
        for out in outputs:
            raw = out.outputs[0].text.strip()
            raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()
            results.append(raw)
        return results


class APILLM:
    """OpenAI-compatible API backend (for DashScope, vLLM server, etc.)."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model_name: str,
        default_temperature: float = 0.0,
        default_max_tokens: int = 1024,
    ):
        from openai import OpenAI

        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response via OpenAI-compatible API.

        Args:
            messages: List of {"role": str, "content": str} dicts.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.

        Returns:
            Generated response string.
        """
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()


def load_model_and_tokenizer(
    model_name: str,
    quantization: str = "4bit",
    lora_checkpoint: Optional[str] = None,
    device_map: str = "auto",
):
    """
    Load a HuggingFace model with optional quantization and LoRA adapter.

    Args:
        model_name: HuggingFace model id (e.g. "Qwen/Qwen3-30B-A3B").
        quantization: "4bit", "8bit", or "none".
        lora_checkpoint: Path to QLoRA adapter weights (optional).
        device_map: Device placement strategy.

    Returns:
        (model, tokenizer) tuple.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"Loading model: {model_name} (quantization={quantization})")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = None
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )

    # Load LoRA adapter if provided
    if lora_checkpoint:
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter: {lora_checkpoint}")
        model = PeftModel.from_pretrained(model, lora_checkpoint)

    model.eval()
    logger.info(f"Model loaded: {model_name} on {model.device}")
    return model, tokenizer


def create_llm(
    config: dict,
    model=None,
    tokenizer=None,
) -> Union["LocalLLM", "APILLM", "VllmLLM"]:
    """
    Create LLM backend from config dict.

    Config keys:
        name: model name/id
        backend: "local", "vllm", or "api"
        api_base: (for API backend)
        api_key: (for API backend)
        quantization: "4bit"/"8bit"/"none"/"awq"/"gptq" (for local/vllm backend)
        lora_checkpoint: path to LoRA adapter (for local backend)
        tensor_parallel_size: (for vllm backend, default 1)
        max_model_len: (for vllm backend, default 4096)
        gpu_memory_utilization: (for vllm backend, default 0.90)

    If model/tokenizer are provided, wraps them in LocalLLM directly.
    """
    import os

    if model is not None and tokenizer is not None:
        return LocalLLM(
            model=model,
            tokenizer=tokenizer,
            model_name=config.get("name", "local"),
            enable_thinking=config.get("enable_thinking"),
        )

    backend = config.get("backend", "api")

    if backend == "vllm":
        # Map BitsAndBytes quantization names to vLLM equivalents
        quant = config.get("quantization")
        if quant in ("4bit", "8bit", "none", None):
            # vLLM doesn't use bnb; pass None (load in native dtype)
            vllm_quant = None
        else:
            vllm_quant = quant  # "awq", "gptq", etc.

        return VllmLLM(
            model_name=config["name"],
            dtype=config.get("dtype", "auto"),
            quantization=vllm_quant,
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
            max_model_len=config.get("max_model_len", 4096),
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.90),
            enable_thinking=config.get("enable_thinking"),
        )
    elif backend == "local":
        m, t = load_model_and_tokenizer(
            model_name=config["name"],
            quantization=config.get("quantization", "4bit"),
            lora_checkpoint=config.get("lora_checkpoint"),
        )
        return LocalLLM(
            model=m,
            tokenizer=t,
            model_name=config["name"],
            enable_thinking=config.get("enable_thinking"),
        )
    else:
        return APILLM(
            api_base=config.get("api_base", ""),
            api_key=config.get("api_key") or os.environ.get("DASHSCOPE_API_KEY", "none"),
            model_name=config["name"],
        )
