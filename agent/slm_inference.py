"""SLM inference backends and sequence log-prob helpers."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Protocol


class ModelAdapter(Protocol):
    model_name: str

    def sequence_logprob(self, prompt: str, candidate: str) -> float:
        """Return log P(candidate | prompt)."""

    def generate_after_prefix(
        self,
        prompt: str,
        prefix: str,
        max_new_tokens: int,
        temperature: float | None = None,
    ) -> str:
        """Generate continuation text after prompt+prefix."""


class TransformersModelAdapter:
    """Transformers adapter with explicit sequence log-prob scoring."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        torch_dtype: str = "float16",
        low_cpu_mem_usage: bool = True,
        use_chat_template: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for real model inference") from exc

        self.model_name = model_name
        self.use_chat_template = bool(use_chat_template)
        self._torch = torch
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        resolved_dtype = dtype_map.get(torch_dtype, torch.float16)
        self._torch_dtype_name = torch_dtype
        force_slow_tokenizer = "mamba2" in model_name.lower()
        if force_slow_tokenizer:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=False,
                    trust_remote_code=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load slow tokenizer for {model_name}. "
                    "Install tokenizer deps: pip install sentencepiece protobuf"
                ) from exc
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )
            except Exception:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        use_fast=False,
                        trust_remote_code=True,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load tokenizer for {model_name}. "
                        "Install tokenizer deps: pip install sentencepiece protobuf"
                    ) from exc
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self._is_encoder_decoder = bool(getattr(cfg, "is_encoder_decoder", False))
        if self._is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=resolved_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=resolved_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                trust_remote_code=True,
            )

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"
        self.chat_template_available = callable(getattr(self.tokenizer, "apply_chat_template", None))
        if self.use_chat_template and not self.chat_template_available:
            raise RuntimeError(
                f"use_chat_template=true but tokenizer for {model_name} does not support apply_chat_template"
            )

        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self._max_input_tokens = self._resolve_max_input_tokens(cfg)
        try:
            gen_cfg = self.model.generation_config
            # Avoid noisy warnings for non-sampling generation.
            gen_cfg.temperature = None
            gen_cfg.top_p = None
            gen_cfg.top_k = None
        except Exception:
            pass

    def _format_prompt_text(self, prompt: str, add_generation_prompt: bool) -> str:
        if not self.use_chat_template:
            return prompt
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise RuntimeError(
                f"chat_template_unavailable: model={self.model_name} use_chat_template=true"
            )
        messages = [{"role": "user", "content": prompt}]
        try:
            return str(
                apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            )
        except TypeError:
            try:
                return str(apply_chat_template(messages, tokenize=False))
            except Exception as exc:
                raise RuntimeError(
                    f"chat_template_apply_failed: model={self.model_name} error={exc}"
                ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"chat_template_apply_failed: model={self.model_name} error={exc}"
            ) from exc

    def _resolve_max_input_tokens(self, cfg) -> int:
        candidates = [
            getattr(cfg, "max_position_embeddings", None),
            getattr(cfg, "n_positions", None),
            getattr(cfg, "max_seq_len", None),
            getattr(cfg, "model_max_length", None),
            getattr(self.tokenizer, "model_max_length", None),
        ]
        vals = []
        for c in candidates:
            if isinstance(c, int) and c > 0 and c < 1_000_000:
                vals.append(c)
        if not vals:
            return 1024
        return int(min(vals))

    def sequence_logprob(self, prompt: str, candidate: str) -> float:
        torch = self._torch
        prompt_text = self._format_prompt_text(prompt, add_generation_prompt=True)
        with torch.no_grad():
            if self._is_encoder_decoder:
                enc = self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self._max_input_tokens,
                ).to(self.device)
                labels = self.tokenizer(
                    candidate,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=True,
                    max_length=min(64, self._max_input_tokens),
                ).input_ids.to(self.device)
                outputs = self.model(**enc, labels=labels)
                if outputs.loss is None or not math.isfinite(float(outputs.loss.item())):
                    raise RuntimeError(
                        f"non_finite_model_forward: model={self.model_name} dtype={self._torch_dtype_name} "
                        "encoder_decoder_loss is non-finite"
                    )
                non_pad = labels.ne(self.tokenizer.pad_token_id).sum().item() if self.tokenizer.pad_token_id is not None else labels.numel()
                return float(-outputs.loss.item() * max(1, int(non_pad)))
            cand_ids = self.tokenizer(candidate, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            cand_len = int(cand_ids.shape[1])
            max_prompt_len = max(8, self._max_input_tokens - cand_len)
            prompt_ids = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=max_prompt_len,
            ).input_ids.to(self.device)
            input_ids = torch.cat([prompt_ids, cand_ids], dim=1)
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits
            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    f"non_finite_model_forward: model={self.model_name} dtype={self._torch_dtype_name} "
                    "decoder logits contain non-finite values"
                )
            log_probs = torch.log_softmax(logits, dim=-1)
            if not torch.isfinite(log_probs).all():
                raise RuntimeError(
                    f"non_finite_model_forward: model={self.model_name} dtype={self._torch_dtype_name} "
                    "log_softmax output contains non-finite values"
                )

            prompt_len = prompt_ids.shape[1]
            total = 0.0
            for j in range(cand_len):
                pos = prompt_len + j - 1
                token_id = cand_ids[0, j]
                total += float(log_probs[0, pos, token_id].item())
            return total

    def generate_after_prefix(
        self,
        prompt: str,
        prefix: str,
        max_new_tokens: int,
        temperature: float | None = None,
    ) -> str:
        torch = self._torch
        input_text = self._format_prompt_text(prompt, add_generation_prompt=True)
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max(8, self._max_input_tokens - max_new_tokens),
        ).to(self.device)
        gen_kwargs = {}
        if self._is_encoder_decoder:
            prefix_ids = self.tokenizer(
                prefix,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=8,
            ).input_ids.to(self.device)
            if prefix_ids.shape[1] > 0:
                gen_kwargs["decoder_input_ids"] = prefix_ids
        else:
            # Decoder-only models use prefix appended to prompt.
            input_text = input_text + "\n" + prefix
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=max(8, self._max_input_tokens - max_new_tokens),
            ).to(self.device)
        with torch.no_grad():
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            do_sample = bool(temperature is not None and float(temperature) > 0.0)
            gen_temperature = float(temperature) if do_sample else None
            if do_sample and gen_temperature <= 0.0:
                do_sample = False
                gen_temperature = None
            gen_args = {
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": 1,
                "do_sample": do_sample,
                "pad_token_id": pad_token_id,
                **gen_kwargs,
            }
            if do_sample and gen_temperature is not None:
                gen_args["temperature"] = gen_temperature
                gen_args["top_p"] = 0.95
            output_ids = self.model.generate(
                **inputs,
                **gen_args,
            )
        if self._is_encoder_decoder:
            seq = output_ids[0]
            if "decoder_input_ids" in gen_kwargs:
                prefix_len = int(gen_kwargs["decoder_input_ids"].shape[1])
                seq = seq[prefix_len:]
            return self.tokenizer.decode(seq, skip_special_tokens=True).strip()
        prompt_len = int(inputs["input_ids"].shape[1])
        continuation_ids = output_ids[0, prompt_len:]
        return self.tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()


def softmax_from_logps(logps: Dict[str, float]) -> Dict[str, float]:
    max_logp = max(logps.values())
    exps = {k: math.exp(v - max_logp) for k, v in logps.items()}
    denom = sum(exps.values())
    return {k: v / denom for k, v in exps.items()}


def build_model_adapters(
    model_names: List[str],
    backend: str,
    device: str,
    seed: int,
    torch_dtype: str = "float16",
    low_cpu_mem_usage: bool = True,
    model_torch_dtype_overrides: Optional[Dict[str, str]] = None,
    use_chat_template: bool = False,
) -> List[ModelAdapter]:
    del seed
    if backend != "transformers":
        raise ValueError("Only backend='transformers' is supported. Mock backend is disabled.")
    adapters: List[ModelAdapter] = []
    overrides = model_torch_dtype_overrides or {}
    for model_name in model_names:
        model_dtype = overrides.get(model_name, torch_dtype)
        adapters.append(
            TransformersModelAdapter(
                model_name=model_name,
                device=device,
                torch_dtype=model_dtype,
                low_cpu_mem_usage=low_cpu_mem_usage,
                use_chat_template=use_chat_template,
            )
        )
    return adapters
