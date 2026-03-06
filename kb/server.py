"""Declarative Memory KB server implementation."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


@dataclass
class DMItem:
    dm_id: str
    dm_text: str
    dm_goal_text: str
    dm_condition_text: str
    metadata: Dict


class BaseEmbedder:
    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class HashEmbedder(BaseEmbedder):
    """Offline fallback embedder for deterministic local tests."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _normalize_text(text).lower().split():
                idx = hash(tok) % self.dim
                vectors[i, idx] += 1.0
        return _l2_normalize(vectors)


class HFEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "google/embeddinggemma-300m", device: str = "cuda:0") -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers and torch are required for HF embedder") from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.dim = int(self.model.config.hidden_size)
        self.model_name = model_name

    def embed(self, texts: List[str]) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            batch = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=256,
            ).to(self.device)
            outputs = self.model(**batch)
            last_hidden = outputs.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            arr = pooled.detach().cpu().numpy().astype(np.float32)
            return _l2_normalize(arr)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class KBServer:
    def __init__(
        self,
        kb_dir: str,
        embedding_model: str = "google/embeddinggemma-300m",
        device: str = "cuda:0",
        beta: float = 5.0,
        retrieval_threshold: float = 0.30,
        enable_near_duplicate: bool = False,
        embedder: Optional[BaseEmbedder] = None,
        allow_hash_fallback: bool = False,
    ) -> None:
        self.kb_dir = kb_dir
        self.beta = beta
        self.retrieval_threshold = retrieval_threshold
        self.enable_near_duplicate = enable_near_duplicate

        os.makedirs(kb_dir, exist_ok=True)
        self.dm_items_path = os.path.join(kb_dir, "dm_items.jsonl")
        self.emb_goal_path = os.path.join(kb_dir, "emb_goal.npy")
        self.emb_condition_path = os.path.join(kb_dir, "emb_condition.npy")
        self.meta_path = os.path.join(kb_dir, "meta.json")

        if embedder is not None:
            self.embedder = embedder
            model_name_record = type(embedder).__name__
            dim_record = getattr(embedder, "dim", None)
        else:
            try:
                self.embedder = HFEmbedder(model_name=embedding_model, device=device)
                model_name_record = embedding_model
                dim_record = self.embedder.dim
            except Exception as exc:
                if not allow_hash_fallback:
                    raise RuntimeError(
                        "Failed to load HFEmbedder and hash fallback is disabled. "
                        "Install runtime deps and model access, or set kb.allow_hash_fallback=true explicitly."
                    ) from exc
                self.embedder = HashEmbedder(dim=384)
                model_name_record = "HashEmbedder"
                dim_record = self.embedder.dim

        self._model_name_record = model_name_record
        self._dim_record = dim_record

        self.dm_items: List[DMItem] = []
        self.emb_goal = np.zeros((0, int(dim_record or 384)), dtype=np.float32)
        self.emb_condition = np.zeros((0, int(dim_record or 384)), dtype=np.float32)
        self._load()

    def _embedder_dim(self) -> Optional[int]:
        dim = getattr(self.embedder, "dim", None)
        if dim is None:
            return None
        return int(dim)

    def _load(self) -> None:
        if os.path.exists(self.dm_items_path):
            with open(self.dm_items_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    self.dm_items.append(
                        DMItem(
                            dm_id=obj["dm_id"],
                            dm_text=obj["dm_text"],
                            dm_goal_text=obj["dm_goal_text"],
                            dm_condition_text=obj["dm_condition_text"],
                            metadata=obj.get("metadata", {}),
                        )
                    )
        if os.path.exists(self.emb_goal_path):
            self.emb_goal = np.load(self.emb_goal_path).astype(np.float32)
        if os.path.exists(self.emb_condition_path):
            self.emb_condition = np.load(self.emb_condition_path).astype(np.float32)

        expected_dim = self._embedder_dim()
        need_rebuild = False
        if len(self.dm_items) != self.emb_goal.shape[0] or len(self.dm_items) != self.emb_condition.shape[0]:
            need_rebuild = bool(self.dm_items)
        elif self.dm_items:
            if self.emb_goal.ndim != 2 or self.emb_condition.ndim != 2:
                need_rebuild = True
            elif self.emb_goal.shape[1] != self.emb_condition.shape[1]:
                need_rebuild = True
            elif expected_dim is not None and self.emb_goal.shape[1] != expected_dim:
                need_rebuild = True
        else:
            if expected_dim is not None:
                self.emb_goal = np.zeros((0, expected_dim), dtype=np.float32)
                self.emb_condition = np.zeros((0, expected_dim), dtype=np.float32)

        if need_rebuild:
            self._rebuild_embeddings()

    def _rebuild_embeddings(self) -> None:
        goal_texts = [it.dm_goal_text for it in self.dm_items]
        cond_texts = [it.dm_condition_text for it in self.dm_items]
        self.emb_goal = self.embedder.embed(goal_texts)
        self.emb_condition = self.embedder.embed(cond_texts)
        self._persist_embeddings()

    def _persist_embeddings(self) -> None:
        np.save(self.emb_goal_path, self.emb_goal.astype(np.float32))
        np.save(self.emb_condition_path, self.emb_condition.astype(np.float32))
        meta = {
            "embedding_model": self._model_name_record,
            "dim": int(self.emb_goal.shape[1] if self.emb_goal.size else (self._dim_record or 0)),
            "beta": self.beta,
            "retrieval_threshold": self.retrieval_threshold,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _next_dm_id(self) -> str:
        return f"dm_{len(self.dm_items) + 1:08d}"

    def _score_vectors(self, q_goal: np.ndarray, q_cond: np.ndarray) -> np.ndarray:
        if self.emb_goal.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        x = self.emb_goal @ q_goal
        y = self.emb_condition @ q_cond
        beta = self.beta
        score = (1.0 / beta) * np.log(np.exp(beta * x) + np.exp(beta * y))
        return score.astype(np.float32)

    def query_topk(self, goal_text: str, thinking_text: str, k: int = 5) -> Dict:
        if not self.dm_items:
            return {"items": []}
        q_goal = self.embedder.embed([goal_text])[0]
        q_cond = self.embedder.embed([thinking_text])[0]
        if self.emb_goal.shape[1] != q_goal.shape[0] or self.emb_condition.shape[1] != q_cond.shape[0]:
            # Existing KB vectors were built with a different embedding dimensionality.
            self._rebuild_embeddings()
            q_goal = self.embedder.embed([goal_text])[0]
            q_cond = self.embedder.embed([thinking_text])[0]
        x_all = self.emb_goal @ q_goal
        y_all = self.emb_condition @ q_cond
        score_all = (1.0 / self.beta) * np.log(np.exp(self.beta * x_all) + np.exp(self.beta * y_all))

        order = sorted(
            range(len(self.dm_items)),
            key=lambda i: (-float(score_all[i]), self.dm_items[i].dm_id),
        )
        out_items = []
        for i in order[: max(0, k)]:
            it = self.dm_items[i]
            out_items.append(
                {
                    "dm_id": it.dm_id,
                    "score": float(score_all[i]),
                    "x_goal": float(x_all[i]),
                    "y_condition": float(y_all[i]),
                    "dm_text": it.dm_text,
                    "dm_goal_text": it.dm_goal_text,
                    "dm_condition_text": it.dm_condition_text,
                }
            )
        return {"items": out_items}

    def retrieve_top1(self, goal_text: str, thinking_text: str) -> Optional[Dict]:
        res = self.query_topk(goal_text=goal_text, thinking_text=thinking_text, k=1)
        if not res["items"]:
            return None
        top = res["items"][0]
        if top["score"] < self.retrieval_threshold:
            return None
        return top

    def score(
        self,
        goal_text: str,
        thinking_text: str,
        candidate_dm_goal_text: str,
        candidate_dm_condition_text: str,
    ) -> Dict:
        q_goal = self.embedder.embed([goal_text])[0]
        q_cond = self.embedder.embed([thinking_text])[0]
        c_goal = self.embedder.embed([candidate_dm_goal_text])[0]
        c_cond = self.embedder.embed([candidate_dm_condition_text])[0]
        x = float(np.dot(q_goal, c_goal))
        y = float(np.dot(q_cond, c_cond))
        score = (1.0 / self.beta) * math.log(math.exp(self.beta * x) + math.exp(self.beta * y))
        return {"score": float(score), "x_goal": x, "y_condition": y, "beta": self.beta}

    def commit_dm_candidates(self, candidates: List[Dict]) -> Dict:
        added = []
        skipped = []

        existing_texts = {_normalize_text(it.dm_text) for it in self.dm_items}
        existing_gc = {
            (_normalize_text(it.dm_goal_text), _normalize_text(it.dm_condition_text)) for it in self.dm_items
        }

        new_goal_texts: List[str] = []
        new_cond_texts: List[str] = []
        new_items: List[DMItem] = []

        for idx, cand in enumerate(candidates):
            dm_text = _normalize_text(cand.get("dm_text", ""))
            dm_goal_text = _normalize_text(cand.get("dm_goal_text", ""))
            dm_condition_text = _normalize_text(cand.get("dm_condition_text", ""))
            if not dm_text or not dm_goal_text or not dm_condition_text:
                skipped.append({"reason": "missing_required_field", "candidate_index": idx})
                continue
            if dm_text in existing_texts:
                skipped.append({"reason": "exact_match_dm_text", "candidate_index": idx})
                continue
            if (dm_goal_text, dm_condition_text) in existing_gc:
                skipped.append({"reason": "exact_match_goal_condition", "candidate_index": idx})
                continue

            dm_id = self._next_dm_id()
            meta = dict(cand.get("metadata", {}))
            meta.setdefault("created_at", datetime.now(timezone.utc).isoformat())
            meta.setdefault("created_by", "teacher_llm")

            item = DMItem(
                dm_id=dm_id,
                dm_text=dm_text,
                dm_goal_text=dm_goal_text,
                dm_condition_text=dm_condition_text,
                metadata=meta,
            )
            self.dm_items.append(item)
            new_items.append(item)
            new_goal_texts.append(dm_goal_text)
            new_cond_texts.append(dm_condition_text)
            existing_texts.add(dm_text)
            existing_gc.add((dm_goal_text, dm_condition_text))
            added.append({"dm_id": dm_id})

        if new_items:
            with open(self.dm_items_path, "a", encoding="utf-8") as f:
                for item in new_items:
                    obj = {
                        "dm_id": item.dm_id,
                        "dm_text": item.dm_text,
                        "dm_goal_text": item.dm_goal_text,
                        "dm_condition_text": item.dm_condition_text,
                        "metadata": item.metadata,
                    }
                    f.write(json.dumps(obj, ensure_ascii=True) + "\n")

            emb_goal_new = self.embedder.embed(new_goal_texts)
            emb_cond_new = self.embedder.embed(new_cond_texts)
            rebuilt = False
            if self.emb_goal.shape[0] == 0 and self.emb_goal.shape[1] != emb_goal_new.shape[1]:
                self.emb_goal = np.zeros((0, emb_goal_new.shape[1]), dtype=np.float32)
                self.emb_condition = np.zeros((0, emb_cond_new.shape[1]), dtype=np.float32)
            elif self.emb_goal.shape[1] != emb_goal_new.shape[1] or self.emb_condition.shape[1] != emb_cond_new.shape[1]:
                self._rebuild_embeddings()
                rebuilt = True
            if not rebuilt:
                self.emb_goal = np.vstack([self.emb_goal, emb_goal_new])
                self.emb_condition = np.vstack([self.emb_condition, emb_cond_new])
                self._persist_embeddings()

        return {"added": added, "skipped_duplicates": skipped, "kb_size": len(self.dm_items)}

    def snapshot(self, out_path: str) -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            for item in self.dm_items:
                f.write(
                    json.dumps(
                        {
                            "dm_id": item.dm_id,
                            "dm_text": item.dm_text,
                            "dm_goal_text": item.dm_goal_text,
                            "dm_condition_text": item.dm_condition_text,
                            "metadata": item.metadata,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )


class OverlayKBServer:
    """Read-only view over global KB + per-problem staged DM candidates."""

    def __init__(self, base_kb: KBServer, staged_candidates: List[Dict]) -> None:
        self.base = base_kb
        self.beta = base_kb.beta
        self.retrieval_threshold = base_kb.retrieval_threshold
        self.embedder = base_kb.embedder

        self.dm_items: List[DMItem] = list(base_kb.dm_items)
        self.emb_goal = base_kb.emb_goal
        self.emb_condition = base_kb.emb_condition

        clean = []
        for i, cand in enumerate(staged_candidates):
            dm_text = _normalize_text(cand.get("dm_text", ""))
            dm_goal_text = _normalize_text(cand.get("dm_goal_text", ""))
            dm_condition_text = _normalize_text(cand.get("dm_condition_text", ""))
            if not dm_text or not dm_goal_text or not dm_condition_text:
                continue
            clean.append(
                DMItem(
                    dm_id=f"stage_{i+1:05d}",
                    dm_text=dm_text,
                    dm_goal_text=dm_goal_text,
                    dm_condition_text=dm_condition_text,
                    metadata=dict(cand.get("metadata", {})),
                )
            )

        if clean:
            goal_texts = [it.dm_goal_text for it in clean]
            cond_texts = [it.dm_condition_text for it in clean]
            emb_goal_new = self.embedder.embed(goal_texts)
            emb_cond_new = self.embedder.embed(cond_texts)
            self.dm_items.extend(clean)
            self.emb_goal = np.vstack([self.emb_goal, emb_goal_new])
            self.emb_condition = np.vstack([self.emb_condition, emb_cond_new])

    def query_topk(self, goal_text: str, thinking_text: str, k: int = 5) -> Dict:
        if not self.dm_items:
            return {"items": []}
        q_goal = self.embedder.embed([goal_text])[0]
        q_cond = self.embedder.embed([thinking_text])[0]
        x_all = self.emb_goal @ q_goal
        y_all = self.emb_condition @ q_cond
        score_all = (1.0 / self.beta) * np.log(np.exp(self.beta * x_all) + np.exp(self.beta * y_all))

        order = sorted(
            range(len(self.dm_items)),
            key=lambda i: (-float(score_all[i]), self.dm_items[i].dm_id),
        )
        out_items = []
        for i in order[: max(0, k)]:
            it = self.dm_items[i]
            out_items.append(
                {
                    "dm_id": it.dm_id,
                    "score": float(score_all[i]),
                    "x_goal": float(x_all[i]),
                    "y_condition": float(y_all[i]),
                    "dm_text": it.dm_text,
                    "dm_goal_text": it.dm_goal_text,
                    "dm_condition_text": it.dm_condition_text,
                }
            )
        return {"items": out_items}

    def retrieve_top1(self, goal_text: str, thinking_text: str) -> Optional[Dict]:
        res = self.query_topk(goal_text=goal_text, thinking_text=thinking_text, k=1)
        if not res["items"]:
            return None
        top = res["items"][0]
        if top["score"] < self.retrieval_threshold:
            return None
        return top

    def score(
        self,
        goal_text: str,
        thinking_text: str,
        candidate_dm_goal_text: str,
        candidate_dm_condition_text: str,
    ) -> Dict:
        return self.base.score(
            goal_text=goal_text,
            thinking_text=thinking_text,
            candidate_dm_goal_text=candidate_dm_goal_text,
            candidate_dm_condition_text=candidate_dm_condition_text,
        )
