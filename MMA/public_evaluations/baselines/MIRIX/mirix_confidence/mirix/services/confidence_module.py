import threading
import os

from typing import Any, Dict, List, Optional, Sequence, Tuple
import datetime as dt
from math import sqrt
from rapidfuzz import fuzz
from sqlalchemy import select
from mirix.settings import settings
from mirix.helpers.converters import deserialize_vector

class ConfidenceModule:
    def __init__(self, config_path: Optional[str] = None):
        self.w_source = settings.confidence_source_weight
        self.w_time = settings.confidence_time_weight
        self.w_consensus = settings.confidence_consensus_weight
        # 公式模式：tri(默认三项), st(来源+时间), tc(时间+共识), cs(共识+来源)
        self.formula_mode = "tri"
        self.link_emb_w = settings.link_embedding_weight
        self.link_text_w = settings.link_text_weight
        self.link_top_k = settings.link_top_k
        self.consensus_top_k = settings.consensus_top_k
        self.half_life_days = settings.time_half_life_days
        self.link_min_weight = settings.link_min_weight

        self.embedding_fields_map = {
            "semantic_memory": ["details_embedding", "name_embedding", "summary_embedding"],
            "episodic_memory": ["details_embedding", "summary_embedding"],
            "resource_memory": ["summary_embedding"],
            "procedural_memory": ["steps_embedding", "summary_embedding"],
            "knowledge_vault": ["caption_embedding"],
        }
        
        # default source score mapping; can be overridden by config
        self.source_scores = {
            "user": 1.0,
            "knowledge": 0.9,
            "web": 0.7,
            "system": 0.6,
            "model": 0.6,
            "import": 0.5,
        }
        if config_path:
            self._apply_config_overrides(config_path)
        # 环境变量覆盖（优先级最高）
        env_mode = os.getenv("MIRIX_CONFIDENCE_FORMULA", "").strip().lower()
        if env_mode:
            self.formula_mode = self._resolve_formula_mode(env_mode) or self.formula_mode
    
    def _apply_config_overrides(self, config_path: str) -> None:
        try:
            cfg = None
            if os.path.exists(config_path):
                # try JSON
                try:
                    import json
                    with open(config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                except Exception:
                    # try YAML if available; otherwise fallback to Python literal
                    try:
                        import yaml  # type: ignore
                        with open(config_path, "r", encoding="utf-8") as f:
                            cfg = yaml.safe_load(f)
                    except Exception:
                        import ast
                        with open(config_path, "r", encoding="utf-8") as f:
                            cfg = ast.literal_eval(f.read())
            if not isinstance(cfg, dict):
                return
            weights = cfg.get("weights") or {}
            
            # support compact keys
            self.w_source = float(weights.get("w_s", self.w_source))
            self.w_time = float(weights.get("w_t", self.w_time))
            self.w_consensus = float(weights.get("w_c", self.w_consensus))
            # 公式模式可由配置覆盖
            raw_mode = (
                cfg.get("formula_mode")
                or cfg.get("confidence_formula")
                or cfg.get("confidence_formula_mode")
            )
            if isinstance(raw_mode, str) and raw_mode.strip():
                resolved = self._resolve_formula_mode(raw_mode.strip().lower())
                if resolved:
                    self.formula_mode = resolved
            
            # optional fine-tuning knobs; keep defaults if missing
            self.link_emb_w = float(cfg.get("link_embedding_weight", self.link_emb_w))
            self.link_text_w = float(cfg.get("link_text_weight", self.link_text_w))
            self.link_top_k = int(cfg.get("link_top_k", self.link_top_k))
            self.consensus_top_k = int(cfg.get("consensus_top_k", self.consensus_top_k))
            self.half_life_days = float(cfg.get("time_half_life_days", self.half_life_days))
            self.link_min_weight = float(cfg.get("link_min_weight", self.link_min_weight))
            
            src_scores = cfg.get("source_scores")
            if isinstance(src_scores, dict) and src_scores:
                self.source_scores = {str(k).lower(): float(v) for k, v in src_scores.items()}
                
            ef_map = cfg.get("confidence_embedding_fields_map")
            if isinstance(ef_map, dict):
                normalized = {}
                for k, v in ef_map.items():
                    if isinstance(v, (list, tuple)):
                        normalized[str(k).lower()] = [str(x) for x in v]
                if normalized:
                    self.embedding_fields_map.update(normalized)
        
        except Exception:
            # swallow errors; fallback to defaults
            return

    def _resolve_formula_mode(self, raw: str) -> Optional[str]:
        alias = {
            "tri": "tri",
            "st": "st",
            "source_time": "st",
            "time_source": "st",
            "tc": "tc",
            "time_consensus": "tc",
            "consensus_time": "tc",
            "cs": "cs",
            "consensus_source": "cs",
            "source_consensus": "cs",
        }
        return alias.get(raw)

    def _get_attr(self, obj, name, default=None):
        if hasattr(obj, name):
            return getattr(obj, name)
        try:
            return obj.get(name, default)  # type: ignore
        except Exception:
            return default

    def _parse_dt(self, value) -> Optional[dt.datetime]:
        if value is None:
            return None
        if isinstance(value, dt.datetime):
            return value
        try:
            parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed
        except Exception:
            return None

    def _source_score(self, item) -> float:
        src = str(self._get_attr(item, "source", "") or "").lower()
        mapping = self.source_scores
        for k, v in mapping.items():
            if k in src:
                return v
        return 0.6

    def _time_score(self, item) -> float:
        created = self._get_attr(item, "created_at") or self._get_attr(item, "occurred_at")
        created_dt = self._parse_dt(created)
        
        if not created_dt:
            return 0.5
        
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=dt.timezone.utc)
            
        now = dt.datetime.now(dt.timezone.utc)
        age_days = max(0.0, (now - created_dt).total_seconds() / 86400.0)
        decay = 0.5 ** (age_days / max(self.half_life_days, 1e-3))
        return decay

    def _cosine_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b:
            return 0.0
        def norm(x):
            return sqrt(sum(v * v for v in x))
        na = norm(a)
        nb = norm(b)
        if na <= 1e-9 or nb <= 1e-9:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        return max(0.0, min(1.0, dot / (na * nb)))

    def _text_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return fuzz.ratio(a, b) / 100.0

    def _select_embedding(self, item, fields: List[str]) -> Optional[List[float]]:
        for f in fields:
            emb = self._get_attr(item, f)
            if emb is None:
                continue
            if isinstance(emb, list):
                return [float(x) for x in emb]
            if isinstance(emb, tuple):
                return [float(x) for x in list(emb)]
            if isinstance(emb, str):
                try:
                    vec = deserialize_vector(emb)
                    return vec if isinstance(vec, list) else None
                except Exception:
                    try:
                        import json
                        if emb.strip().startswith("["):
                            return json.loads(emb)
                        return [float(x.strip()) for x in emb.split(",") if x.strip()]
                    except Exception:
                        continue
        return None

    def _flatten_text_value(self, v) -> str:
        try:
            parts: List[str] = []
            def flatten(x):
                if x is None:
                    return
                if isinstance(x, dict):
                    for k, val in x.items():
                        parts.append(str(k))
                        flatten(val)
                elif isinstance(x, (list, tuple, set)):
                    for it in x:
                        flatten(it)
                else:
                    parts.append(str(x))
            flatten(v)
            return " ".join(p for p in parts if p.strip())
        except Exception:
            return str(v)
            
    def _select_text(self, item, fields: List[str]) -> str:
        parts: List[str] = []
        for f in fields:
            v = self._get_attr(item, f)
            if not v:
                continue
            if isinstance(v, (dict, list, tuple, set)):
                parts.append(self._flatten_text_value(v))
            elif isinstance(v, str):
                if f == "metadata_" and v.strip() and (v.strip().startswith("{") or v.strip().startswith("[")):
                    try:
                        import json
                        parsed = json.loads(v)
                        parts.append(self._flatten_text_value(parsed))
                    except Exception:
                        parts.append(v)
                else:
                    parts.append(v)
            else:
                parts.append(str(v))
        return "\n".join(parts)

    def compute_v1(self, item) -> float:
        s = self._source_score(item)
        t = self._time_score(item)
        v1 = self.w_source * s + self.w_time * t
        return max(0.0, min(1.0, v1))

    def generate_and_store_links(self, session, target_class, item, organization_id, embedding_fields, text_fields) -> Dict[str, Any]:
        query = select(target_class)
        if organization_id:
            query = query.where(getattr(target_class, "organization_id") == organization_id)
        rows = session.execute(query).scalars().all()

        this_id = self._get_attr(item, "id")
        this_emb = self._select_embedding(item, embedding_fields)
        this_text = self._select_text(item, text_fields)

        scored: List[Tuple[str, float, str]] = []
        for r in rows:
            rid = self._get_attr(r, "id")
            if rid == this_id:
                continue
            emb = self._select_embedding(r, embedding_fields)
            txt = self._select_text(r, text_fields)

            emb_sim = self._cosine_similarity(this_emb or [], emb or [])
            txt_sim = self._text_similarity(this_text, txt)
            weight = self.link_emb_w * emb_sim + self.link_text_w * txt_sim
            if weight >= self.link_min_weight:
                reason = "embedding+text" if this_emb and emb else ("text" if txt_sim > 0 else "none")
                scored.append((rid, weight, reason))

        scored.sort(key=lambda x: x[1], reverse=True)
        neighbors = [
            {
                "target_id": rid,
                "weight": round(w, 6),
                "type": "similarity",
                "reason": rsn,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            for rid, w, rsn in scored[: self.link_top_k]
        ]

        links = {"neighbors": neighbors, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "org": organization_id}

        try:
            setattr(item, "links", links)
            item.update(session)
        except Exception:
            pass

        return links

    def generate_links_from_candidates(self, session, target_class, item, candidates, embedding_fields, text_fields) -> Dict[str, Any]:
        this_id = getattr(item, "id", None)
        this_emb = self._select_embedding(item, embedding_fields)
        this_text = self._select_text(item, text_fields)

        scored = []
        for r in candidates:
            rid = getattr(r, "id", None)
            if rid == this_id:
                continue
            emb = self._select_embedding(r, embedding_fields)
            txt = self._select_text(r, text_fields)

            emb_sim = self._cosine_similarity(this_emb or [], emb or [])
            txt_sim = self._text_similarity(this_text, txt)
            weight = self.link_emb_w * emb_sim + self.link_text_w * txt_sim
            if weight >= self.link_min_weight:
                reason = "embedding+text" if this_emb and emb else ("text" if txt_sim > 0 else "none")
                scored.append((rid, weight, reason))

        scored.sort(key=lambda x: x[1], reverse=True)
        neighbors = [
            {
                "target_id": rid,
                "weight": round(w, 6),
                "type": "similarity",
                "reason": rsn,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            for rid, w, rsn in scored[: self.link_top_k]
        ]

        links = {"neighbors": neighbors, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "org": getattr(item, "organization_id", None)}

        try:
            setattr(item, "links", links)
            item.update(session)
        except Exception:
            pass

        return links

    def compute_consensus(self, session, target_class, item) -> float:
        links = self._get_attr(item, "links", {}) or {}
        neighbors = links.get("neighbors") or []
        if not neighbors:
            return 0.0

        neighbors = neighbors[: self.consensus_top_k]
        id_to_weight = {n["target_id"]: float(n.get("weight", 0.0)) for n in neighbors if n.get("target_id")}
        if not id_to_weight:
            return 0.0

        query = select(target_class).where(getattr(target_class, "id").in_(list(id_to_weight.keys())))
        rows = session.execute(query).scalars().all()

        emb_fields = self._embedding_fields_for_class(target_class)
        item_emb = self._select_embedding(item, emb_fields)

        total_w = 0.0
        acc = 0.0
        for r in rows:
            cid = self._get_attr(r, "id")
            conf = float(self._get_attr(r, "confidence", 0.0) or 0.0)
            w = float(id_to_weight.get(cid, 0.0))
            total_w += w

            support_factor = 1.0
            if item_emb:
                neighbor_emb = self._select_embedding(r, emb_fields)
                if neighbor_emb:
                    support_factor = self._cosine_similarity_signed(item_emb, neighbor_emb)

            acc += w * conf * support_factor

        if total_w <= 1e-9:
            return 0.0
        consensus = acc / total_w
        
        if consensus > 1.0:
            return 1.0
        if consensus < -1.0:
            return -1.0
        return consensus
        
    def _cosine_similarity_signed(self, a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b:
            return 0.0
        def norm(x):
            return sqrt(sum(v * v for v in x))
        na = norm(a)
        nb = norm(b)
        if na <= 1e-9 or nb <= 1e-9:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        val = dot / (na * nb)
        if val > 1.0:
            return 1.0
        if val < -1.0:
            return -1.0
        return val

    def _class_key(self, target_class) -> str:
        try:
            tbl = getattr(target_class, "__tablename__", None)
            if tbl:
                return str(tbl).lower()
        except Exception:
            pass
        try:
            name = getattr(target_class, "__name__", None)
            if name:
                return str(name).lower()
        except Exception:
            pass
        return ""
        
    def _embedding_fields_for_class(self, target_class) -> List[str]:
        key = self._class_key(target_class)
        if key in self.embedding_fields_map:
            return self.embedding_fields_map[key]
        for alias in (key, key.replace("_memory", ""), key + "_memory"):
            if alias in self.embedding_fields_map:
                return self.embedding_fields_map[alias]
        return ["summary_embedding", "details_embedding", "name_embedding", "steps_embedding", "caption_embedding"]
    
    def compute_v2(self, session, target_class, item) -> float:
        """
        Computes the final confidence score using a self-normalizing weighted sum of 
        source, time, and consensus.
        
        支持消融模式：
        - tri: 三项加权（默认）
        - st: 仅来源+时间，权重归一化到两项
        - tc: 仅时间+共识，权重归一化到两项
        - cs: 仅共识+来源，权重归一化到两项
        """
        s = self._source_score(item)
        t = self._time_score(item)
        consensus = self.compute_consensus(session, target_class, item)
    
        w_s = float(self.w_source)
        w_t = float(self.w_time)
        w_c = float(self.w_consensus)
    
        mode = (self.formula_mode or "tri").lower()
        if mode == "st":
            total = w_s + w_t
            if total <= 1e-9:
                return 0.0
            w_s_norm = w_s / total
            w_t_norm = w_t / total
            v_raw = w_s_norm * s + w_t_norm * t
            return max(0.0, min(1.0, v_raw))
        elif mode == "tc":
            total = w_t + w_c
            if total <= 1e-9:
                return 0.0
            w_t_norm = w_t / total
            w_c_norm = w_c / total
            v_raw = w_t_norm * t + w_c_norm * consensus
            return max(0.0, min(1.0, v_raw))
        elif mode == "cs":
            total = w_c + w_s
            if total <= 1e-9:
                return 0.0
            w_c_norm = w_c / total
            w_s_norm = w_s / total
            v_raw = w_c_norm * consensus + w_s_norm * s
            return max(0.0, min(1.0, v_raw))
        else:
            total_weight = w_s + w_t + w_c
            if total_weight <= 1e-9:
                return 0.0 
            w_s_norm = w_s / total_weight
            w_t_norm = w_t / total_weight
            w_c_norm = w_c / total_weight
            v2_raw = w_s_norm * s + w_t_norm * t + w_c_norm * consensus
            return max(0.0, min(1.0, v2_raw))

_singleton: Optional[ConfidenceModule] = None
_lock = threading.Lock()

def get_confidence_module() -> ConfidenceModule:
    global _singleton
    if _singleton is None:
        initialize_confidence_module()
    return _singleton

def initialize_confidence_module(config_path: Optional[str] = None):
    global _singleton
    if _singleton is not None:
        return
    with _lock:
        if _singleton is not None:
            return
        if not config_path:
            config_path = os.getenv("MIRIX_CONFIDENCE_CONFIG") or os.path.join(
                os.path.dirname(__file__), "..", "..", "configs", "confidence_v2.yaml"
            )
        _singleton = ConfidenceModule(config_path=config_path)

def reset_confidence_module():
    global _singleton
    with _lock:
        _singleton = None