"""
app.py — Gradio chat-style demo for MeloMatch.

Users type free-form preference text in a chat box. The app converts the
input into lightweight structured preferences, runs retrieval + generation,
and returns recommended musicals with reasons.
"""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from src.utils import load_config, load_jsonl
from src.llm_backend import create_llm
from src.retrieval.indexer import PreferenceIndex, KnowledgeBaseIndex
from src.retrieval.retriever import MeloRetriever
from src.generation.generator import MeloGenerator
from src.post_rank.global_prior import (
    build_global_quality_priors,
    build_user_preference_maps,
    normalize_musical_name,
    rerank_with_global_priors,
)
from sentence_transformers import SentenceTransformer


# ======================== Load Resources ========================

def load_resources():
    """Load config, indices, encoder, and generator."""
    config = load_config(
        str(Path(__file__).resolve().parent.parent / "configs" / "config.yaml")
    )

    ret_config = config["retrieval"]
    gen_config = config["generation"]
    index_dir = config["paths"].get("index_dir", "data/indices")

    print("[demo] Loading encoder...")
    encoder = SentenceTransformer(ret_config["embedding_model"])

    print("[demo] Loading indices...")
    pos_index = PreferenceIndex("positive")
    pos_index.load(index_dir)
    neg_index = PreferenceIndex("negative")
    neg_index.load(index_dir)
    kb_index = KnowledgeBaseIndex("knowledge_base")
    kb_index.load(index_dir)

    retriever = MeloRetriever(pos_index, neg_index, kb_index, encoder)

    allowed_names = [r.get("name", "").strip() for r in kb_index.records if r.get("name")]

    # Use zero-shot model for demo
    model_cfg = gen_config["models"]["zero_shot"]
    llm = create_llm(model_cfg)
    generator = MeloGenerator(
        llm=llm,
        temperature=gen_config["temperature"],
        max_tokens=gen_config["max_tokens"],
        num_recommendations=gen_config["num_recommendations"],
        allowed_musical_names=allowed_names,
    )

    prior_cfg = config.get("post_rank", {}).get("global_prior", {})
    final_top_n = int(prior_cfg.get("final_top_n", 10))
    global_priors = build_global_quality_priors(
        pos_records=pos_index.records,
        neg_records=neg_index.records,
        kb_records=kb_index.records,
        quality_weight=float(prior_cfg.get("quality_weight", 0.7)),
        rarity_weight=float(prior_cfg.get("rarity_weight", 0.3)),
        neg_penalty=float(prior_cfg.get("neg_penalty", 0.4)),
        high_neg_ratio_threshold=float(prior_cfg.get("high_neg_ratio_threshold", 0.0)),
        high_neg_ratio_penalty=float(prior_cfg.get("high_neg_ratio_penalty", 0.0)),
    )
    users_path = Path(config["paths"]["processed_data"]) / "users.jsonl"
    users_data = load_jsonl(str(users_path))
    user_like_map, user_dislike_map = build_user_preference_maps(users_data)
    global_positive_bank = _build_global_positive_bank(pos_index.records, limit_per_musical=10)
    global_negative_bank = _build_global_negative_bank(neg_index.records, limit_per_musical=10)

    return (
        config,
        retriever,
        generator,
        allowed_names,
        global_priors,
        user_like_map,
        user_dislike_map,
        final_top_n,
        global_positive_bank,
        global_negative_bank,
    )


# ======================== Core Logic ========================

WELCOME_TEXT = (
    "您好！我可以根据主观感受为您推荐音乐剧。请输入您喜欢/不喜欢的具体剧目及理由或喜欢/不喜欢的类型，来获取推荐吧！"
)

POS_MARKERS = ("喜欢", "爱", "偏好", "想看", "打动", "共情")
NEG_MARKERS = ("不喜欢", "讨厌", "避雷", "雷点", "不想看", "不要推荐", "不爱看", "踩雷")


def _normalize_name(name: str) -> str:
    return normalize_musical_name(name)


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"[。！？!?\n；;]+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _build_name_aliases(allowed_names: list[str]) -> dict[str, str]:
    aliases = {}
    for name in allowed_names:
        norm = _normalize_name(name)
        if norm:
            aliases[norm] = name
    return aliases


def _build_global_positive_bank(pos_records: list[dict], limit_per_musical: int = 10) -> dict[str, list[str]]:
    """
    Build a global evidence bank from full positive corpus:
    musical -> liked-user reasons.
    """
    bank: dict[str, list[str]] = {}
    for row in pos_records or []:
        musical = str(row.get("musical", "")).strip()
        reason = str(row.get("reason", "")).strip()
        if not musical or not reason:
            continue
        key = _normalize_name(musical)
        bucket = bank.setdefault(key, [])
        if reason in bucket:
            continue
        if len(bucket) >= limit_per_musical:
            continue
        bucket.append(reason)
    return bank


def _build_global_negative_bank(neg_records: list[dict], limit_per_musical: int = 10) -> dict[str, list[str]]:
    """
    Build a global negative evidence bank from full negative corpus:
    musical -> disliked-user reasons.
    """
    bank: dict[str, list[str]] = {}
    for row in neg_records or []:
        musical = str(row.get("musical", "")).strip()
        reason = str(row.get("reason", "")).strip()
        if not musical or not reason:
            continue
        key = _normalize_name(musical)
        bucket = bank.setdefault(key, [])
        if reason in bucket:
            continue
        if len(bucket) >= limit_per_musical:
            continue
        bucket.append(reason)
    return bank


def _name_variants_for_evidence(name: str) -> list[str]:
    """
    Build normalized alias variants for loose evidence lookup.
    """
    base = _normalize_name(name)
    if not base:
        return []
    variants = {base}
    # Common wrappers/edition words in Chinese musical naming.
    for token in ("中文版", "中国版", "音乐剧", "韩版", "日版", "法版", "英版", "原版"):
        variants.add(base.replace(token, ""))
    variants = {v for v in variants if v}
    return sorted(variants, key=len, reverse=True)


def _lookup_evidence_with_alias(
    key_name: str,
    bank: dict[str, list],
) -> tuple[list, str]:
    """
    Try exact -> alias -> fuzzy containment lookup.
    """
    variants = _name_variants_for_evidence(key_name)
    if not variants:
        return [], "none"

    # 1) exact/alias direct hit
    for v in variants:
        if v in bank and bank[v]:
            return bank[v], "alias_direct"

    # 2) fuzzy containment hit
    for v in variants:
        if len(v) < 2:
            continue
        for k, items in bank.items():
            if not items:
                continue
            if v in k or k in v:
                return items, "alias_fuzzy"
    return [], "none"


def _extract_structured_preferences(user_text: str, allowed_names: list[str]) -> dict:
    """
    Parse free-form Chinese text into structured likes/dislikes:
    - Extract explicit musical mentions from KB names.
    - Classify sentence-level sentiment using positive/negative markers.
    - Keep original sentence fragments as reasons to preserve user voice.
    """
    text = (user_text or "").strip()
    if not text:
        return {"likes": [], "dislikes": []}

    sentences = _sentence_split(text)
    sorted_names = sorted(allowed_names, key=len, reverse=True)
    likes_map: dict[str, str] = {}
    dislikes_map: dict[str, str] = {}

    for sent in sentences:
        lower_sent = sent.lower()
        has_pos = any(m in sent for m in POS_MARKERS)
        has_neg = any(m in sent for m in NEG_MARKERS)

        matched_names = [name for name in sorted_names if name and name.lower() in lower_sent]
        matched_names = matched_names[:4]

        if matched_names:
            for name in matched_names:
                key = _normalize_name(name)
                if has_neg and not has_pos:
                    dislikes_map.setdefault(key, sent)
                elif has_pos and not has_neg:
                    likes_map.setdefault(key, sent)
                elif has_pos and has_neg:
                    if re.search(rf"(不喜欢|不要推荐|讨厌).{{0,8}}{re.escape(name.lower())}", lower_sent):
                        dislikes_map.setdefault(key, sent)
                    else:
                        likes_map.setdefault(key, sent)
                else:
                    likes_map.setdefault(key, sent)
            continue

        # Preferences without explicit musical names.
        if has_neg:
            dislikes_map.setdefault(_normalize_name("用户回避偏好"), sent)
        elif has_pos:
            likes_map.setdefault(_normalize_name("用户正向偏好"), sent)

    name_aliases = _build_name_aliases(allowed_names)

    likes = []
    for key, reason in likes_map.items():
        musical_name = name_aliases.get(key, "用户正向偏好")
        likes.append({"musical": musical_name, "reason": reason})

    dislikes = []
    for key, reason in dislikes_map.items():
        musical_name = name_aliases.get(key, "用户回避偏好")
        dislikes.append({"musical": musical_name, "reason": reason})

    if not likes and text:
        likes = [{"musical": "用户正向偏好", "reason": text}]

    return {"likes": likes[:6], "dislikes": dislikes[:6]}


def _build_demo_user_from_text(user_text: str, allowed_names: list[str]) -> dict:
    prefs = _extract_structured_preferences(user_text, allowed_names)
    return {
        "user_id": "demo_user",
        "likes": prefs["likes"],
        "dislikes": prefs["dislikes"],
        "meta": {},
    }


def _format_evidence_quotes(fallback_evidence: list[str]) -> str:
    """
    Evidence must come from retrieved real user comments.
    Do NOT trust/echo model-generated evidence_quotes here.
    """
    quotes = [q.strip() for q in fallback_evidence if isinstance(q, str) and q.strip()][:2]
    if not quotes:
        return "（未命中该剧的直接用户评论证据）"
    return "；".join([f"“{q}”" for q in quotes])


def _build_evidence_bank(retrieved: dict, limit_per_musical: int = 3) -> dict[str, list[dict]]:
    """
    Evidence grouped by musical, built from retrieved positive neighbors only.
    This ensures evidence means "users who liked this musical".
    Keep full reason text (no truncation) for better readability.
    """
    bank: dict[str, list[dict]] = {}
    positive_pairs = retrieved.get("positive_pairs", []) or []
    for record, _score in positive_pairs:
        musical = str(record.get("musical", "")).strip()
        reason = str(record.get("reason", "")).strip()
        if not musical or not reason:
            continue
        key = _normalize_name(musical)
        bucket = bank.setdefault(key, [])
        if len(bucket) >= limit_per_musical:
            continue
        if any(x.get("reason") == reason for x in bucket):
            continue
        bucket.append({"musical": musical, "reason": reason})
    return bank


def _build_negative_evidence_bank(retrieved: dict, limit_per_musical: int = 3) -> dict[str, list[dict]]:
    """
    Negative evidence grouped by musical, built from retrieved negative neighbors.
    """
    bank: dict[str, list[dict]] = {}
    negative_pairs = retrieved.get("negative_pairs", []) or []
    for record, _score in negative_pairs:
        musical = str(record.get("musical", "")).strip()
        reason = str(record.get("reason", "")).strip()
        if not musical or not reason:
            continue
        key = _normalize_name(musical)
        bucket = bank.setdefault(key, [])
        if len(bucket) >= limit_per_musical:
            continue
        if any(x.get("reason") == reason for x in bucket):
            continue
        bucket.append({"musical": musical, "reason": reason})
    return bank


def _collect_global_evidence(evidence_bank: dict[str, list[dict]], limit: int = 5) -> list[dict]:
    merged: list[dict] = []
    for items in evidence_bank.values():
        for item in items:
            merged.append(item)
            if len(merged) >= limit:
                return merged
    return merged


def _evidence_for_musical(
    musical: str,
    local_evidence_bank: dict[str, list[dict]],
    global_positive_bank: dict[str, list[str]],
) -> tuple[list[str], str]:
    local_items, local_hit = _lookup_evidence_with_alias(musical, local_evidence_bank)
    if local_items:
        return [x["reason"] for x in local_items[:2]], (
            "positive_pairs_local" if local_hit == "alias_direct" else "positive_pairs_local_fuzzy"
        )
    global_items, global_hit = _lookup_evidence_with_alias(musical, global_positive_bank)
    if global_items:
        return global_items[:2], (
            "positive_pairs_global" if global_hit == "alias_direct" else "positive_pairs_global_fuzzy"
        )
    return [], "none"


def _negative_evidence_for_musical(
    musical: str,
    local_negative_bank: dict[str, list[dict]],
    global_negative_bank: dict[str, list[str]],
) -> tuple[list[str], str]:
    local_items, local_hit = _lookup_evidence_with_alias(musical, local_negative_bank)
    if local_items:
        return [x["reason"] for x in local_items[:2]], (
            "negative_pairs_local" if local_hit == "alias_direct" else "negative_pairs_local_fuzzy"
        )
    global_items, global_hit = _lookup_evidence_with_alias(musical, global_negative_bank)
    if global_items:
        return global_items[:2], (
            "negative_pairs_global" if global_hit == "alias_direct" else "negative_pairs_global_fuzzy"
        )
    return [], "none"


def _is_generic_fallback_reason(reason: str) -> bool:
    return "基于检索到的知识库候选进行回退推荐" in (reason or "")


def _make_non_template_reason(rec: dict, evidence_for_this_musical: list[str], user: dict) -> str:
    """
    Replace generic fallback reason with a user-aware, evidence-grounded explanation.
    """
    raw_reason = str(rec.get("reason", "")).strip()
    if raw_reason and not _is_generic_fallback_reason(raw_reason):
        return raw_reason

    lead = evidence_for_this_musical[0].strip() if evidence_for_this_musical else ""
    if lead:
        return f"结合相似用户的真实评论，这部剧在你关注的情感共鸣与剧情表达上更贴合。参考片段：{lead}"

    likes = [x.get("musical", "") for x in user.get("likes", []) if x.get("musical") and x.get("musical") not in ("用户正向偏好", "偏好描述")]
    dislikes = [x.get("musical", "") for x in user.get("dislikes", []) if x.get("musical")]
    if likes or dislikes:
        return "该剧与您的喜欢/避雷方向整体更一致，并在检索结果中综合得分更高。"
    return "该剧在当前检索候选中的综合得分较高。"


def _candidate_level_summary(
    generator: MeloGenerator,
    musical: str,
    user: dict,
    positive_evidence_for_this_musical: list[str],
    negative_evidence_for_this_musical: list[str],
    fallback_reason: str,
) -> str:
    """
    Candidate-level summarization:
    - Summarize one candidate at a time.
    - Only use evidence matched to this candidate from retrieved positive pairs.
    - If evidence is missing, return evidence-insufficient explicitly.
    """
    positive_evidence = [
        e.strip() for e in positive_evidence_for_this_musical if isinstance(e, str) and e.strip()
    ][:2]
    negative_evidence = [
        e.strip() for e in negative_evidence_for_this_musical if isinstance(e, str) and e.strip()
    ][:2]

    if not positive_evidence and not negative_evidence:
        return "证据不足（未命中该剧的直接用户评论证据），暂不对该剧做强结论。"

    like_names = [x.get("musical", "") for x in user.get("likes", []) if x.get("musical")]
    dislike_names = [x.get("musical", "") for x in user.get("dislikes", []) if x.get("musical")]
    profile_line = (
        f"用户偏好剧目: {', '.join(like_names) if like_names else '无'}\n"
        f"用户避雷剧目: {', '.join(dislike_names) if dislike_names else '无'}"
    )

    system_prompt = (
        "你是音乐剧推荐解释助手。"
        "你只能基于给定证据写候选级总结，不得泛化。"
        "你必须在内部执行以下三步，但不要输出步骤过程："
        "第1步：从证据中提取可被直接支持的事实；"
        "第2步：判断哪些常见结论无法被证据支持；"
        "第3步：仅基于可支持事实生成最终理由。"
        "如果证据未明确支持某判断，必须写“证据不足”，不要编造。"
    )
    user_prompt = (
        f"候选剧目：{musical}\n"
        f"{profile_line}\n"
        "候选剧正向证据（喜欢该剧的用户评论，仅此可用）：\n"
        + ("\n".join([f"- {q}" for q in positive_evidence]) if positive_evidence else "- 无")
        + "\n\n候选剧负向证据（不喜欢该剧的用户评论，仅此可用）：\n"
        + ("\n".join([f"- {q}" for q in negative_evidence]) if negative_evidence else "- 无")
        + "\n\n写作约束：\n"
        "- 只输出 1-2 句中文，不要分点。\n"
        "- 先在内部完成“三步判断”，但不要输出思考过程或中间步骤。\n"
        "- 不得使用“剧情强”“情感共鸣”“逻辑清晰”等空泛词，除非证据原句明确提到对应信息。\n"
        "- 若证据不够支撑，必须包含“证据不足”四个字。\n"
        "- 禁止引用候选剧以外的证据。\n"
        "- 若负向证据明确包含剧情弱/逻辑混乱等风险，必须在理由中提示风险，不能只写正向结论。\n"
    )

    try:
        summary = generator.llm.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=160,
        ).strip()
        if not summary:
            return fallback_reason
        return summary
    except Exception:
        return fallback_reason


def _format_recommendation_reply(
    result: dict,
    retrieved: dict,
    user: dict,
    generator: MeloGenerator,
    global_positive_bank: dict[str, list[str]],
    global_negative_bank: dict[str, list[str]],
) -> str:
    recs = result.get("recommendations", [])
    if not recs:
        return "暂时没有生成有效推荐，请换一种表达再试一次。"

    likes = [
        x.get("musical", "")
        for x in user.get("likes", [])
        if x.get("musical") and x.get("musical") not in ("用户正向偏好", "偏好描述")
    ]
    dislikes = [
        x.get("musical", "")
        for x in user.get("dislikes", [])
        if x.get("musical") and x.get("musical") != "用户回避偏好"
    ]

    lines = []
    lines.append("我先提取到你的偏好：")
    lines.append(f"- 喜欢：{', '.join(likes) if likes else '未识别到明确剧名'}")
    lines.append(f"- 不喜欢/避雷：{', '.join(dislikes) if dislikes else '未识别到明确剧名'}")
    lines.append("")
    lines.append("已根据你的描述生成推荐（已加入全局质量权重重排）：")

    evidence_bank = _build_evidence_bank(retrieved)
    negative_evidence_bank = _build_negative_evidence_bank(retrieved)
    for idx, rec in enumerate(recs, start=1):
        musical = rec.get("musical", "未知剧目")
        per_musical_evidence, evidence_source = _evidence_for_musical(
            musical, evidence_bank, global_positive_bank
        )
        per_musical_negative_evidence, negative_evidence_source = _negative_evidence_for_musical(
            musical, negative_evidence_bank, global_negative_bank
        )
        fallback_reason = _make_non_template_reason(rec, per_musical_evidence, user)
        reason = _candidate_level_summary(
            generator,
            musical,
            user,
            per_musical_evidence,
            per_musical_negative_evidence,
            fallback_reason,
        )
        evidence = _format_evidence_quotes(per_musical_evidence)
        matched_count = len(per_musical_evidence)
        negative_matched_count = len(per_musical_negative_evidence)
        quality = rec.get("_prior_quality", 0.5)
        mentions = rec.get("_prior_mentions", 0)
        neg_ratio = rec.get("_prior_neg_ratio", 0.0)
        user_term = rec.get("_user_term", 0.0)
        dislike_penalty = rec.get("_dislike_hit_penalty", 0.0)
        lines.append(f"{idx}. {musical}")
        lines.append(f"   推荐理由：{reason}")
        lines.append(f"   真实评论证据：{evidence}")
        lines.append(f"   证据来源：evidence_source={evidence_source}, matched_count={matched_count}")
        if negative_matched_count > 0:
            lines.append(
                f"   负向风险证据：{_format_evidence_quotes(per_musical_negative_evidence)}"
            )
            lines.append(
                f"   风险证据来源：negative_evidence_source={negative_evidence_source}, negative_matched_count={negative_matched_count}"
            )
        lines.append(
            f"   打分分解：quality={quality:.2f}, mentions={mentions}, neg_ratio={neg_ratio:.2f}, user_term={user_term:.2f}, dislike_penalty={dislike_penalty:.2f}, final={rec.get('_rerank_score', 0.0):.2f}"
        )

    return "\n".join(lines)


def chat_recommend(message: str, history: list[tuple[str, str]]):
    """Single-turn chat handler: free-text preferences -> recommendations."""
    text = (message or "").strip()
    if not text:
        return "", history

    if RESOURCES is None:
        raise RuntimeError("Demo resources are not initialized. Please restart demo/app.py.")

    (
        config,
        retriever,
        generator,
        allowed_names,
        global_priors,
        user_like_map,
        user_dislike_map,
        final_top_n,
        global_positive_bank,
        global_negative_bank,
    ) = RESOURCES
    user = _build_demo_user_from_text(text, allowed_names)
    top_k = config["retrieval"]["default_top_k"]
    kb_top_m = config["retrieval"]["kb_top_m"]
    prior_cfg = config.get("post_rank", {}).get("global_prior", {})

    try:
        retrieved = retriever.retrieve(
            user=user,
            signal="dual",
            method="dense",
            top_k=top_k,
            kb_top_m=kb_top_m,
            exclude_user=False,
        )
        output = generator.generate(user, retrieved, signal="dual_enhanced")
        output["recommendations"] = rerank_with_global_priors(
            output.get("recommendations", []),
            global_priors,
            base_rank_weight=float(prior_cfg.get("base_rank_weight", 0.55)),
            prior_weight=float(prior_cfg.get("prior_weight", 0.20)),
            user_term_weight=float(prior_cfg.get("user_term_weight", 0.25)),
            dislike_hit_penalty=float(prior_cfg.get("dislike_hit_penalty", 0.0)),
            retrieved=retrieved,
            user_like_map=user_like_map,
            user_dislike_map=user_dislike_map,
            lambda_like=float(prior_cfg.get("lambda_like", 1.0)),
            lambda_dislike=float(prior_cfg.get("lambda_dislike", 1.2)),
            attach_debug_fields=True,
        )
        output["recommendations"] = output.get("recommendations", [])[:final_top_n]
        reply = _format_recommendation_reply(
            output, retrieved, user, generator, global_positive_bank, global_negative_bank
        )
    except Exception as e:
        reply = f"推荐流程出错：{e}"

    history = history + [(text, reply)]
    return "", history


# ======================== Gradio UI ========================

def build_ui():
    with gr.Blocks(
        title="MeloMatch — Musical Theatre Recommender",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# 🎭 MeloMatch 交互式推荐 Demo")
        gr.Markdown(WELCOME_TEXT)

        chatbot = gr.Chatbot(
            label="对话框",
            height=520,
            placeholder=WELCOME_TEXT,
        )
        with gr.Row():
            msg = gr.Textbox(
                label="请输入你的偏好",
                placeholder="例如：我喜欢法语音乐剧，偏剧情和舞美，不太喜欢通唱式和过于意识流的表达。",
                lines=3,
                scale=10,
            )
            send_btn = gr.Button("发送并推荐", variant="primary", scale=2)

        msg.submit(chat_recommend, inputs=[msg, chatbot], outputs=[msg, chatbot])
        send_btn.click(chat_recommend, inputs=[msg, chatbot], outputs=[msg, chatbot])

        gr.Markdown(
            "提示：为了提高推荐质量，尽量同时写出“喜欢/不喜欢”以及具体原因（剧情、音乐、舞美、卡司、语言风格等）。"
        )

    return demo


# ======================== Main ========================

RESOURCES = None

if __name__ == "__main__":
    print("[demo] Initializing MeloMatch...")
    RESOURCES = load_resources()
    demo = build_ui()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
