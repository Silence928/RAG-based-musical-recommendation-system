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

from src.utils import load_config
from src.llm_backend import create_llm
from src.retrieval.indexer import PreferenceIndex, KnowledgeBaseIndex
from src.retrieval.retriever import MeloRetriever
from src.generation.generator import MeloGenerator
from src.post_rank.global_prior import (
    build_global_quality_priors,
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
    global_priors = build_global_quality_priors(
        pos_records=pos_index.records,
        neg_records=neg_index.records,
        kb_records=kb_index.records,
        quality_weight=float(prior_cfg.get("quality_weight", 0.7)),
        rarity_weight=float(prior_cfg.get("rarity_weight", 0.3)),
        neg_penalty=float(prior_cfg.get("neg_penalty", 0.4)),
    )

    return config, retriever, generator, allowed_names, global_priors


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


def _format_evidence_quotes(rec: dict, fallback_evidence: list[str]) -> str:
    quotes = rec.get("evidence_quotes", [])
    if isinstance(quotes, list):
        quotes = [q.strip() for q in quotes if isinstance(q, str) and q.strip()]
    else:
        quotes = []

    if not quotes:
        quotes = fallback_evidence[:2]
    else:
        quotes = quotes[:2]

    if not quotes:
        return "（未返回可用证据）"
    return "；".join([f"“{q}”" for q in quotes])


def _collect_retrieval_evidence(retrieved: dict, limit: int = 4) -> list[str]:
    evidences: list[str] = []
    for record, _score in retrieved.get("positive_pairs", []):
        reason = str(record.get("reason", "")).strip()
        if reason:
            evidences.append(reason[:80])
        if len(evidences) >= limit:
            break
    return evidences


def _format_recommendation_reply(result: dict, retrieved: dict, user: dict) -> str:
    recs = result.get("recommendations", [])
    if not recs:
        return "暂时没有生成有效推荐，请换一种表达再试一次。"

    likes = [x.get("musical", "") for x in user.get("likes", []) if x.get("musical")]
    dislikes = [x.get("musical", "") for x in user.get("dislikes", []) if x.get("musical")]

    lines = []
    lines.append("我先提取到你的偏好：")
    lines.append(f"- 喜欢：{', '.join(likes) if likes else '未识别到明确剧名'}")
    lines.append(f"- 不喜欢/避雷：{', '.join(dislikes) if dislikes else '未识别到明确剧名'}")
    lines.append("")
    lines.append("已根据你的描述生成推荐（已加入全局质量权重重排）：")

    fallback_evidence = _collect_retrieval_evidence(retrieved)
    for idx, rec in enumerate(recs, start=1):
        musical = rec.get("musical", "未知剧目")
        reason = rec.get("reason", "").strip() or "（无理由）"
        evidence = _format_evidence_quotes(rec, fallback_evidence)
        quality = rec.get("_prior_quality", 0.5)
        mentions = rec.get("_prior_mentions", 0)
        lines.append(f"{idx}. {musical}")
        lines.append(f"   推荐理由：{reason}")
        lines.append(f"   真实评论证据：{evidence}")
        lines.append(
            f"   质量先验：quality={quality:.2f}, mentions={mentions}, final={rec.get('_rerank_score', 0.0):.2f}"
        )

    if fallback_evidence:
        lines.append("")
        lines.append("检索到的相似用户真实评论片段（节选）：")
        for quote in fallback_evidence[:3]:
            lines.append(f"- {quote}")
    return "\n".join(lines)


def chat_recommend(message: str, history: list[tuple[str, str]]):
    """Single-turn chat handler: free-text preferences -> recommendations."""
    text = (message or "").strip()
    if not text:
        return "", history

    if RESOURCES is None:
        raise RuntimeError("Demo resources are not initialized. Please restart demo/app.py.")

    config, retriever, generator, allowed_names, global_priors = RESOURCES
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
            base_rank_weight=float(prior_cfg.get("base_rank_weight", 0.6)),
            prior_weight=float(prior_cfg.get("prior_weight", 0.4)),
            attach_debug_fields=True,
        )
        reply = _format_recommendation_reply(output, retrieved, user)
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
