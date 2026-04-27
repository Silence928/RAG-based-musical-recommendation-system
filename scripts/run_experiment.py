"""
run_experiment.py — Run all experimental conditions (4 signals × 3 methods × 2 backbones = 24).

Signals: positive, dual, negative, dual_enhanced
Methods: bm25, dense, hybrid
Backbones: zero_shot, fine_tuned

New modules (Apr 2026):
  - Contrastive re-ranking (Seshadri et al., RecSys 2024)
  - NLI post-retrieval filtering (ARAG-inspired)
  - Structured preference profiling (DiscomfortFilter-inspired)
  - Local HuggingFace transformers inference (no API dependency)

Usage:
    cd E:/csc5051_final_proj
    python scripts/run_experiment.py [--config configs/config.yaml]
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from itertools import product
from datetime import datetime

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, load_jsonl, set_seed, setup_logging, save_run_metadata
from src.data.knowledge import load_knowledge_base
from src.data.splits import hold_out_one_like, stratified_user_subset
from src.retrieval.indexer import PreferenceIndex, KnowledgeBaseIndex
from src.retrieval.retriever import MeloRetriever
from src.generation.generator import MeloGenerator
from src.evaluation.metrics import evaluate_condition, save_metrics
from src.filtering.nli_filter import NLIFilter
from src.filtering.reranker import ContrastiveReranker
from src.profiles.preference_profiler import PreferenceProfiler
from src.llm_backend import create_llm, load_model_and_tokenizer
from src.domain_lexicon import load_domain_lexicon
from src.post_rank.global_prior import (
    build_global_quality_priors,
    rerank_with_global_priors,
)

from sentence_transformers import SentenceTransformer


def _load_backbone(model_cfg: dict, logger):
    """Load a single backbone model and return (model, tokenizer) or (None, None) for API/vllm."""
    backend = model_cfg.get("backend", "api")
    if backend == "local":
        logger.info(f"Loading local model: {model_cfg['name']} ({model_cfg.get('quantization', '4bit')})")
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_cfg["name"],
            quantization=model_cfg.get("quantization", "4bit"),
            lora_checkpoint=model_cfg.get("lora_checkpoint"),
        )
        return model, tokenizer
    # vllm and api backends are handled by create_llm() directly
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Run MeloMatch experiments")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--subset", type=int, default=None,
                        help="Run on N users only (data scaling ablation)")
    parser.add_argument("--ablation-k", action="store_true",
                        help="Run retrieval depth ablation (k in top_k_options)")
    parser.add_argument("--ablation-scale", action="store_true",
                        help="Run data scaling ablation (user_subsets)")
    args = parser.parse_args()

    config = load_config(args.config)
    exp = config["experiment"]
    gen_config = config["generation"]
    ret_config = config["retrieval"]
    seed = exp["seed"]

    set_seed(seed)

    signals = exp["signals"]
    methods = exp["methods"]
    backbones = exp["backbones"]

    conditions = list(product(signals, methods, backbones))

    # Setup run directory & logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    results_dir = Path(config["paths"]["results_dir"])
    run_dir = results_dir / run_name

    logger = setup_logging(str(results_dir), run_name)
    logger.info(f"{len(conditions)} conditions to run")
    for s, m, b in conditions:
        logger.info(f"  signal={s}, method={m}, backbone={b}")

    if args.dry_run:
        logger.info("Dry run — exiting.")
        return

    # Load data
    processed = config["paths"]["processed_data"]
    users = load_jsonl(f"{processed}/users.jsonl")
    kb_entries = load_knowledge_base(
        str(Path(config["paths"]["knowledge_base"]) / "musicals.jsonl")
    )
    allowed_musical_names = [e.get("name", "") for e in kb_entries if e.get("name")]

    # Data scaling ablation
    if args.subset:
        users = stratified_user_subset(users, args.subset, seed)
        logger.info(f"Subsampled to {len(users)} users (stratified)")
    else:
        logger.info(f"{len(users)} users loaded")

    # Save run metadata for reproducibility
    save_run_metadata(str(run_dir), config, extra={
        "n_users": len(users),
        "subset": args.subset,
        "conditions": [f"{s}_{m}_{b}" for s, m, b in conditions],
    })

    # Load encoder + indices
    logger.info(f"Loading encoder: {ret_config['embedding_model']}")
    encoder = SentenceTransformer(ret_config["embedding_model"])
    index_dir = config["paths"].get("index_dir", "data/indices")

    pos_index = PreferenceIndex("positive")
    pos_index.load(index_dir)
    neg_index = PreferenceIndex("negative")
    neg_index.load(index_dir)
    kb_index = KnowledgeBaseIndex("knowledge_base")
    kb_index.load(index_dir)

    lex_cfg = (
        config.get("text_expansion", {})
        .get("domain_lexicon", {})
    )
    lex_enabled = lex_cfg.get("enabled", False)
    lex_mode = lex_cfg.get("mode", "prompt_context")
    lex_path = lex_cfg.get("path", "data/raw/domain_lexicon.json")
    lex_max_exp = int(lex_cfg.get("max_expansions_per_text", 3))
    lexicon_entries = load_domain_lexicon(lex_path) if lex_enabled else []
    if lex_enabled:
        logger.info(
            f"Domain lexicon enabled (mode={lex_mode}): "
            f"{len(lexicon_entries)} entries from {lex_path}"
        )

    retriever = MeloRetriever(
        pos_index,
        neg_index,
        kb_index,
        encoder,
        lexicon_entries=lexicon_entries,
        max_lexicon_expansions=lex_max_exp,
        lexicon_mode=lex_mode,
    )

    # ---- Load LLM backbones ONCE (expensive GPU load) ----
    # With vLLM backend, models are loaded into GPU by vLLM engine.
    # With local backend, models are loaded via HuggingFace transformers.
    # Split deployment: zero_shot on Colab A100, fine_tuned on Kaggle T4×2.
    backbone_llms = {}
    backbone_models = {}  # Keep references for cleanup

    for backbone_name in backbones:
        model_cfg = gen_config["models"][backbone_name]
        model_cfg.setdefault("enable_thinking", False)
        backend = model_cfg.get("backend", "api")
        if backend == "local":
            model, tokenizer = _load_backbone(model_cfg, logger)
            llm = create_llm(model_cfg, model=model, tokenizer=tokenizer)
            backbone_models[backbone_name] = (model, tokenizer)
        else:
            # vllm or api — handled by create_llm()
            llm = create_llm(model_cfg)
        backbone_llms[backbone_name] = llm
        logger.info(
            f"Backbone '{backbone_name}' ready: {model_cfg['name']} ({backend}, "
            f"enable_thinking={model_cfg.get('enable_thinking', False)})"
        )

    # ---- New modules: contrastive reranker, NLI filter, preference profiler ----
    rerank_cfg = ret_config.get("contrastive_rerank", {})
    reranker = None
    if rerank_cfg.get("enabled", False):
        reranker = ContrastiveReranker(
            encoder=encoder,
            alpha=rerank_cfg.get("alpha", 0.3),
        )
        logger.info(f"Contrastive reranker enabled (alpha={reranker.alpha})")

    # NLI filter and profiler use the zero_shot backbone LLM (or any available)
    zs_llm = backbone_llms.get("zero_shot") or backbone_llms.get("fine_tuned")

    nli_cfg = gen_config.get("nli_filter", {})
    nli_filter = None
    if nli_cfg.get("enabled", False) and zs_llm:
        nli_filter = NLIFilter(
            llm=zs_llm,
            threshold=nli_cfg.get("threshold", 0.6),
        )
        logger.info(f"NLI filter enabled (threshold={nli_filter.threshold})")

    prof_cfg = gen_config.get("preference_profile", {})
    profiler = None
    if prof_cfg.get("enabled", False) and zs_llm:
        profiler = PreferenceProfiler(
            llm=zs_llm,
            use_llm=prof_cfg.get("use_llm", True),
        )
        logger.info("Preference profiler enabled")

    # Optional global-prior post-ranking (shared with demo for consistency)
    prior_cfg = config.get("post_rank", {}).get("global_prior", {})
    prior_enabled = prior_cfg.get("enabled", True)
    prior_quality_weight = float(prior_cfg.get("quality_weight", 0.7))
    prior_rarity_weight = float(prior_cfg.get("rarity_weight", 0.3))
    prior_neg_penalty = float(prior_cfg.get("neg_penalty", 0.4))
    prior_base_rank_weight = float(prior_cfg.get("base_rank_weight", 0.6))
    prior_blend_weight = float(prior_cfg.get("prior_weight", 0.4))
    global_priors = {}
    if prior_enabled:
        global_priors = build_global_quality_priors(
            pos_records=pos_index.records,
            neg_records=neg_index.records,
            kb_records=kb_index.records,
            quality_weight=prior_quality_weight,
            rarity_weight=prior_rarity_weight,
            neg_penalty=prior_neg_penalty,
        )
        logger.info(
            "Global-prior rerank enabled "
            f"(base={prior_base_rank_weight:.2f}, prior={prior_blend_weight:.2f}, "
            f"quality={prior_quality_weight:.2f}, rarity={prior_rarity_weight:.2f}, "
            f"neg_penalty={prior_neg_penalty:.2f})"
        )

    all_metrics = {}

    for signal, method, backbone in conditions:
        condition_name = f"{signal}_{method}_{backbone}"
        logger.info(f"Running: {condition_name}")

        llm = backbone_llms.get(backbone)
        if llm is None:
            logger.warning(f"LLM not configured for {backbone}, skipping.")
            continue

        generator = MeloGenerator(
            llm=llm,
            temperature=gen_config["temperature"],
            max_tokens=gen_config["max_tokens"],
            num_recommendations=gen_config["num_recommendations"],
            allowed_musical_names=None,
        )

        condition_results = []
        top_k = ret_config["default_top_k"]
        kb_top_m = ret_config["kb_top_m"]

        for user in tqdm(users, desc=condition_name):
            modified_user, held_out = hold_out_one_like(user, seed)
            if not held_out:
                continue

            try:
                # For dual_enhanced, retrieve as "dual" then enhance
                retrieval_signal = "dual" if signal == "dual_enhanced" else signal
                retrieved = retriever.retrieve(
                    modified_user, retrieval_signal, method, top_k, kb_top_m
                )

                # Apply contrastive re-ranking if enabled and signal uses negatives
                if reranker and signal in ("dual", "dual_enhanced", "negative"):
                    retrieved = reranker.rerank_retrieved(retrieved, modified_user)

                # Apply NLI filtering on KB entries if enabled
                if nli_filter and signal in ("dual_enhanced",):
                    dislike_reasons = [
                        d["reason"] for d in modified_user.get("dislikes", [])
                        if d.get("reason")
                    ]
                    if dislike_reasons and retrieved.get("kb_entries"):
                        kb_candidates = [entry for entry, _ in retrieved["kb_entries"]]
                        filtered = nli_filter.filter_candidates(kb_candidates, dislike_reasons)
                        retrieved["kb_entries"] = filtered

                # Build avoidance profile for dual_enhanced prompt
                if profiler and signal == "dual_enhanced":
                    profile = profiler.extract_profile(modified_user)
                    retrieved["avoidance_profile"] = profiler.profile_to_prompt_section(profile)

                output = generator.generate(modified_user, retrieved, signal)
                if prior_enabled:
                    output["recommendations"] = rerank_with_global_priors(
                        output.get("recommendations", []),
                        global_priors,
                        base_rank_weight=prior_base_rank_weight,
                        prior_weight=prior_blend_weight,
                    )

                condition_results.append({
                    "user": user,
                    "held_out": held_out,
                    "recommendations": output["recommendations"],
                    "raw_response": output["raw_response"],
                    "signal": signal,
                    "method": method,
                    "backbone": backbone,
                })
            except Exception as e:
                logger.error(f"Failed for user {user['user_id']} in {condition_name}: {e}")
                continue

        # Evaluate
        metrics = evaluate_condition(
            condition_results,
            kb_entries,
            hit_k_values=config["evaluation"]["hit_k"],
            avoidance_k_values=config["evaluation"]["avoidance_k"],
            tag_threshold=config["evaluation"]["soft_match_tag_threshold"],
        )
        all_metrics[condition_name] = metrics
        logger.info(f"  {condition_name}: {json.dumps(metrics)}")

        # Save per-condition results
        cond_path = run_dir / f"{condition_name}.jsonl"
        with open(cond_path, "w", encoding="utf-8") as f:
            for r in condition_results:
                slim = {
                    **r,
                    "user": {
                        "user_id": r["user"]["user_id"],
                        "dislikes": r["user"]["dislikes"],
                    },
                }
                f.write(json.dumps(slim, ensure_ascii=False) + "\n")

    # Pure LLM baseline (no retrieval)
    logger.info("Running: baseline (no retrieval)")
    for backbone in backbones:
        llm = backbone_llms.get(backbone)
        if llm is None:
            continue

        generator = MeloGenerator(
            llm=llm,
            temperature=gen_config["temperature"],
            max_tokens=gen_config["max_tokens"],
            num_recommendations=gen_config["num_recommendations"],
            allowed_musical_names=allowed_musical_names,
        )

        baseline_results = []
        for user in tqdm(users, desc=f"baseline_{backbone}"):
            modified_user, held_out = hold_out_one_like(user, seed)
            if not held_out:
                continue
            try:
                empty_retrieved = {"positive_pairs": [], "negative_pairs": [], "kb_entries": []}
                output = generator.generate(modified_user, empty_retrieved, "baseline")
                if prior_enabled:
                    output["recommendations"] = rerank_with_global_priors(
                        output.get("recommendations", []),
                        global_priors,
                        base_rank_weight=prior_base_rank_weight,
                        prior_weight=prior_blend_weight,
                    )
                baseline_results.append({
                    "user": user,
                    "held_out": held_out,
                    "recommendations": output["recommendations"],
                    "raw_response": output["raw_response"],
                    "signal": "baseline",
                    "method": "none",
                    "backbone": backbone,
                })
            except Exception as e:
                logger.error(f"Baseline failed for user {user['user_id']}: {e}")
                continue

        metrics = evaluate_condition(baseline_results, kb_entries)
        cond_name = f"baseline_none_{backbone}"
        all_metrics[cond_name] = metrics
        logger.info(f"  {cond_name}: {json.dumps(metrics)}")

    # Final save
    save_metrics(all_metrics, str(run_dir / "metrics.json"))

    # ======================== Ablation: Retrieval Depth (k) ========================
    if args.ablation_k:
        logger.info("Running retrieval depth ablation...")
        k_options = ret_config.get("top_k_options", [1, 3, 5, 10])
        abl_method = "hybrid"
        abl_backbone = None
        for b in backbones:
            if backbone_llms.get(b):
                abl_backbone = b
                break
        if abl_backbone is None:
            logger.warning("No backbone configured for k-ablation, skipping.")
        else:
            generator = MeloGenerator(
                llm=backbone_llms[abl_backbone],
                temperature=gen_config["temperature"],
                max_tokens=gen_config["max_tokens"],
                num_recommendations=gen_config["num_recommendations"],
                allowed_musical_names=allowed_musical_names,
            )
            k_ablation_metrics = {}
            for k_val in k_options:
                for signal in signals:
                    cond_name = f"k_ablation_{signal}_{abl_method}_{abl_backbone}_k{k_val}"
                    logger.info(f"  Running: {cond_name}")
                    cond_results = []
                    for user in tqdm(users, desc=cond_name):
                        modified_user, held_out = hold_out_one_like(user, seed)
                        if not held_out:
                            continue
                        try:
                            retrieval_signal = "dual" if signal == "dual_enhanced" else signal
                            retrieved = retriever.retrieve(
                                modified_user, retrieval_signal, abl_method, k_val, ret_config["kb_top_m"]
                            )
                            output = generator.generate(modified_user, retrieved, signal)
                            if prior_enabled:
                                output["recommendations"] = rerank_with_global_priors(
                                    output.get("recommendations", []),
                                    global_priors,
                                    base_rank_weight=prior_base_rank_weight,
                                    prior_weight=prior_blend_weight,
                                )
                            cond_results.append({
                                "user": user,
                                "held_out": held_out,
                                "recommendations": output["recommendations"],
                                "raw_response": output["raw_response"],
                                "signal": signal,
                                "method": abl_method,
                                "backbone": abl_backbone,
                            })
                        except Exception as e:
                            logger.error(f"k-ablation failed for {user['user_id']} in {cond_name}: {e}")
                            continue
                    metrics = evaluate_condition(cond_results, kb_entries)
                    k_ablation_metrics[cond_name] = metrics
                    logger.info(f"    {cond_name}: {json.dumps(metrics)}")
            save_metrics(k_ablation_metrics, str(run_dir / "k_ablation_metrics.json"))
            logger.info("Retrieval depth ablation complete.")

    # ======================== Ablation: Data Scaling ========================
    if args.ablation_scale:
        logger.info("Running data scaling ablation...")
        user_subsets = exp.get("user_subsets", [25, 50, 100])
        abl_signal = "dual"
        abl_method = "hybrid"
        abl_backbone = None
        for b in backbones:
            if backbone_llms.get(b):
                abl_backbone = b
                break
        if abl_backbone is None:
            logger.warning("No backbone configured for scale ablation, skipping.")
        else:
            generator = MeloGenerator(
                llm=backbone_llms[abl_backbone],
                temperature=gen_config["temperature"],
                max_tokens=gen_config["max_tokens"],
                num_recommendations=gen_config["num_recommendations"],
                allowed_musical_names=allowed_musical_names,
            )
            scale_metrics = {}
            top_k = ret_config["default_top_k"]
            for n_users in user_subsets:
                subset = stratified_user_subset(users, n_users, seed)
                cond_name = f"scale_{abl_signal}_{abl_method}_{abl_backbone}_n{n_users}"
                logger.info(f"  Running: {cond_name} ({len(subset)} users)")
                cond_results = []
                for user in tqdm(subset, desc=cond_name):
                    modified_user, held_out = hold_out_one_like(user, seed)
                    if not held_out:
                        continue
                    try:
                        retrieved = retriever.retrieve(
                            modified_user, abl_signal, abl_method, top_k, ret_config["kb_top_m"]
                        )
                        output = generator.generate(modified_user, retrieved, abl_signal)
                        if prior_enabled:
                            output["recommendations"] = rerank_with_global_priors(
                                output.get("recommendations", []),
                                global_priors,
                                base_rank_weight=prior_base_rank_weight,
                                prior_weight=prior_blend_weight,
                            )
                        cond_results.append({
                            "user": user,
                            "held_out": held_out,
                            "recommendations": output["recommendations"],
                            "raw_response": output["raw_response"],
                            "signal": abl_signal,
                            "method": abl_method,
                            "backbone": abl_backbone,
                        })
                    except Exception as e:
                        logger.error(f"Scale ablation failed for {user['user_id']} in {cond_name}: {e}")
                        continue
                metrics = evaluate_condition(cond_results, kb_entries)
                scale_metrics[cond_name] = metrics
                logger.info(f"    {cond_name}: {json.dumps(metrics)}")
            save_metrics(scale_metrics, str(run_dir / "scale_ablation_metrics.json"))
            logger.info("Data scaling ablation complete.")

    # Cleanup GPU memory
    for name, (model, tokenizer) in backbone_models.items():
        del model, tokenizer
    backbone_models.clear()
    backbone_llms.clear()
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass

    logger.info(f"All results saved to {run_dir}")


if __name__ == "__main__":
    main()
