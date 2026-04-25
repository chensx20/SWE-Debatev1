#!/usr/bin/env python3
"""
从 SWE-Debatev1 的 pipeline_cache.json 中提取轻量化定位提示 (localization hints)，
用于注入 mini-swe-agent 的 prompt 中以提高缺陷修复性能。

轻量化策略：
- 只提取 文件路径 + 关键实体 + 修改计划 + 策略共识
- 不注入源代码（agent 可以自己读取）
- 总 token 预算控制在 500-1000 tokens

用法:
    python extract_hints.py                          # 提取所有 instance
    python extract_hints.py django__django-10973     # 提取指定 instance
    python extract_hints.py --output-dir ./my_hints  # 指定输出目录
"""

import json
import os
import glob
import argparse
import logging
from typing import Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 默认路径
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entity_pipeline_cache")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "localization_hints")


def find_latest_cache(instance_dir: str) -> Optional[str]:
    """找到某个 instance 目录中最新的 cache 文件"""
    json_files = sorted(glob.glob(os.path.join(instance_dir, "*_pipeline_cache.json")))
    if not json_files:
        return None
    return json_files[-1]


def extract_target_files_from_winning_chain(stage6_data: Dict) -> List[Dict]:
    """从 stage_6 投票结果的 winning_chain 中提取目标文件列表"""
    target_files = []
    seen_files = set()

    voting_result = stage6_data.get("voting_result", stage6_data)
    winning_chain = voting_result.get("winning_chain", {})

    if not winning_chain:
        return target_files

    # 从 winning chain 的 entities_with_code 中提取文件级实体
    for entity in winning_chain.get("entities_with_code", []):
        entity_id = entity.get("entity_id", "")
        entity_type = entity.get("type", "")

        # 提取文件路径
        if ":" in entity_id:
            file_path = entity_id.split(":")[0]
        else:
            file_path = entity_id

        # 跳过目录和空路径
        if entity_type == "directory" or not file_path or not file_path.endswith(".py"):
            continue

        if file_path in seen_files:
            continue
        seen_files.add(file_path)

        # 收集该文件中的关键实体
        key_entities = []
        for e in winning_chain.get("entities_with_code", []):
            eid = e.get("entity_id", "")
            etype = e.get("type", "")
            if eid.startswith(file_path + ":") and etype in ("class", "function"):
                entity_name = eid.split(":")[-1]
                start = e.get("start_line")
                end = e.get("end_line")
                if start and end:
                    key_entities.append(f"{entity_name} (lines {start}-{end})")
                else:
                    key_entities.append(entity_name)

        target_files.append({
            "file_path": file_path,
            "key_entities": key_entities
        })

    return target_files


def extract_modification_plan_from_stage7(stage7_data: Dict) -> Optional[Dict]:
    """从 stage_7 最终计划中提取修改计划"""
    final_plan_wrapper = stage7_data.get("final_plan", {})

    # 处理嵌套的 final_plan 结构
    if "final_plan" in final_plan_wrapper:
        final_plan = final_plan_wrapper["final_plan"]
    else:
        final_plan = final_plan_wrapper

    if not final_plan or not final_plan.get("modifications"):
        return None

    # 提取简洁的修改步骤
    steps = []
    for mod in final_plan.get("modifications", []):
        step = {
            "step": mod.get("step", len(steps) + 1),
            "action": mod.get("instruction", ""),
            "file": _extract_file_from_context(mod.get("context", "")),
            "priority": mod.get("priority", "medium"),
        }
        # 只在有有价值的实现细节时才添加 details
        impl_notes = mod.get("implementation_notes", "")
        if impl_notes and len(impl_notes) < 200:
            step["details"] = impl_notes
        steps.append(step)

    return {
        "summary": final_plan.get("summary", ""),
        "steps": steps
    }


def _extract_file_from_context(context_str: str) -> str:
    """从 context 字符串中提取文件路径 (e.g., 'django/db/foo.py, lines 5-10' -> 'django/db/foo.py')"""
    if not context_str:
        return ""
    parts = context_str.split(",")
    file_part = parts[0].strip()
    # 验证看起来像是一个文件路径
    if "/" in file_part and file_part.endswith(".py"):
        return file_part
    return context_str.split(",")[0].strip()


def extract_strategy_consensus(stage6_data: Dict) -> str:
    """从 stage_6 投票结果中提取修改策略共识"""
    voting_result = stage6_data.get("voting_result", stage6_data)

    # 从 all_votes 中提取 modification_strategy
    all_votes = voting_result.get("all_votes", [])
    strategies = []
    for vote in all_votes:
        strategy = vote.get("modification_strategy", "")
        if strategy:
            strategies.append(strategy)

    if strategies:
        # 返回第一个（通常是最有代表性的）
        return strategies[0]

    return ""


def extract_confidence(stage7_data: Dict, stage6_data: Dict) -> int:
    """提取置信度分数"""
    # 优先从 stage7 取
    final_plan = stage7_data.get("final_plan", {})
    conf = final_plan.get("confidence", 0)
    if conf:
        return conf

    # 回退到 stage6
    voting_result = stage6_data.get("voting_result", stage6_data)
    return voting_result.get("average_confidence", 0)


def extract_from_round1_analyses(stage7_data: Dict) -> Optional[Dict]:
    """
    当 stage_7_final_plan 不存在时，回退到从 stage_7_round1_analysis 提取。
    从第一轮分析中提取共识最强的修改位置。
    """
    round1 = stage7_data.get("first_round_analyses", [])
    if not round1:
        return None

    # 收集所有分析中的高优先级修改位置
    all_locations = []
    for agent_analysis in round1:
        analysis = agent_analysis.get("analysis")
        if not analysis:
            continue
        for loc in analysis.get("modification_locations", []):
            if loc.get("priority") == "high":
                all_locations.append(loc)

    if not all_locations:
        return None

    # 取第一个高优先级位置作为主要修改目标
    primary = all_locations[0]
    return {
        "summary": primary.get("reasoning", "")[:200],
        "steps": [{
            "step": 1,
            "action": primary.get("suggested_approach", "")[:200] if primary.get("suggested_approach") else "",
            "file": _extract_file_from_context(primary.get("entity_id", "")),
            "priority": "high"
        }]
    }


def extract_hint(cache_path: str) -> Optional[Dict]:
    """
    从单个 pipeline_cache.json 中提取定位提示。

    优先提取顺序:
    1. stage_7_final_plan (最佳)
    2. stage_7_round1_analysis (次选)
    3. stage_6_voting_result (兜底)
    """
    logger.info(f"Loading cache: {cache_path}")

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    # 确定 instance_id
    instance_id = cache.get("instance_id", os.path.basename(os.path.dirname(cache_path)))

    hint = {
        "instance_id": instance_id,
        "localization_confidence": 0,
        "target_files": [],
        "modification_plan": None,
        "strategy_consensus": "",
        "source": ""  # 记录数据来源
    }

    # --- 从 stage_6 提取目标文件 ---
    stage6 = cache.get("stage_6_voting_result", {}).get("data", {})
    if stage6:
        hint["target_files"] = extract_target_files_from_winning_chain(stage6)
        hint["strategy_consensus"] = extract_strategy_consensus(stage6)
        logger.info(f"  Stage 6: {len(hint['target_files'])} target files found")

    # --- 从 stage_7 提取修改计划 ---
    stage7_final = cache.get("stage_7_final_plan", {}).get("data", {})
    stage7_round1 = cache.get("stage_7_round1_analysis", {}).get("data", {})

    if stage7_final and stage7_final.get("final_plan"):
        # 最佳情况：有完整的 final_plan
        hint["modification_plan"] = extract_modification_plan_from_stage7(stage7_final)
        hint["localization_confidence"] = extract_confidence(stage7_final, stage6)
        hint["source"] = "stage_7_final_plan"
        logger.info(f"  Stage 7 final plan: confidence={hint['localization_confidence']}")

    elif stage7_round1:
        # 次选：从 round1 分析中提取
        hint["modification_plan"] = extract_from_round1_analyses(stage7_round1)
        hint["localization_confidence"] = 70  # round1 置信度较低
        hint["source"] = "stage_7_round1_analysis"
        logger.info(f"  Stage 7 round1 analysis: using first-round consensus")

    elif stage6:
        # 兜底：只有 stage6 的投票信息
        hint["modification_plan"] = {
            "summary": hint["strategy_consensus"],
            "steps": []
        }
        hint["localization_confidence"] = 60
        hint["source"] = "stage_6_voting_result"
        logger.info(f"  Fallback to stage 6 voting result")

    else:
        logger.warning(f"  No stage 6 or 7 data available, skipping")
        return None

    # 验证提取结果
    if not hint["target_files"] and not hint.get("modification_plan"):
        logger.warning(f"  Extracted hint is empty, skipping")
        return None

    return hint


def format_hint_as_text(hint: Dict) -> str:
    """
    将 hint 格式化为可直接注入 prompt 的纯文本。
    这是轻量版：不包含代码，只包含路径、实体和修改计划。
    """
    lines = []

    # 目标文件
    if hint.get("target_files"):
        lines.append("**Target Files for Modification:**")
        for tf in hint["target_files"]:
            lines.append(f"- `{tf['file_path']}`")
            if tf.get("key_entities"):
                for entity in tf["key_entities"]:
                    lines.append(f"  - {entity}")
        lines.append("")

    # 修改计划
    plan = hint.get("modification_plan")
    if plan:
        if plan.get("summary"):
            lines.append(f"**Modification Strategy:** {plan['summary']}")
            lines.append("")

        if plan.get("steps"):
            lines.append("**Modification Steps:**")
            for step in plan["steps"]:
                priority_marker = "🔴" if step.get("priority") == "critical" else "🟡" if step.get("priority") == "high" else "⚪"
                lines.append(f"{priority_marker} Step {step['step']}: {step['action']}")
                if step.get("file"):
                    lines.append(f"   File: `{step['file']}`")
                if step.get("details"):
                    lines.append(f"   Details: {step['details']}")
            lines.append("")

    # 策略共识（如果和 plan summary 不同）
    if hint.get("strategy_consensus") and (not plan or hint["strategy_consensus"] != plan.get("summary", "")):
        lines.append(f"**Consensus Strategy:** {hint['strategy_consensus']}")
        lines.append("")

    return "\n".join(lines)


def process_all_instances(cache_dir: str, output_dir: str, instance_filter: Optional[str] = None):
    """处理所有（或指定的）instance，提取 hints 并保存"""
    os.makedirs(output_dir, exist_ok=True)

    results = {"success": 0, "skipped": 0, "failed": 0, "instances": {}}

    for instance_name in sorted(os.listdir(cache_dir)):
        instance_path = os.path.join(cache_dir, instance_name)
        if not os.path.isdir(instance_path):
            continue

        if instance_filter and instance_name != instance_filter:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {instance_name}")
        logger.info(f"{'='*60}")

        latest_cache = find_latest_cache(instance_path)
        if not latest_cache:
            logger.warning(f"  No cache file found, skipping")
            results["skipped"] += 1
            continue

        try:
            hint = extract_hint(latest_cache)
            if hint is None:
                results["skipped"] += 1
                results["instances"][instance_name] = {"status": "skipped", "reason": "insufficient data"}
                continue

            # 保存 JSON
            output_json = os.path.join(output_dir, f"{instance_name}.json")
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(hint, f, indent=2, ensure_ascii=False)

            # 保存纯文本（用于直接粘贴或调试）
            output_txt = os.path.join(output_dir, f"{instance_name}.txt")
            text = format_hint_as_text(hint)
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(text)

            results["success"] += 1
            results["instances"][instance_name] = {
                "status": "success",
                "source": hint["source"],
                "confidence": hint["localization_confidence"],
                "target_files": len(hint["target_files"]),
                "steps": len(hint.get("modification_plan", {}).get("steps", [])),
                "text_length": len(text),
            }

            logger.info(f"  ✅ Saved: {output_json}")
            logger.info(f"  📝 Text hint: {len(text)} chars")

        except Exception as e:
            logger.error(f"  ❌ Failed: {e}")
            results["failed"] += 1
            results["instances"][instance_name] = {"status": "failed", "error": str(e)}

    # 保存汇总
    summary_path = os.path.join(output_dir, "_extraction_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"Extraction Summary:")
    logger.info(f"  Success: {results['success']}")
    logger.info(f"  Skipped: {results['skipped']}")
    logger.info(f"  Failed:  {results['failed']}")
    logger.info(f"  Output:  {output_dir}")
    logger.info(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract localization hints from SWE-Debatev1 pipeline cache"
    )
    parser.add_argument(
        "instance_id",
        nargs="?",
        default=None,
        help="Optional: specific instance ID to process (e.g., django__django-10973)"
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )

    args = parser.parse_args()
    process_all_instances(args.cache_dir, args.output_dir, args.instance_id)


if __name__ == "__main__":
    main()
