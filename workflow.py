from dotenv import load_dotenv
load_dotenv(".env")
import os
import sys
import json
import random
import argparse
import logging
import glob
import time
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
# Suppress Pydantic serialization warnings that clutter the logs
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

# Add localization directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'localization'))

from moatless.benchmark.utils import get_moatless_instance
from moatless.benchmark.swebench import create_repository
from moatless.index.code_index import CodeIndex
from moatless.index.settings import IndexSettings
# from moatless.runtime.testbed import TestbedEnvironment

from moatless.file_context import FileContext
from moatless.feedback.feedback_agent import FeedbackAgent
from moatless.value_function.base import ValueFunction

from moatless.actions import FindClass, FindFunction, FindCodeSnippet, SemanticSearch, ViewCode, Finish, Reject, RunTests, StringReplace, CreateFile
from moatless.agent.code_agent import CodingAgent
from moatless.agent.code_prompts import *
from moatless.search_tree import SearchTree
from moatless.completion.completion import (
    LLMResponseFormat,
    CompletionModel,
)

from moatless.discriminator import AgentDiscriminator


import traceback
import re
from enum import Enum

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RetryReason(Enum):
    JSON_PARSE_ERROR = "json_parse_error"
    PATCH_GENERATION_FAILED = "patch_generation_failed"  
    NONE_TYPE_ACTION = "none_type_action"
    TESTBED_TIMEOUT = "testbed_timeout"
    UNKNOWN_ERROR = "unknown_error"

def should_retry_error(error_message: str, exception: Exception = None) -> Tuple[bool, RetryReason]:

    error_lower = error_message.lower()
    
    # 1. django__django-11815: 'NoneType' object has no attribute 'name'
    if "'nonetype' object has no attribute 'name'" in error_lower:
        return True, RetryReason.NONE_TYPE_ACTION
    
    # 2. sympy__sympy-18189: TestbedTimeoutError: Request to ... timed out after 3 retries  
    if "testbedtimeouterror" in error_lower or "apply-patch timed out" in error_lower:
        return True, RetryReason.TESTBED_TIMEOUT
    
    # 3. Has True finish nodes but Unable to generate patch
    if "Unable to generate patch" in error_lower and "The real finished_node exists but there is no patch" in error_lower:
        return True, RetryReason.PATCH_GENERATION_FAILED
    
    return False, RetryReason.UNKNOWN_ERROR


def is_instance_completed(instance_id: str, max_iterations: int, base_dir: str = "tmp") -> bool:
    exp_name = os.path.basename(base_dir)
    patches_file = os.path.join(base_dir, f"model_patches_{exp_name}.jsonl")
    
    if os.path.exists(patches_file):
        with open(patches_file, 'r') as f:
            for line in f:
                try:
                    patch_record = json.loads(line)
                    if patch_record["instance_id"] == instance_id:
                        return True
                except json.JSONDecodeError:
                    continue
    
    trajectory_files = glob.glob(f"{base_dir}/trajectory/{instance_id}/*_trajectory.json")
    
    if not trajectory_files:
        logger.info(f"No trajectory file found for {instance_id}")
        return False
    
    latest_trajectory = max(trajectory_files)
    
    try:
        with open(latest_trajectory, 'r', encoding='utf-8') as f:
            trajectory_data = json.load(f)
        
        def find_all_nodes(node_data, nodes_list=None):
            if nodes_list is None:
                nodes_list = []
            if isinstance(node_data, dict):
                nodes_list.append(node_data)
                if "children" in node_data:
                    for child in node_data["children"]:
                        find_all_nodes(child, nodes_list)
            return nodes_list
        
        all_nodes = []
        if "root" in trajectory_data:
            all_nodes = find_all_nodes(trajectory_data["root"])
        
        report_files = glob.glob(f"{base_dir}/experience/{instance_id}/*_report.json")
        
        if report_files:
            latest_report = max(report_files)
            
            try:
                with open(latest_report, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                
                patch_applied = report_data.get("patch_applied", False)
                patch = report_data.get("patch", "")
                
                if patch_applied and patch:
                    logger.info(f"Instance {instance_id}: Successfully completed with patch applied and tested")
                    return True
                    
            except Exception as e:
                logger.error(f"Error reading report file {latest_report}: {e}")

        current_date = datetime.now().strftime("%Y-%m-%d")
        execution_log_path = f'{base_dir}/trajectory/{instance_id}/{current_date}_execution.log'
        
        if os.path.exists(execution_log_path):
            try:
                with open(execution_log_path, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                    
                testbed_errors = [
                    "Health check failed",
                    "TestbedTimeoutError", 
                    "Error running tests for instance",
                    "apply-patch timed out",
                    "Request failed, retrying",
                    "Operation timed out after"
                ]
                
                for error_pattern in testbed_errors:
                    if error_pattern in log_content:
                        logger.info(f"Instance {instance_id}: Detected testbeds failure ({error_pattern}), marking as incomplete for retry")
                        return False
                        
            except Exception as e:
                logger.warning(f"Error reading execution log for {instance_id}: {e}")
        
        target_node_id = max_iterations - 1
        target_node = None
        
        for node in all_nodes:
            if node.get("node_id") == target_node_id:
                target_node = node
                break
        
        if target_node:
            visits = target_node.get("visits", 0)
            has_observation = False

            if "action_steps" in target_node:
                for step in target_node["action_steps"]:
                    if step.get("observation") is not None:
                        has_observation = True
                        break
            
            if visits > 0 and has_observation:
                logger.info(f"Instance {instance_id}: Max iterations completed")
                return True

        finish_nodes = []
        for node in all_nodes:
            if "action_steps" in node:
                for step in node["action_steps"]:
                    action = step.get("action", {})
                    if action.get("action_name") == "Finish" or action.get("name") == "Finish":
                        finish_nodes.append(node)
                        break
        
        if finish_nodes:
            logger.info(f"Instance {instance_id}: Found {len(finish_nodes)} finish nodes but no successful patch generated")

            return False

        if len(all_nodes) >= max_iterations * 0.8: 
            logger.info(f"Instance {instance_id}: Search appears to have run significantly ({len(all_nodes)} nodes) but no completion criteria met")
            return False
        
        logger.info(f"Instance {instance_id}: Search not completed (only {len(all_nodes)} nodes, target: {max_iterations})")
        return False
            
    except Exception as e:
        logger.error(f"Error checking completion status for {instance_id}: {e}")
        return False


def load_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line.strip()))
    return data_list


def save2json(data, path):
    directory = os.path.dirname(path)
    
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def save_to_jsonl(data_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def add_data_to_jsonl(file_path, new_data):
    if file_path and os.path.exists(file_path):
        data_list = load_jsonl(file_path)
    else:
        data_list = []

    if isinstance(new_data, list):
        data_list.extend(new_data)
    else:
        data_list.append(new_data)

    save_to_jsonl(data_list, file_path)


def get_leaves_with_patch(tree) -> Dict:
    leaves = tree.get_leaf_nodes()
    parent_ids = set()
    tmp = dict()
    counter = 0
    for leaf in leaves:
        if leaf.file_context.generate_git_patch():
            if leaf.is_terminal() and leaf.parent.node_id not in parent_ids:
                parent_ids.add(leaf.parent.node_id)
                patch = leaf.file_context.generate_git_patch()
                # result = leaf.file_context.run_evaluation()
                report_ = {
                    "leaf_id": leaf.node_id,
                    # "patch_applied": result.patch_applied,
                    # "resolved": result.resolved,
                    "patch": patch,
                }
                tmp[str(leaf.node_id)] = report_
                counter += 1
        else:
            report_ = {
                "leaf_id": leaf.node_id,
                # "patch_applied": False,
                # "resolved": False,
                "patch": None,
            }
            tmp[str(leaf.node_id)] = report_
            counter += 1
    return tmp

def get_path_to_leaf(leaf_node_id, tree):
    try:
        target_node = None
        
        def find_node(node):
            nonlocal target_node
            if node.node_id == leaf_node_id:
                target_node = node
                return
            for child in node.children:
                find_node(child)
        
        find_node(tree.root)
        
        if not target_node:
            return f"Node ID not found: {leaf_node_id}"
        
        path_nodes = []
        current = target_node
        
        while current:
            path_nodes.append(current)
            current = current.parent

        path_nodes.reverse()

        result_parts = []
        for i, node in enumerate(path_nodes):
            if hasattr(node, 'action_steps') and node.action_steps:
                action_step = node.action_steps[0]
                
                action_str = f"{action_step.action.__class__.__name__}: {action_step.action}"
                observation_str = action_step.observation if action_step.observation else "No observation"
                
                result_parts.append(f"Action{node.node_id}:")
                result_parts.append(f"{action_str}")
                result_parts.append(f"Observation: {observation_str}")
                result_parts.append("---")
        
        return "\n".join(result_parts)
        
    except Exception as e:
        return f"error: {e}"


def main(instance_id, max_iterations, max_finish_nodes, result_path=None, use_testbed=False, stop_after_stage=None):
    if result_path is None:
        base_dir = os.path.abspath("tmp")
    else:
        base_dir = result_path
    
    instance_logger = setup_instance_logging(instance_id, base_dir)

    instance_logger.info("=" * 100)
    instance_logger.info("🚀 STARTING NEW EXPERIMENT RUN")
    instance_logger.info("=" * 100)
    instance_logger.info(f"� Instance ID: {instance_id}")
    instance_logger.info(f"📁 Base directory: {base_dir}")
    instance_logger.info(f"🔄 Max iterations: {max_iterations}")
    instance_logger.info(f"🎯 Max finished nodes: {max_finish_nodes}")
    instance_logger.info(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    instance_logger.info("=" * 100)

    retry_count = 0
    MAX_RETRIES = 3 

    while True:
        try:
            global_model = os.getenv("GLOBAL_MODEL", "deepseek/deepseek-chat")
            instance_logger.info(f"🤖 Using model: {global_model}")

            from moatless.completion.react import ReActCompletionModel
            react_completion_model = ReActCompletionModel(model=global_model, temperature=0.7)
            completion_model = CompletionModel(model=global_model, temperature=0.7)
            discriminator_model = CompletionModel(model=global_model, temperature=1)
            value_model = CompletionModel(model=global_model, temperature=0.4)

            react_completion_model.response_format = LLMResponseFormat.REACT
            completion_model.response_format = LLMResponseFormat.REACT
            discriminator_model.response_format = LLMResponseFormat.REACT
            value_model.response_format = LLMResponseFormat.REACT

            instance_logger.info("📥 Loading instance from SWE-bench...")
            instance = get_moatless_instance(instance_id=instance_id)
            instance_logger.info(f"✅ Instance loaded: {instance.get('instance_id', 'unknown')}")
            instance_logger.info(f"📝 Problem statement preview: {instance.get('problem_statement', '')[:200]}...")
            
            instance_logger.info("🏗️ Creating repository...")
            repository = create_repository(instance)
            instance_logger.info(f"✅ Repository created at: {repository.repo_dir if hasattr(repository, 'repo_dir') else 'N/A'}")
        
            index_store_dir = os.getenv("INDEX_STORE_DIR", os.path.abspath("tmp/index_store"))
            
            instance_logger.info("🧠 Loading Code Index...")
            code_index = CodeIndex.from_index_name(
                instance["instance_id"], file_repo=repository, index_store_dir=index_store_dir
            )
            instance_logger.info("✅ Code Index loaded successfully")

            instance_logger.info("📄 Initializing file context...")
            file_context = FileContext(repo=repository)
            instance_logger.info("✅ File context initialized")

            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            instance_path = f'{base_dir}/trajectory/{instance_id}/'
            persist_path = f'{base_dir}/trajectory/{instance_id}/run_{current_datetime}_trajectory.json'
            
            instance_logger.info(f"📊 Trajectory will be saved to: {persist_path}")

            instance_logger.info("⚙️ Setting up actions and system prompt...")
            
            value_function = ValueFunction(completion_model=value_model)
            actions = [
                FindClass(completion_model=react_completion_model, code_index=code_index, repository=repository),
                FindFunction(completion_model=react_completion_model, code_index=code_index, repository=repository),
                FindCodeSnippet(completion_model=react_completion_model, code_index=code_index, repository=repository),
                SemanticSearch(completion_model=react_completion_model, code_index=code_index, repository=repository),
                ViewCode(completion_model=react_completion_model, repository=repository),
                StringReplace(repository=repository, code_index=code_index),
                CreateFile(repository=repository, code_index=code_index),
                Finish(),
            ]
            instance_logger.info(f"   Available actions: {[action.__class__.__name__ for action in actions]}")

            system_prompt = AGENT_ROLE
            if completion_model.response_format == LLMResponseFormat.REACT:
                system_prompt += REACT_CORE_OPERATION_RULES
            elif completion_model.response_format == LLMResponseFormat.TOOLS:
                system_prompt += REACT_GUIDELINES
            workflow_prompt = generate_workflow_prompt(actions, has_runtime=True)
            system_prompt += workflow_prompt + generate_guideline_prompt(has_runtime=True) + ADDITIONAL_NOTES

            instance_logger.info(f"✅ System prompt length: {len(system_prompt)} characters")
            instance_logger.info("🤖 Initializing Coding Agent...")
            agent = CodingAgent(system_prompt=system_prompt, actions=actions, completion=react_completion_model)
            instance_logger.info("✅ Agent initialized successfully")

            instance_logger.info("🎭 Initializing Discriminator (Multi-Agent Debate)...")
            discriminator = AgentDiscriminator(
                completion=discriminator_model,
                n_agents=5,
                n_rounds=3,
            )
            instance_logger.info(f"   Debate config: {5} agents, {3} rounds")

            instance_logger.info("💬 Initializing Feedback Agent...")
            feedback_generator = FeedbackAgent(
                completion_model=completion_model, instance_dir=instance_path
            )
            instance_logger.info("✅ Feedback agent initialized")

            instance_logger.info("=" * 100)
            instance_logger.info("🎯 ENTITY LOCALIZATION PHASE")
            instance_logger.info("=" * 100)

            from entity_localization_pipeline import EntityLocalizationPipeline

            pipeline = EntityLocalizationPipeline()
            instance_logger.info("🔍 Running entity localization pipeline...")
            results = pipeline.run_pipeline(
                instance, "context",
                max_initial_entities=5,
                stop_after_stage=stop_after_stage,
            )
            instance_logger.info(f"✅ Entity localization completed")
            results_str = str(results)
            instance_logger.info(f"   Results: {results_str[:500]}..." if len(results_str) > 500 else f"   Results: {results}")

            if stop_after_stage == "stage_7_final_plan":
                # In stage-7-only mode, mark success only when a valid final plan exists.
                if isinstance(results, dict):
                    if results.get("error"):
                        instance_logger.warning(
                            f"⚠️ stage_7_final_plan not produced: {results.get('error')}"
                        )
                        return False

                    if results.get("success") is False:
                        instance_logger.warning(
                            f"⚠️ stage_7_final_plan generation failed: {results.get('error', 'unknown error')}"
                        )
                        return False

                    final_plan = results.get("final_plan")
                    if not final_plan:
                        instance_logger.warning("⚠️ stage_7_final_plan is missing in pipeline result")
                        return False

                instance_logger.info("🛑 stop_after_stage=stage_7_final_plan set, skipping stage 8+ and MCTS search")
                return True

            if stop_after_stage == "stage_8_edit_agent_prompt":
                # In stage-8-only mode, mark success only when edit prompt exists.
                if not isinstance(results, str) or not results.strip():
                    instance_logger.warning("⚠️ stage_8_edit_agent_prompt is missing or empty")
                    return False

                instance_logger.info("🛑 stop_after_stage=stage_8_edit_agent_prompt set, skipping MCTS search")
                return True

            instance_logger.info("=" * 100)
            instance_logger.info("🌳 SEARCH TREE INITIALIZATION")
            instance_logger.info("=" * 100)

            search_tree = SearchTree.create(
                message=f'{results}',
                agent=agent,
                file_context=file_context,
                value_function=value_function,
                discriminator=discriminator,
                feedback_generator=feedback_generator,
                max_finished_nodes=max_finish_nodes,
                max_iterations=max_iterations,
                max_expansions=3,
                max_depth=20,
                max_duplicate_count=5, 
                persist_path=persist_path,
            )
            instance_logger.info(f"✅ Search tree created")
            instance_logger.info(f"   Max iterations: {max_iterations}")
            instance_logger.info(f"   Max finished nodes: {max_finish_nodes}")
            instance_logger.info(f"   Max expansions per node: 3")
            instance_logger.info(f"   Max depth: 20")

            instance_logger.info("=" * 100)
            instance_logger.info("🔍 STARTING MCTS SEARCH")
            instance_logger.info("=" * 100)
            
            finished_node = search_tree.run_search()
            
            instance_logger.info("=" * 100)
            instance_logger.info("💾 SAVING SEARCH TREE")
            instance_logger.info("=" * 100)
            search_tree.persist(persist_path)
            instance_logger.info(f"✅ Search tree saved to: {persist_path}")

            if finished_node:
                instance_logger.info("=" * 100)
                instance_logger.info("🎉 SOLUTION FOUND - GENERATING PATCH")
                instance_logger.info("=" * 100)
                
                if finished_node.file_context.generate_git_patch():
                    patch = finished_node.file_context.generate_git_patch()
                    instance_logger.info(f"✅ Git patch generated successfully")
                    instance_logger.info(f"   Patch length: {len(patch)} characters")
                    instance_logger.info(f"   Patch preview:\n{patch[:500]}..." if len(patch) > 500 else f"   Patch:\n{patch}")

                    eva2rollout = get_leaves_with_patch(search_tree)
                    
                    eva2rollout['source_tree_path'] = persist_path
                    eva2rollout['debate_node'] = str(finished_node.node_id)
                    save2json(eva2rollout, f'/tmp/trajectory/{instance_id}/eval2rollout.json')
                    instance_logger.info(f"✅ Evaluation data saved to: /tmp/trajectory/{instance_id}/eval2rollout.json")

                    if not search_tree.get_finished_nodes() and finished_node.file_context.generate_git_patch():
                        instance_logger.info("📝 Saving single finished node patch...")
                        tmp = {
                            "model_name_or_path": "DeepSeek_IA",
                            "instance_id": instance_id,
                            "model_patch": finished_node.file_context.generate_git_patch(),
                            "leaf_id": finished_node.node_id,
                            'source_tree_path': persist_path,
                            'debate_node': str(finished_node.node_id),
                        }
                        add_data_to_jsonl('/tmp_patch_1.jsonl', tmp)
                        add_data_to_jsonl('/tmp_patch_2.jsonl', tmp)
                        instance_logger.info("✅ Patch saved to /tmp_patch_1.jsonl and /tmp_patch_2.jsonl")

                    new_eval_objects = []
                    finished_nodes = search_tree.get_finished_nodes()
                    instance_logger.info(f"📊 Found {len(finished_nodes)} finished nodes in search tree")
                    
                    for i in finished_nodes:
                        trajectory = f'Issue: {instance["problem_statement"]}\nTrajectory:\n'
                        trajectory += get_path_to_leaf(i.node_id, search_tree)
                        trajectory += f"\nGenerated Patch:\n{i.file_context.generate_git_patch()}"
                        tmp = {
                            "model_name_or_path": "DeepSeek_IA",
                            "instance_id": instance_id,
                            "model_patch": i.file_context.generate_git_patch(),
                            "leaf_id": i.node_id,
                            'source_tree_path': persist_path,
                            'debate_node': str(finished_node.node_id),
                            'trajectory': trajectory,
                        }
                        new_eval_objects.append(tmp)

                    if len(new_eval_objects) > 1:
                        instance_logger.info(f"💾 Saving {len(new_eval_objects)} evaluation objects...")
                        add_data_to_jsonl('/tmp_patch_1.jsonl', new_eval_objects[0])
                        add_data_to_jsonl('/tmp_patch_2.jsonl', new_eval_objects[1])
                    elif len(new_eval_objects) == 1:
                        instance_logger.info(f"💾 Saving 1 evaluation object...")
                        add_data_to_jsonl('/tmp_patch_1.jsonl', new_eval_objects)
                        add_data_to_jsonl('/tmp_patch_2.jsonl', new_eval_objects)
                    

                    tmp = {
                        "model_name_or_path": "DeepSeek_IA",
                        "instance_id": instance_id,
                        "model_patch": finished_node.file_context.generate_git_patch(),
                    }   
                    add_data_to_jsonl('/tmp_prediction_patch.jsonl', tmp)
                    instance_logger.info("✅ Prediction patch saved to /tmp_prediction_patch.jsonl")
                    
                    instance_logger.info("=" * 100)
                    instance_logger.info("✅ EXPERIMENT COMPLETED SUCCESSFULLY")
                    instance_logger.info("=" * 100)
                    instance_logger.info(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    instance_logger.info(f"🎯 Solution found at node: {finished_node.node_id}")
                    instance_logger.info("=" * 100)

            else:
                instance_logger.warning("=" * 100)
                instance_logger.warning("⚠️ NO SOLUTION FOUND")
                instance_logger.warning("=" * 100)
                instance_logger.warning("The search ended normally but no solution was found")
                instance_logger.warning(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                instance_logger.warning("=" * 100)
                return False
                
        except Exception as e:
            error_msg = str(e)
            instance_logger.error("=" * 100)
            instance_logger.error("❌ EXCEPTION OCCURRED")
            instance_logger.error("=" * 100)
            instance_logger.error(f"Error message: {error_msg}")
            import traceback
            full_traceback = traceback.format_exc()
            instance_logger.error(f"📋 Detailed traceback:\n{full_traceback}")
            
            if "MCTS_DUPLICATES_EXCEEDED" in error_msg:
                if retry_count >= MAX_RETRIES:
                    instance_logger.error(f"❌ Maximum retry limit ({MAX_RETRIES}) reached, abandoning this instance")
                    instance_logger.error("=" * 100)
                    return False
                retry_count += 1
                instance_logger.warning(f"⚠️ Too many duplicate nodes detected, retrying attempt {retry_count}...")
                instance_logger.warning("=" * 100)
                continue

            should_retry, retry_reason = should_retry_error(error_msg, e)
            if should_retry:
                if retry_count >= MAX_RETRIES:
                    instance_logger.error(f"❌ Maximum retry limit ({MAX_RETRIES}) reached, abandoning this instance")
                    instance_logger.error("=" * 100)
                    return False
                retry_count += 1
                instance_logger.warning(f"⚠️ Retryable error ({retry_reason.value}) occurred, retry attempt {retry_count}...")
                instance_logger.warning("=" * 100)
                continue

            instance_logger.error("=" * 100)
            return False  


def parse_slice(slice_str: str) -> slice:
    if not slice_str:
        return slice(None)
    
    try:
        parts = slice_str.split(':')
        
        if len(parts) > 3:
            raise ValueError(f"Invalid slice syntax: {slice_str}")
        
        while len(parts) < 3:
            parts.append('')
        
        def parse_int(s):
            return int(s) if s else None
        
        start = parse_int(parts[0])
        stop = parse_int(parts[1])
        step = parse_int(parts[2])
        
        return slice(start, stop, step)
        
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid slice syntax '{slice_str}': {e}")


def setup_instance_logging(instance_id: str, base_dir: str):
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f'{base_dir}/trajectory/{instance_id}/'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = f'{log_dir}/run_{current_datetime}_execution.log'
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    instance_logger = logging.getLogger(f"instance_{instance_id}")
    instance_logger.setLevel(logging.INFO)
    
    for handler in instance_logger.handlers[:]:
        instance_logger.removeHandler(handler)
    
    instance_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(f'[{instance_id}] %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.WARNING) 
    instance_logger.addHandler(console_handler)

    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.INFO)

    for handler in litellm_logger.handlers[:]:
        litellm_logger.removeHandler(handler)

    litellm_logger.addHandler(file_handler)
    litellm_logger.propagate = False 
    

    moatless_loggers = [
        "moatless",
        "moatless.search_tree", 
        "moatless.agent",
        "moatless.actions",
        "moatless.completion",
        "moatless.runtime",
        "moatless.file_context",
        "moatless.feedback",
        "moatless.discriminator",
        "moatless.node",
        "moatless.tree",
        "moatless.value_function",
        "moatless.benchmark"
    ]
    
    for logger_name in moatless_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        logger.addHandler(file_handler)
        logger.propagate = False 
    
    other_loggers = [
        "openai",
        "httpx", 
        "anthropic",
        "deepseek",
        "testbeds"
    ]
    
    for logger_name in other_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        logger.addHandler(file_handler)
        logger.propagate = False 

    root_logger = logging.getLogger()
    

    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):

            handler.setLevel(logging.WARNING)
        elif isinstance(handler, logging.FileHandler):

            root_logger.removeHandler(handler)
    

    root_logger.setLevel(logging.WARNING)
    
    return instance_logger

def process_single_instance(args_tuple: Tuple[str, int, int, str, bool, str, str]) -> Tuple[str, bool, str]:
    instance_id, max_iterations, max_finished_nodes, base_dir, resume, result_path, stop_after_stage = args_tuple
    
    def print_progress(message: str):
        print(f"[{instance_id}] {message}")

    retry_count = 0
    retry_history = []
    MAX_RETRIES = 3
    
    while retry_count <= MAX_RETRIES:
        try:
            if retry_count == 0:
                print_progress("📋 start...")
            else:
                print_progress(f"🔄  {retry_count}/{MAX_RETRIES} - : {retry_history[-1]}")
            
            if resume and retry_count == 0:  
                if is_instance_completed(instance_id, max_iterations, base_dir):
                    return (instance_id, True, f"Already completed {max_iterations} iterations")
            
            success = main(
                instance_id,
                max_iterations,
                max_finished_nodes,
                result_path,
                stop_after_stage=stop_after_stage,
            )
            
            if success:
                retry_info = f" (after {retry_count} retries)" if retry_count > 0 else ""
                message = f"Success{retry_info}"
                
                import random
                delay = random.uniform(1.0, 3.0)
                time.sleep(delay)
                
                return (instance_id, success, message)
            else:
                print_progress("❌ resolved: False")
                return (instance_id, False, "Failed to resolve")
        
        except Exception as e:
            error_msg = str(e)
            full_traceback = traceback.format_exc()

            should_retry, retry_reason = should_retry_error(error_msg, e)
            
            if should_retry and retry_count < MAX_RETRIES:
                retry_count += 1
                retry_history.append(retry_reason.value)

                wait_time = min(60, 10 * (2 ** (retry_count - 1))) 
                time.sleep(wait_time)
                
                current_date = datetime.now().strftime("%Y-%m-%d")
                log_file_path = f'{base_dir}/trajectory/{instance_id}/{current_date}_execution.log'
                try:
                    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                    with open(log_file_path, 'a', encoding='utf-8') as f:
                        f.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - RETRY - INFO - "
                               f" {retry_count}/{MAX_RETRIES}, : {retry_reason.value}\n")
                        f.write(f": {error_msg}\n")
                except Exception:
                    pass  
                
                continue  
            else:
                if retry_count >= MAX_RETRIES:
                    final_error_msg = f"({MAX_RETRIES}). : {error_msg}"
                    print_progress(f"💥 : {final_error_msg[:100]}...")
                    
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    log_file_path = f'{base_dir}/trajectory/{instance_id}/{current_date}_execution.log'
                    try:
                        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                        with open(log_file_path, 'a', encoding='utf-8') as f:
                            f.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - FINAL_FAILURE - ERROR - "
                                   f" {retry_history}\n")
                            f.write(f"{error_msg}\n")
                            f.write(f":\n{full_traceback}\n")
                    except Exception:
                        pass
                        
                    return (instance_id, False, final_error_msg)
                else:
                    return (instance_id, False, f"Non-retryable error: {error_msg}")

    return (instance_id, False, "Unexpected end of retry loop")


def init_worker():
    """Initializer for multiprocessing workers."""
    # Keep workers from handling Ctrl+C directly; let main process coordinate shutdown.
    try:
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass


def process_instances_parallel(instance_ids: List[str], max_iterations: int, 
                             max_finished_nodes: int, base_dir: str, 
                             resume: bool = False, num_processes: int = 1, result_path: str = None,
                             stop_after_stage: str = None) -> List[str]:
    pass_instances = []
    
    args_list = [
        (instance_id, max_iterations, max_finished_nodes, base_dir, resume, result_path, stop_after_stage)
        for instance_id in instance_ids
    ]
    
    if num_processes == 1:
        for args_tuple in tqdm(args_list, desc="Processing instances",
                              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
            instance_id, success, message = process_single_instance(args_tuple)
            if success:
                if "Already completed" not in message:
                    pass_instances.append(instance_id)
                    time.sleep(60)
                else:
                    print(f"[{instance_id}] ⏭️ jump")
            else:
                print(f"[{instance_id}] ❌ : {message}")
    else:

        batch_size = num_processes * 2  
        total_batches = (len(args_list) + batch_size - 1) // batch_size
        
        
        with mp.Pool(processes=num_processes, initializer=init_worker) as pool:
            results = []
            completed_tasks = 0
            
            with tqdm(total=len(args_list), desc="Processing instances", 
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
                
                pending_results = {}  
                submitted_count = 0
                max_concurrent = min(batch_size, num_processes * 2)  
                timeout_seconds = 7200  
                
                while completed_tasks < len(args_list):
                    while (submitted_count < len(args_list) and 
                           len(pending_results) < max_concurrent):
                        
                        args_tuple = args_list[submitted_count]
                        instance_id = args_tuple[0]
                        
                        async_result = pool.apply_async(process_single_instance, (args_tuple,))
                        start_time = time.time()
                        pending_results[async_result] = (submitted_count, instance_id, start_time)
                        
                        if submitted_count % 10 == 0:  
                            print(f"📤  {submitted_count + 1}/{len(args_list)}: {instance_id}")
                        
                        submitted_count += 1
                        
                        time.sleep(random.uniform(0.2, 0.8))
                    
                    completed_results = []
                    timeout_results = []
                    current_time = time.time()
                    
                    for async_result, (task_idx, instance_id, start_time) in list(pending_results.items()):
                        elapsed_time = current_time - start_time
                        
                        if async_result.ready():
                            completed_results.append((async_result, task_idx, instance_id, elapsed_time))
                        elif elapsed_time > timeout_seconds:
                            timeout_results.append((async_result, task_idx, instance_id, elapsed_time))
                    
                    for async_result, task_idx, instance_id, elapsed_time in timeout_results:
                        try:
                            try:
                                async_result.get(timeout=0.1)
                            except mp.TimeoutError:
                                try:
                                    pass
                                except Exception as terminate_error:
                                    print(f"⚠️ : {terminate_error}")
                            except Exception:
                                pass
                            
                            timeout_message = f"Timeout after {elapsed_time//60:.1f} minutes (worker process may still be running)"
                            results.append((instance_id, False, timeout_message))
                            completed_tasks += 1
                            
                            pbar.set_postfix_str(f"{instance_id} : {len(pending_results)-1}")
                            pbar.update(1)
                            
                        except Exception as e:
                            pbar.update(1)
                            completed_tasks += 1
                        
                        del pending_results[async_result]
                    
                    for async_result, task_idx, instance_id, elapsed_time in completed_results:
                        try:
                            result_id, success, message = async_result.get(timeout=1)
                            results.append((result_id, success, message))
                            completed_tasks += 1
                            
                            if success and "Already completed" not in message:
                                pass_instances.append(result_id)
                            
                            status = "✅" if success else "❌"
                            if "Already completed" in message:
                                status = "⏭️"
                            
                            minutes = elapsed_time // 60
                            seconds = elapsed_time % 60
                            time_str = f"{minutes:.0f}m{seconds:.0f}s" if minutes > 0 else f"{seconds:.0f}s"
                            
                            concurrent_info = f"concurrent: {len(pending_results)-1}"
                            pbar.set_postfix_str(f"{instance_id} {status}({time_str}) | {concurrent_info}")
                            pbar.update(1)
                            
                            if completed_tasks % 5 == 0:  
                                pass
                            
                            if success:
                                delay = random.uniform(1.0, 3.0)
                                time.sleep(delay)
                            
                        except mp.TimeoutError:
                            pbar.update(1)
                            completed_tasks += 1
                        except Exception as e:
                            pbar.update(1)
                            completed_tasks += 1
                        
                        del pending_results[async_result]
                    
                    if not completed_results:
                        time.sleep(1)  
                    
                    if completed_tasks > 0 and completed_tasks % 20 == 0:
                        concurrent_count = len(pending_results)
                        remaining = len(args_list) - submitted_count
                        
                        if concurrent_count > 0:
                            rest_time = random.uniform(2.0, 5.0)
                            time.sleep(rest_time)
    
    return pass_instances


def get_retry_info_for_instance(instance_id: str, base_dir: str) -> List[str]:
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_file_path = f'{base_dir}/trajectory/{instance_id}/{current_date}_execution.log'
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r', encoding='utf-8') as f:
                return [line for line in f.readlines() if "RETRY" in line or "retry" in line]
    except Exception:
        pass
    return []

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process some arguments.")


    parser.add_argument("--instance_ids", type=str, required=True,
                        help="The file path to instance ID(s)")

    parser.add_argument("--max_iterations", type=int, default=50, help="Max iteration for tree search (default: 50, recommended: 50-200 depending on complexity)")

    parser.add_argument("--max_finished_nodes", type=int, default=3, help="Max finished nodes for tree search")

    parser.add_argument("--resume", action="store_true", help="Resume from the last instance")

    parser.add_argument("--num_processes", type=int, default=4, help="Number of processes for parallel processing")

    parser.add_argument("--result_path", type=str, default=None, help="Custom result directory path (default: tmp/experience)")

    parser.add_argument("--slice", type=str, default=None, help="Python slice syntax to select instances (e.g., '0:10', '10:', ':5', '10:20:2')")
    parser.add_argument(
        "--stop_after_stage",
        type=str,
        default=None,
        choices=["stage_7_final_plan", "stage_8_edit_agent_prompt"],
        help="Stop after the specified localization stage and skip later phases",
    )

    args = parser.parse_args()

    with open(args.instance_ids, "r", encoding='utf-8') as f:
        instance_ids = [line.strip() for line in f if line.strip()]

    if args.slice:
        try:
            slice_obj = parse_slice(args.slice)
            original_count = len(instance_ids)
            instance_ids = instance_ids[slice_obj]
            print(f"🔢 Applying slice '{args.slice}': {original_count} -> {len(instance_ids)} instances")
            if len(instance_ids) == 0:
                print("⚠️ No instances left after slicing, program exits")
                exit(0)
        except ValueError as e:
            print(f"❌ Slice parameter error: {e}")
            exit(1)
    
    print(f"📋 Finally processing {len(instance_ids)} instances")
    if args.result_path is None:
        base_dir = os.path.abspath("tmp")
    else:
        base_dir = args.result_path
    
    pass_instances = []

    if len(instance_ids) == 1:

        instance_id = instance_ids[0]

        args_tuple = (
            instance_id,
            args.max_iterations,
            args.max_finished_nodes,
            base_dir,
            args.resume,
            args.result_path,
            args.stop_after_stage,
        )
        instance_id_result, success, message = process_single_instance(args_tuple)
        
        if success:
            if "Already completed" not in message:
                pass_instances.append(instance_id)
                print("🎉 Pass@1: 1")
            else:
                print("✅ Pass@1: Already completed")
        else:
            print(f"❌ Pass@1: 0 - {message}")
            if "retry" in message:
                print(f"📊 retry: {message}")
    else:
        pass_instances = process_instances_parallel(
            instance_ids,
            args.max_iterations,
            args.max_finished_nodes,
            base_dir,
            args.resume,
            args.num_processes,
            args.result_path,
            args.stop_after_stage,
        )
        success_rate = len(pass_instances) / len(instance_ids)
        print(f"\n🎯 Final Result:")
        print(f"   Success Rate: {success_rate:.2%} ({len(pass_instances)}/{len(instance_ids)})")
        print(f"   Successful Instances: {pass_instances}")

        retry_count = sum(1 for instance_id in pass_instances if any(
            "retry" in log_line for log_line in get_retry_info_for_instance(instance_id, base_dir)
        ))
        if retry_count > 0:
            print(f" 💫 Instances that successfully passed retry: {retry_count}/{len(pass_instances)}")

    print('\n🏁 All done!')