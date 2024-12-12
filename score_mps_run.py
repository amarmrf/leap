import os
import random
import json
import threading
import argparse
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing_extensions import TypedDict
from peft import PeftModel, LoraConfig, get_peft_model
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    StoppingCriteria, StoppingCriteriaList
)
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import radon.complexity as radon_complexity
from sympy import simplify, SympifyError
from sympy.parsing.sympy_parser import parse_expr
import ast
import wandb

# Initialize NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        logger.info(f"Seed set to {seed}.")
    except Exception as e:
        logger.error(f"Error setting seed: {e}")
        raise RuntimeError("Failed to set seed.") from e


@dataclass
class Config:
    """
    Configuration dataclass for training parameters.
    """
    beta_1: float = 0.01
    beta_2: float = 0.1
    alpha: float = 5.0
    learning_rate: float = 1e-5
    batch_size: int = 1
    max_seq_len: int = 2048
    max_new_tokens: int = 2048
    num_epochs_stage_one: int = 1
    num_epochs_stage_two: int = 1

    # device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device: torch.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    seed: int = 42
    task: str = 'CODE'
    model_variant: str = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'
    ablation: str = 'none'
    data_path: str = './data'
    output_dir: str = './outputs'
    num_workers: int = 2
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    save_steps: int = 1000
    logging_steps: int = 10
    eval_steps: int = 1000
    max_eval_samples: int = 500
    mixed_precision: bool = False
    save_total_limit: int = 2
    compute_bleu: bool = False
    compute_rouge: bool = False
    compute_cyclomatic_complexity: bool = False

    def validate(self) -> None:
        """
        Validate configuration parameters.
        """
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer.")
        if self.num_epochs_stage_one < 0 or self.num_epochs_stage_two < 0:
            raise ValueError("Number of epochs must be non-negative.")
        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        if not os.path.isdir(self.output_dir):
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                logger.info(f"Created output directory at {self.output_dir}.")
            except Exception as e:
                logger.error(f"Failed to create output directory: {e}")
                raise


def get_code_first_turn_prompt(problem: str, test_list: List[str]) -> str:
    """Generate the first turn prompt for code problems.
    
    Args:
        problem (str): Problem description
        test_list (List[str]): List of test cases
        
    Returns:
        str: Formatted prompt for first attempt
    """
    examples = [
        {
            "problem": "Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].",
            "tests": [
                "assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8",
                "assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12", 
                "assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16"
            ],
            "solution": """R = 3
C = 3
def min_cost(cost, m, n): 
    tc = [[0 for x in range(C)] for x in range(R)] 
    tc[0][0] = cost[0][0] 
    for i in range(1, m+1): 
        tc[i][0] = tc[i-1][0] + cost[i][0] 
    for j in range(1, n+1): 
        tc[0][j] = tc[0][j-1] + cost[0][j] 
    for i in range(1, m+1): 
        for j in range(1, n+1): 
            tc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j] 
    return tc[m][n]"""
        },
        {
            "problem": "Write a function to find the similar elements from the given two tuple lists.",
            "tests": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"
            ],
            "solution": """def similar_elements(test_tup1, test_tup2):
    res = tuple(set(test_tup1) & set(test_tup2))
    return (res)"""
        },
        {
            "problem": "Write a python function to identify non-prime numbers.",
            "tests": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True", 
                "assert is_not_prime(35) == True"
            ],
            "solution": """import math
def is_not_prime(n):
    result = False
    for i in range(2,int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result"""
        }
    ]

    # Build examples section
    examples_str = ""
    for i, example in enumerate(examples, 1):
        examples_str += f"\nExample {i}:\n"
        examples_str += f"Problem: {example['problem']}\n"
        examples_str += f"Your code should pass these tests:\n"
        examples_str += "\n".join(example["tests"]) + "\n"
        examples_str += "[BEGIN]\n"
        examples_str += example["solution"] + "\n"
        examples_str += "[DONE]\n"
    
    test_cases = "\n".join(test_list) if test_list else ""
    
    prompt = (
        "<|im_start|>system\nYou are an expert Python programmer. Here are some examples of problems and their test cases:\n"
        f"{examples_str}"
        "<|im_end|>\n"
        f"<|im_start|>user\nNow please solve this problem:\n{problem}\n\n"
        f"Your code should pass these tests:\n{test_cases}\n[BEGIN]\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt

def get_code_correction_prompt(problem: str, prev_attempt: str) -> str:
    """Generate the self-correction prompt for code problems.
    
    Args:
        problem (str): Original problem description including function signature and test cases 
        prev_attempt (str): Previous code attempt to be corrected
        
    Returns:
        str: Formatted prompt for correction attempt
    """
    return (
        f"{prev_attempt}\n\n"
        "<|im_start|>user\n"
        "There might be an error in the code above because of lack of understanding of the question. "
        "Please correct the error, if any, and rewrite the solution. Only output the final correct Python program!\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


class BaseDataset(Dataset):
    """
    Base dataset class for loading data.
    """

    def __init__(self, data: List[Dict[str, Any]], task: str = 'CODE'):
        self.data = data
        self.task = task

    def __len__(self) -> int:
        return len(self.data)
    def prepare_prompt(self, item: Dict[str, Any], turn: int = 1, prev_attempt: Optional[str] = None) -> str:
        """
        Prepare prompt based on task and turn number.
        
        Args:
            item: Data item containing problem/prompt
            turn: Turn number (1 or 2)
            prev_attempt: Previous attempt for turn 2
            
        Returns:
            Formatted prompt string
        """
        if self.task == 'CODE':
            if turn == 1:
                test_list = item.get('test_list', [])
                return get_code_first_turn_prompt(item.get('text', item.get('prompt', '')), test_list)
            else:
                return get_code_correction_prompt(item.get('text', item.get('prompt', '')), prev_attempt)
        else:
            raise NotImplementedError(f"Task {self.task} is not implemented")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            item = self.data[idx]
            # Format prompt for first turn
            item['formatted_prompt'] = self.prepare_prompt(item)
            return item
        except IndexError as e:
            logger.error(f"Index {idx} out of range for dataset of size {len(self.data)}.")
            raise IndexError("Dataset index out of range.") from e
        except Exception as e:
            logger.error(f"Error retrieving item at index {idx}: {e}")
            raise


def load_json(file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load data from a JSON or JSONL file.

    Args:
        file_path (str): Path to the JSON or JSONL file.
        max_samples (Optional[int]): Maximum number of samples to load.

    Returns:
        List[Dict[str, Any]]: Loaded data.
    """
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be a non-negative integer or None")

    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                for idx, line in enumerate(f):
                    if max_samples is not None and idx >= max_samples:
                        break
                    if line.strip():  # Skip empty lines
                        data.append(json.loads(line))
            else:
                file_content = f.read().strip()
                if file_content:
                    loaded_data = json.loads(file_content)
                    if isinstance(loaded_data, list):
                        data = loaded_data[:max_samples] if max_samples else loaded_data
                    else:
                        data = [loaded_data]
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}") from e
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in file {file_path}: {e}")
        raise ValueError(f"Invalid JSON format in file: {file_path}") from e
    except Exception as e:
        logger.error(f"Unexpected error while loading JSON from {file_path}: {e}")
        raise RuntimeError(f"Failed to load data from {file_path}") from e

    logger.info(f"Loaded {len(data)} samples from {file_path}.")
    return data


class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for text generation."""
    def __init__(self, stop_token_ids: List[List[int]], min_length: int = 20):
        self.stop_token_ids = stop_token_ids
        self.min_length = min_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Don't stop if we haven't generated minimum length
        if input_ids.shape[-1] < self.min_length:
            return False
            
        # Check for stop sequences
        for stop_ids in self.stop_token_ids:
            if len(stop_ids) > 0 and torch.all((input_ids[0][-len(stop_ids):] == torch.tensor(stop_ids).to(input_ids.device))).item():
                return True
        return False

class AdvancedModel(nn.Module):
    """
    Advanced model wrapper with tokenizer and generation capabilities.
    """

    def __init__(self, model_name: str, device: torch.device):
        super().__init__()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                padding_side='left',
                trust_remote_code=True  # Add this for Qwen
            )
            # Update markers for Qwen
            self.system_marker = "<|im_start|>system"
            self.user_marker = "<|im_start|>user"
            self.assistant_marker = "<|im_start|>assistant"
            self.stop_sequences = [
                "<|im_end|>",
                self.system_marker,
                self.user_marker,
                "Previous Attempt:",
                "Instructions:"
            ]
            logger.info(f"Tokenizer loaded for {model_name}.")
        except Exception as e:
            logger.error(f"Error loading tokenizer for {model_name}: {e}")
            raise RuntimeError(f"Failed to load tokenizer for {model_name}") from e

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Using EOS token as PAD token.")

        try:
            lora_config = LoraConfig(
                r=1,
                lora_alpha=1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Updated for Qwen
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,  # Add this for Qwen
                device_map=device
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            logger.info(f"Model loaded and moved to {device}.")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise RuntimeError(f"Failed to load model {model_name}") from e


        try:
            if not self.tokenizer.pad_token:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
                logger.info("Added pad token and resized token embeddings.")
        except Exception as e:
            logger.error(f"Error adding pad token or resizing embeddings: {e}")
            raise RuntimeError("Failed to add pad token or resize embeddings.") from e

        self.device = device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Logits from the model.
        """
        try:
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        except Exception as e:
            logger.error(f"Error during forward pass: {e}")
            raise RuntimeError("Forward pass failed.") from e
      
    def generate_text(
        self,
        inputs: Dict[str, torch.Tensor],
        max_length: int = 4096,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        min_length: int = 20  # Minimum length of generated text
    ) -> torch.Tensor:
        """
        Generate text using the model.

        Args:
            inputs (Dict[str, torch.Tensor]): Tokenized inputs.
            max_length (int): Maximum length of generated text.
            temperature (float): Sampling temperature.
            num_return_sequences (int): Number of sequences to generate.

        Returns:
            torch.Tensor: Generated token IDs.
        """
        try:
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_length,
                min_new_tokens=min_length,  # Add minimum tokens
                temperature=max(temperature, 1e-7),
                do_sample=temperature > 0,
                top_p=0.95,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            return outputs

        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise RuntimeError("Text generation failed.") from e


class RewardsDict(TypedDict):
    """
    TypedDict for rewards and related metrics.
    """
    rewards: torch.Tensor
    bleu: List[float]
    rouge: List[Dict[str, float]]
    cyclomatic: List[float]


class SCoReTrainer:
    """
    Trainer class for the SCoRe system.
    """

    def __init__(
        self,
        model: AdvancedModel,
        ref_model: AdvancedModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config
    ):
        self.task = config.task
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.global_step = 0
        self.reward_history: List[float] = []
        self.edit_distance_ratios: List[float] = []
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision and torch.cuda.is_available())
        self.use_wandb = False


        try:
            wandb.login(key="5846629ab2a2094c5948b4c032301fdae772fbb0", relogin=True) 
            wandb.init(
                project="score-training",
                config={
                    "task": config.task,
                    "model_variant": config.model_variant,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "beta_1": config.beta_1,
                    "beta_2": config.beta_2,
                    "alpha": config.alpha
                }
            )
            self.use_wandb = True
            logger.info("Weights & Biases initialized successfully.")
        except Exception as e:
            logger.warning(f"Failed to initialize Weights & Biases: {e}")
            self.use_wandb = False

    def compute_kl_divergence(self, logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between model logits and reference logits.

        Args:
            logits (torch.Tensor): Logits from the model.
            ref_logits (torch.Tensor): Logits from the reference model.

        Returns:
            torch.Tensor: KL divergence loss.
        """
        try:
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            ref_probs = nn.functional.softmax(ref_logits, dim=-1)
            kl_div = self.kl_loss_fn(log_probs, ref_probs)
            return kl_div
        except Exception as e:
            logger.error(f"Error computing KL divergence: {e}")
            raise RuntimeError("KL divergence computation failed.") from e

    def _save_trace(self, trace_info: Dict) -> None:
        """
        Save trace information to a JSON file with pretty printing.
        """
        try:
            trace_file = os.path.join(self.config.output_dir, 'reward_traces.jsonl')
            with open(trace_file, 'a') as f:
                # Pretty print the JSON with indentation
                json_str = json.dumps(trace_info, indent=2)
                # Add a newline after each JSON object
                f.write(json_str + '\n\n')
        except Exception as e:
            logger.error(f"Error saving trace information: {e}")

    def safe_execute_code(self, code: str, test: str, timeout: int = 5) -> bool:
        """
        Safely execute generated code with a test case.

        Args:
            code (str): Generated code.
            test (str): Test case code.
            timeout (int): Timeout in seconds.

        Returns:
            bool: Execution success status.
        """
        def target(exec_globals: Dict[str, Any]) -> None:
            try:
                exec(code, exec_globals)
                exec(test, exec_globals)
                exec_globals['exec_success'] = True
            except Exception as e:
                logger.warning(f"Execution error: {e}")
                exec_globals['exec_success'] = False

        exec_globals: Dict[str, Any] = {}
        thread = threading.Thread(target=target, args=(exec_globals,), daemon=True)
        try:
            thread.start()
            thread.join(timeout)
            success = exec_globals.get('exec_success', False)
            if not success and thread.is_alive():
                logger.warning("Code execution timed out.")
                return False
            return success
        except Exception as e:
            logger.error(f"Error during code execution thread: {e}")
            return False

    def compute_cyclomatic_complexity(self, code: str) -> float:
        """
        Compute cyclomatic complexity of the given code.

        Args:
            code (str): Code to analyze.

        Returns:
            float: Average cyclomatic complexity.
        """
        try:
            complexity = radon_complexity.cc_visit(code)
            avg_complexity = np.mean([block.complexity for block in complexity]) if complexity else 0.0
            logger.debug(f"Cyclomatic complexity: {avg_complexity}")
            return avg_complexity
        except SyntaxError as e:
            logger.warning(f"SyntaxError while computing cyclomatic complexity: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error computing cyclomatic complexity: {e}")
            return 0.0

    def reward_function_code(self, code: str, test: str) -> Tuple[float, float]:
        """
        Compute rewards for code tasks with detailed validation and testing.
        
        Args:
            code (str): Generated code to evaluate
            test (str): Test cases to run
            
        Returns:
            Tuple containing (reward, cyclomatic_complexity)
        """
        logger.info("\n=== Code Reward Computation ===")
        
        trace_info = {
            "generated_code": {
                "raw": code,
                "cleaned": None,
                "ast_valid": False,
                "execution_result": None,
            },
            "test_cases": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "execution_details": []
            },
            "metrics": {
                "cyclomatic_complexity": 0.0,
                "execution_time": None,
            }
        }

        try:
            # Step 1: Clean and normalize the code
            cleaned_code = self._clean_code_response(code)
            trace_info["generated_code"]["cleaned"] = cleaned_code
            
            # Step 2: Extract function name and normalize test cases if needed
            try:
                actual_func_name = self.extract_function_name(cleaned_code)
                if actual_func_name:
                    test_cases = test.split('\n')
                    if test_cases and test_cases[0].strip():
                        expected_func_name = test_cases[0].split('(')[0].replace('assert ', '')
                        if actual_func_name != expected_func_name:
                            test_cases = [test.replace(expected_func_name, actual_func_name) for test in test_cases]
                            test = '\n'.join(test_cases)
                            logger.info(f"Normalized function name from {expected_func_name} to {actual_func_name}")
            except Exception as e:
                logger.warning(f"Function name extraction failed: {e}")

            # Step 3: Validate code syntax using AST
            try:
                ast.parse(cleaned_code)
                trace_info["generated_code"]["ast_valid"] = True
            except SyntaxError as e:
                logger.warning(f"Code syntax validation failed: {str(e)}")
                self._save_trace(trace_info)
                return 0.0, 0.0

            # Step 4: Execute code and run test cases
            exec_globals = {}
            all_tests_passed = True
            test_cases = [t for t in test.split('\n') if t.strip()]
            trace_info["test_cases"]["total"] = len(test_cases)
            
            try:
                # First execute the solution code
                start_time = time.time()
                exec(cleaned_code, exec_globals)
                logger.info("✓ Code execution successful")

                # Then try all test cases
                for i, test_case in enumerate(test_cases, 1):
                    test_result = {
                        "test_case": test_case,
                        "passed": False,
                        "error": None
                    }
                    
                    try:
                        exec(test_case, exec_globals)
                        test_result["passed"] = True
                        trace_info["test_cases"]["passed"] += 1
                        logger.info(f"Test {i}: ✓ {test_case}")
                    except AssertionError:
                        all_tests_passed = False
                        test_result["error"] = "Assertion failed"
                        trace_info["test_cases"]["failed"] += 1
                        logger.info(f"Test {i}: × Failed assertion: {test_case}")
                    except Exception as e:
                        all_tests_passed = False
                        test_result["error"] = str(e)
                        trace_info["test_cases"]["failed"] += 1
                        logger.info(f"Test {i}: × Failed with error: {str(e)}")
                        
                    trace_info["test_cases"]["execution_details"].append(test_result)
                
                execution_time = time.time() - start_time
                trace_info["metrics"]["execution_time"] = execution_time
                trace_info["generated_code"]["execution_result"] = "success"
                
            except Exception as e:
                logger.warning(f"Code execution failed: {str(e)}")
                trace_info["generated_code"]["execution_result"] = f"failed: {str(e)}"
                self._save_trace(trace_info)
                return 0.0, 0.0

            # Step 5: Compute cyclomatic complexity if enabled
            if self.config.compute_cyclomatic_complexity:
                try:
                    complexity = radon_complexity.cc_visit(cleaned_code)
                    avg_complexity = np.mean([block.complexity for block in complexity]) if complexity else 0.0
                    trace_info["metrics"]["cyclomatic_complexity"] = avg_complexity
                except Exception as e:
                    logger.warning(f"Error computing cyclomatic complexity: {e}")
                    trace_info["metrics"]["cyclomatic_complexity"] = 0.0

            # Step 6: Compute final reward
            reward = 1.0 if all_tests_passed else 0.0
            self._save_trace(trace_info)
            
            return reward, trace_info["metrics"]["cyclomatic_complexity"]

        except Exception as e:
            logger.error(f"Error in reward computation: {str(e)}")
            trace_info["error"] = str(e)
            self._save_trace(trace_info)
            return 0.0, 0.0
    
    def extract_function_name(self, code: str) -> Optional[str]:
        """Extract the main function name from code using AST parsing."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except Exception as e:
            logger.warning(f"Failed to extract function name: {e}")
        return None

    def _clean_code_response(self, text: str) -> str:
        """
        Clean generated code while preserving indentation and structure.
        """
        try:
            # Remove markdown code blocks
            if '```' in text:
                code_blocks = text.split('```')
                for i, block in enumerate(code_blocks):
                    if i % 2 == 1:  # Only keep content between backticks
                        lines = block.splitlines()
                        # Skip the language identifier line (python, Python, etc.)
                        if lines and lines[0].lower().strip() in ['python', 'py']:
                            lines = lines[1:]
                        # Skip empty lines at start and end
                        while lines and not lines[0].strip():
                            lines = lines[1:]
                        while lines and not lines[-1].strip():
                            lines = lines.pop()
                        return '\n'.join(lines)
                
            # If no code blocks found, clean the raw text
            lines = []
            for line in text.splitlines():
                # Skip the language identifier if it appears at the start
                if line.lower().strip() in ['python', 'py']:
                    continue
                # Skip empty lines and comments
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                lines.append(line)
                            
            return '\n'.join(lines)
                    
        except Exception as e:
            logger.error(f"Error cleaning code response: {e}")
            return text

        # Add debug logging
        finally:
            if hasattr(self, 'global_step') and self.global_step % self.config.logging_steps == 0:
                logger.debug("Code Cleaning Results:")
                logger.debug(f"Original:\n{text}")
                logger.debug(f"Cleaned:\n{'\n'.join(lines)}")
        
    def compute_rewards(
        self,
        generated: List[str],
        correct: List[str],
        test_cases: Optional[List[str]]
    ) -> RewardsDict:
        """
        Compute rewards for a batch of generated outputs.

        Args:
            generated (List[str]): List of generated outputs.
            correct (List[str]): List of correct answers or code.
            test_cases (Optional[List[str]]): List of test cases for code tasks.

        Returns:
            RewardsDict: Dictionary containing rewards and metrics.
        """
        rewards = []
        bleu = []
        rouge = []
        cyclomatic = []

        for i, gen in enumerate(generated):
            try:
                if self.config.task == 'CODE':
                    # Clean the generated code by removing markdown and docstrings
                    gen = self._clean_code_response(gen)
                    
                    test = test_cases[i] if test_cases and i < len(test_cases) else ''
                    if test:
                        r, c = self.reward_function_code(gen, test)
                    else:
                        logger.warning(f"Missing test case for CODE task at index {i}. Assigning zero reward.")
                        r, c = 0.0, 0.0
                    rewards.append(r)
                    cyclomatic.append(c)
            except Exception as e:
                logger.error(f"Error computing rewards for index {i}: {e}")
                rewards.append(0.0)
                if self.config.task == 'CODE':
                    cyclomatic.append(0.0)

        rewards_tensor = torch.tensor(rewards, device=self.config.device)
        logger.debug(f"Rewards computed: {rewards}")
        return {
            'rewards': rewards_tensor,
            'bleu': bleu,
            'rouge': rouge,
            'cyclomatic': cyclomatic
        }

    def compute_edit_distance_ratio(self, s1: str, s2: str) -> float:
        """
        Compute the edit distance ratio between two strings.

        Args:
            s1 (str): First string.
            s2 (str): Second string.

        Returns:
            float: Edit distance ratio.
        """
        try:
            ratio = SequenceMatcher(None, s1, s2).ratio()
            logger.debug(f"Edit distance ratio between '{s1}' and '{s2}': {ratio}")
            return ratio
        except Exception as e:
            logger.error(f"Error computing edit distance ratio: {e}")
            return 0.0


    def prepare_batch(
        self,
        batch: List[Dict[str, Any]],
        turn: int = 1,
        prev_attempts: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str], Optional[List[str]]]:
        """
        Prepare a batch of data for processing.
        
        Args:
            batch: List of data items containing problems/prompts
            turn: Turn number (1 or 2)
            prev_attempts: Previous attempts for turn 2
            
        Returns:
            Tuple containing (inputs, correct answers, test cases)
        """
        try:
            if isinstance(batch, dict):
                batch = [batch]
                
            if self.task == 'CODE':
                # Extract problem text and handle array wrapping
                problems = []
                for item in batch:
                    text = item.get('text', item.get('prompt', ''))
                    if isinstance(text, list):
                        text = text[0] if text else ''
                    problems.append(text)
                
                # Handle array-wrapped correct answers
                correct = []
                for item in batch:
                    solution = item.get('code', item.get('canonical_solution', ''))
                    if isinstance(solution, list):
                        solution = solution[0] if solution else ''
                    correct.append(solution)
                
                # Convert test lists to strings and handle tuples
                test_lists = []
                for item in batch:
                    test_list = item.get('test_list', [])
                    if isinstance(test_list, tuple):
                        test_list = [str(test) for test in test_list]
                    elif isinstance(test_list, list):
                        test_list = [str(test[0]) if isinstance(test, tuple) else str(test) for test in test_list]
                    test_lists.append(test_list)
                
                if turn == 1:
                    inputs = [
                        get_code_first_turn_prompt(p, t) 
                        for p, t in zip(problems, test_lists)
                    ]
                else:
                    inputs = [
                        get_code_correction_prompt(p, pa) 
                        for p, pa in zip(problems, prev_attempts)
                    ]
                    
                # Join test cases into single strings
                tests = ['\n'.join(test_list) for test_list in test_lists]
                return inputs, correct, tests
                
        except Exception as e:
            logger.error(f"Error preparing batch: {e}")
            raise RuntimeError("Failed to prepare batch.") from e



    def train(self) -> None:
        """
        Train the model through both training stages.
        """
        try:
            logger.info("Starting training process.")
            for epoch in range(self.config.num_epochs_stage_one):
                logger.info(f"Starting Stage I Training - Epoch {epoch + 1}")
                self.stage_one()
            # for epoch in range(self.config.num_epochs_stage_two):
            #     logger.info(f"Starting Stage II Training - Epoch {epoch + 1}")
            #     self.stage_two()
            logger.info("Training completed successfully.")
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log training metrics to wandb and console."""
        if not self.use_wandb:
            return

        try:
            # Prepare metrics dictionary
            wandb_metrics = {
                # Loss components
                "train/total_loss": metrics["total_loss"],
                "train/kl_loss": metrics.get("kl_loss", 0.0),
                "train/reward_loss": metrics.get("reward_loss", 0.0),
                
                # Rewards
                "train/mean_reward_t1": metrics["rewards_t1"].mean().item(),
                "train/mean_reward_t2": metrics["rewards_t2"].mean().item(),
                "train/reward_improvement": (metrics["rewards_t2"] - metrics["rewards_t1"]).mean().item(),
                
                # Training dynamics
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                
                # Edit distance metrics
                "train/edit_distance_ratio": np.mean(metrics.get("edit_distance_ratios", [0.0])),
                
                # Task-specific metrics
                "train/bleu_score": np.mean(metrics.get("bleu_scores", [0.0])) if self.config.compute_bleu else None,
                "train/rouge_score": np.mean(metrics.get("rouge_scores", [0.0])) if self.config.compute_rouge else None,
                "train/cyclomatic_complexity": np.mean(metrics.get("cyclomatic_scores", [0.0])) if self.config.compute_cyclomatic_complexity else None
            }

            # Remove None values
            wandb_metrics = {k: v for k, v in wandb_metrics.items() if v is not None}

            # Log to wandb
            if step is not None:
                wandb.log(wandb_metrics, step=step)
            else:
                wandb.log(wandb_metrics)

            # Log summary metrics to console
            if self.global_step % self.config.logging_steps == 0:
                logger.info(
                    f"Step {self.global_step} - "
                    f"Loss: {metrics['total_loss']:.4f}, "
                    f"Reward T1: {wandb_metrics['train/mean_reward_t1']:.4f}, "
                    f"Reward T2: {wandb_metrics['train/mean_reward_t2']:.4f}, "
                    f"Improvement: {wandb_metrics['train/reward_improvement']:.4f}"
                )

        except Exception as e:
            logger.error(f"Error logging metrics to wandb: {e}")

    def stage_one(self) -> None:
        """
        Stage I training: Train the model to produce high-reward responses at the second attempt
        while constraining first attempt to be close to base model.
        """
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Stage I Training"):
            self.global_step += 1
            try:
                inputs, correct, tests = self.prepare_batch(batch, turn=1)
                
                # First attempt: Constrain to base model
                first_encodings = self.model.tokenizer(
                    inputs,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_len
                ).to(self.config.device)
                
                # Get logits for first attempt
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision and torch.cuda.is_available()):
                    first_logits = self.model(first_encodings['input_ids'], first_encodings['attention_mask'])
                    with torch.no_grad():
                        ref_logits = self.ref_model(first_encodings['input_ids'], first_encodings['attention_mask'])
                    
                    # KL divergence on first attempt to stay close to base model
                    kl_loss = self.compute_kl_divergence(first_logits, ref_logits) * self.config.beta_2
                    
                    # Generate first attempt response
                    first_ids = self.model.generate_text(first_encodings, max_length=self.config.max_seq_len, temperature=1.0)
                    first_responses = self.model.tokenizer.batch_decode(first_ids, skip_special_tokens=True)
                    
                    # Print first attempt details
                    if self.global_step % self.config.logging_steps == 0:
                        for idx, (inp, resp, corr) in enumerate(zip(inputs, first_responses, correct)):
                            logger.info(f"\n=== Sample {idx + 1} First Attempt ===")
                            # logger.info(f"Input:\n{inp}")
                            # logger.info(f"Model Response:\n{resp}")
                            logger.info(f"Correct Answer:\n{corr}")
                    
                    # Create second attempt inputs
                    second_inputs, correct, tests = self.prepare_batch(
                        batch, 
                        turn=2,
                        prev_attempts=first_responses
                    )
                    
                    # Second attempt: Optimize for high reward
                    second_encodings = self.model.tokenizer(
                        second_inputs,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_seq_len
                    ).to(self.config.device)
                    
                    second_ids = self.model.generate_text(second_encodings, max_length=self.config.max_seq_len)
                    second_responses = self.model.tokenizer.batch_decode(second_ids, skip_special_tokens=True)
                    
                    # Print second attempt details
                    if self.global_step % self.config.logging_steps == 0:
                        for idx, (prompt, resp) in enumerate(zip(second_inputs, second_responses)):
                            logger.info(f"\n=== Sample {idx + 1} Second Attempt ===")
                            #   logger.info(f"Prompt:\n{prompt}")
                            logger.info(f"Model Response:\n{resp}")
                    
                    # Compute rewards
                    rewards = self.compute_rewards(second_responses, correct, tests)['rewards']
                    
                    # Print rewards
                    if self.global_step % self.config.logging_steps == 0:
                        logger.info(f"\nRewards: {rewards.tolist()}")
                    
                    # Total loss is negative reward for second attempt plus KL penalty on first attempt
                    loss = -rewards.mean() + kl_loss

                    try:
                        # Collect metrics
                        metrics = {
                            "total_loss": loss.item(),
                            "kl_loss": kl_loss.item(),
                            "reward_loss": -rewards.mean().item(),
                            "rewards_t1": torch.zeros_like(rewards),  # First attempt has no rewards in Stage I
                            "rewards_t2": rewards,
                            "edit_distance_ratios": [self.compute_edit_distance_ratio(f, s) for f, s in zip(first_responses, second_responses)]
                        }

                        # Add task-specific metrics
                        if self.config.task == 'CODE' and self.config.compute_cyclomatic_complexity:
                            metrics["cyclomatic_scores"] = self.compute_rewards(second_responses, correct, tests)['cyclomatic']

                        # Log metrics
                        self.log_metrics(metrics, step=self.global_step)

                    except Exception as e:
                        logger.error(f"Error collecting or logging metrics in Stage I: {e}")

            except Exception as e:
                logger.error(f"Error during Stage I forward pass: {e}")
                continue

            try:
                # Optimization step
                self.optimizer.zero_grad()
                if self.config.mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()

            except Exception as e:
                logger.error(f"Error during Stage I backward pass: {e}")
                continue

            if self.global_step % self.config.logging_steps == 0:
                logger.info(f"Stage I - Step {self.global_step}, Loss: {loss.item():.4f}")



    def stage_two(self) -> None:
        """
        Stage II training: Jointly optimize both attempts with reward shaping
        to prevent collapse to non-correcting behavior.
        """
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Stage II Training"):
            self.global_step += 1
            try:
                inputs, correct, tests = self.prepare_batch(batch, turn=1)
                
                # First attempt
                first_encodings = self.model.tokenizer(
                    inputs,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_len
                ).to(self.config.device)
                
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision and torch.cuda.is_available()):
                    # Get logits for first attempt
                    first_logits = self.model(first_encodings['input_ids'], first_encodings['attention_mask'])
                    first_probs = torch.softmax(first_logits, dim=-1)
                    
                    # Generate first attempt
                    first_ids = self.model.generate_text(first_encodings, max_length=self.config.max_seq_len, temperature=1.0)
                    first_responses = self.model.tokenizer.batch_decode(first_ids, skip_special_tokens=True)
                    first_rewards = self.compute_rewards(first_responses, correct, tests)['rewards']
                    
                    # Second attempt with self-correction instruction
                    second_inputs, correct, tests = self.prepare_batch(
                        batch, 
                        turn=2,
                        prev_attempts=first_responses
                    )
                    
                    second_encodings = self.model.tokenizer(
                        second_inputs,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_seq_len
                    ).to(self.config.device)
                    
                    # Get logits for second attempt
                    second_logits = self.model(second_encodings['input_ids'], second_encodings['attention_mask'])
                    second_probs = torch.softmax(second_logits, dim=-1)
                    
                    second_ids = self.model.generate_text(second_encodings, max_length=self.config.max_seq_len)
                    second_responses = self.model.tokenizer.batch_decode(second_ids, skip_special_tokens=True)
                    second_rewards = self.compute_rewards(second_responses, correct, tests)['rewards']
                    
                    # Compute reward bonus for making progress
                    progress_bonus = self.config.alpha * (second_rewards - first_rewards)
                    
                    # Convert rewards to tensors with gradients
                    first_rewards = first_rewards.to(self.config.device)
                    second_rewards = second_rewards.to(self.config.device)
                    progress_bonus = progress_bonus.to(self.config.device)
                    
                    total_rewards = second_rewards + progress_bonus
            
                    # Compute policy loss using log probabilities
                    first_policy_loss = -torch.mean(first_probs * first_rewards.unsqueeze(-1))
                    second_policy_loss = -torch.mean(second_probs * (second_rewards + progress_bonus).unsqueeze(-1))
                    
                    # KL regularization for both attempts
                    with torch.no_grad():
                        first_ref_logits = self.ref_model(first_encodings['input_ids'], first_encodings['attention_mask'])
                        second_ref_logits = self.ref_model(second_encodings['input_ids'], second_encodings['attention_mask'])
                    
                    kl_loss = (self.compute_kl_divergence(first_logits, first_ref_logits) + 
                            self.compute_kl_divergence(second_logits, second_ref_logits)) * self.config.beta_1
                    
                    # Final loss
                    loss = first_policy_loss + second_policy_loss + kl_loss

                    try:
                        # Collect metrics
                        metrics = {
                            "total_loss": loss.item(),
                            "kl_loss": kl_loss.item(),
                            "reward_loss": -(total_rewards.mean().item()),
                            "rewards_t1": first_rewards,
                            "rewards_t2": second_rewards,
                            "edit_distance_ratios": [self.compute_edit_distance_ratio(f, s) for f, s in zip(first_responses, second_responses)]  # Fixed variable names
                        }

                        # Add task-specific metrics
                        if self.config.task == 'CODE' and self.config.compute_cyclomatic_complexity:
                            metrics["cyclomatic_scores"] = self.compute_rewards(second_responses, correct, tests)['cyclomatic']

                        # Log metrics
                        self.log_metrics(metrics, step=self.global_step)

                    except Exception as e:
                        logger.error(f"Error collecting or logging metrics in Stage II: {e}")


            except Exception as e:
                logger.error(f"Error during Stage II forward pass: {e}")
                continue

            try:
                # Optimization step
                self.optimizer.zero_grad()
                if self.config.mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()

            except Exception as e:
                logger.error(f"Error during Stage II backward pass: {e}")
                continue

            if self.global_step % self.config.logging_steps == 0:
                logger.info(
                    f"Stage II - Step {self.global_step}, Loss: {loss.item():.4f}, "
                    f"Total Reward: {total_rewards.mean().item():.4f}"
                )



    def __del__(self):
        """Cleanup wandb on deletion."""
        if self.use_wandb:
            try:
                wandb.finish()
            except Exception as e:
                logger.error(f"Error closing wandb: {e}")

    def evaluate(self) -> None:
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        total_correct_t1, total_correct_t2, total_samples = 0.0, 0.0, 0
        delta_i_to_c, delta_c_to_i = 0, 0
        bleu_scores, rouge_scores, cyclomatic_complexities = [], [], []

        try:
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc="Evaluation"):
                    try:
                        inputs, correct, tests = self.prepare_batch(batch, turn=1)
                        encodings = self.model.tokenizer(
                            inputs,
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=self.config.max_seq_len
                        ).to(self.config.device)
                    except Exception as e:
                        logger.error(f"Error during batch encoding in evaluation: {e}")
                        continue

                    try:
                        # Generate first attempt
                        first_ids = self.model.generate_text(encodings, max_length=self.config.max_seq_len, temperature=1.0)
                        first = self.model.tokenizer.batch_decode(first_ids, skip_special_tokens=True)
                        # Generate second attempt based on first
                        second_inputs, correct, tests = self.prepare_batch(
                            batch,
                            turn=2,
                            prev_attempts=first
                        )
                        second_encodings = self.model.tokenizer(
                            second_inputs,
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=self.config.max_seq_len
                        ).to(self.config.device)
                        second_ids = self.model.generate_text(second_encodings, max_length=self.config.max_seq_len, temperature=1.0)
                        second = self.model.tokenizer.batch_decode(second_ids, skip_special_tokens=True)
                        # Compute rewards
                        if self.global_step % self.config.logging_steps == 0:
                            for idx, (inp, f_resp, s_resp, corr) in enumerate(zip(inputs, first, second, correct)):
                                logger.info(f"\n=== Sample {idx + 1} ===")
                                logger.info(f"Problem:\n{batch[idx]['problem']}")
                                logger.info(f"First attempt:\n{f_resp}")
                                logger.info(f"Second attempt:\n{s_resp}")
                                logger.info(f"Correct answer:\n{corr}")
                        rewards_first = self.compute_rewards(first, correct, tests)['rewards']
                        rewards_second = self.compute_rewards(second, correct, tests)['rewards']
                    except Exception as e:
                        logger.error(f"Error during text generation or reward computation in evaluation: {e}")
                        continue

                    for i in range(len(inputs)):
                        try:
                            r1 = rewards_first[i].item()
                            r2 = rewards_second[i].item()
                            total_correct_t1 += r1
                            total_correct_t2 += r2
                            if r1 == 0 and r2 == 1:
                                delta_i_to_c += 1
                            elif r1 == 1 and r2 == 0:
                                delta_c_to_i += 1
                            total_samples += 1

                            if self.config.task == 'CODE':
                                if self.config.compute_cyclomatic_complexity:
                                    cyclomatic = self.compute_rewards([second[i]], [correct[i]], tests)['cyclomatic'][0]
                                    cyclomatic_complexities.append(cyclomatic)

                            # Compute edit distance ratio
                            ratio = self.compute_edit_distance_ratio(first[i], second[i])
                            self.edit_distance_ratios.append(ratio)
                        except Exception as e:
                            logger.error(f"Error during evaluation metrics computation for sample {i}: {e}")

            # Compute final metrics
            accuracy_t1 = total_correct_t1 / total_samples if total_samples > 0 else 0.0
            accuracy_t2 = total_correct_t2 / total_samples if total_samples > 0 else 0.0
            delta = accuracy_t2 - accuracy_t1
            delta_i_to_c_frac = delta_i_to_c / total_samples if total_samples > 0 else 0.0
            delta_c_to_i_frac = delta_c_to_i / total_samples if total_samples > 0 else 0.0

            logger.info(f"Accuracy@t1: {accuracy_t1:.4f}")
            logger.info(f"Accuracy@t2: {accuracy_t2:.4f}")
            logger.info(f"Δ(t1,t2): {delta:.4f}")
            logger.info(f"Δ_i→c(t1,t2): {delta_i_to_c_frac:.4f}")
            logger.info(f"Δ_c→i(t1,t2): {delta_c_to_i_frac:.4f}")

            if self.config.task == 'CODE':
                if self.config.compute_cyclomatic_complexity and cyclomatic_complexities:
                    avg_cyclomatic = np.mean(cyclomatic_complexities)
                    logger.info(f"Average Cyclomatic Complexity: {avg_cyclomatic:.4f}")

            self.plot_reward_history()
            self.plot_edit_distance_ratios()

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise

    def plot_reward_history(self) -> None:
        """
        Plot and save the training reward history.
        """
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(self.reward_history, label='Average Reward')
            plt.xlabel('Training Steps')
            plt.ylabel('Average Reward')
            plt.title('Training Reward Over Time')
            plt.legend()
            plt.tight_layout()
            reward_path = os.path.join(self.config.output_dir, 'training_reward.png')
            plt.savefig(reward_path)
            plt.close()
            logger.info(f"Saved reward history plot to {reward_path}.")
        except Exception as e:
            logger.error(f"Error plotting reward history: {e}")

    def plot_edit_distance_ratios(self) -> None:
        """
        Plot and save the histogram of edit distance ratios.
        """
        try:
            plt.figure(figsize=(10, 5))
            plt.hist(self.edit_distance_ratios, bins=50, color='skyblue', edgecolor='black')
            plt.xlabel('Edit Distance Ratio')
            plt.ylabel('Frequency')
            plt.title('Edit Distance Ratios between Attempts')
            plt.tight_layout()
            edit_distance_path = os.path.join(self.config.output_dir, 'edit_distance_ratios.png')
            plt.savefig(edit_distance_path)
            plt.close()
            logger.info(f"Saved edit distance ratios plot to {edit_distance_path}.")
        except Exception as e:
            logger.error(f"Error plotting edit distance ratios: {e}")


def main():
    """
    Main function to parse arguments and initiate training and evaluation.
    """
    parser = argparse.ArgumentParser(description="Advanced SCoRe System with Enhanced Features")
    parser.add_argument('--task', type=str, default='CODE', choices=['MATH', 'CODE'], help="Task type: MATH or CODE")
    parser.add_argument('--model_variant', type=str, default='Qwen/Qwen2.5-Coder-0.5B-Instruct', help="Model variant to use")
    parser.add_argument('--ablation', type=str, default='none', help="Ablation setting")
    parser.add_argument('--data_path', type=str, default='./data', help="Path to the data directory")
    parser.add_argument('--output_dir', type=str, default='./outputs', help="Directory to save outputs")
    parser.add_argument('--mixed_precision', action='store_true', help="Enable mixed precision training")
    parser.add_argument('--no_bleu', action='store_false', dest='compute_bleu', help="Disable BLEU score computation")
    parser.add_argument('--no_rouge', action='store_false', dest='compute_rouge', help="Disable ROUGE score computation")
    parser.add_argument('--no_cyclomatic', action='store_false', dest='compute_cyclomatic_complexity', help="Disable cyclomatic complexity computation")
    args = parser.parse_args()

    # Initialize configuration
    config = Config(
        task=args.task,
        model_variant='Qwen/Qwen2.5-Coder-0.5B-Instruct',  # Update default model
        ablation=args.ablation,
        data_path=args.data_path,
        output_dir=args.output_dir,
        mixed_precision=args.mixed_precision,
        compute_bleu=args.compute_bleu,
        compute_rouge=args.compute_rouge,
        compute_cyclomatic_complexity=args.compute_cyclomatic_complexity,
        logging_steps=1
    )

    try:
        config.validate()
    except Exception as e:
        logger.critical(f"Configuration validation failed: {e}")
        return

    try:
        set_seed(config.seed)
    except Exception as e:
        logger.critical(f"Failed to set seed: {e}")
        return

    # Determine data files based on task
    if config.task == 'CODE':
        train_file = os.path.join(config.data_path, 'mbpp_train.jsonl')
        val_file = os.path.join(config.data_path, 'mbpp_test.jsonl')
    else:
        logger.critical("Invalid task specified. Choose between 'MATH' and 'CODE'.")
        return

    # Check data file existence
    for file in [train_file, val_file]:
        if not os.path.isfile(file):
            logger.critical(f"Data file {file} does not exist.")
            return

    # Load datasets
    try:
        if config.task == 'CODE':
            train_data = load_json(train_file, 1000)
            val_data = load_json(val_file, 100)
        train_dataset = BaseDataset(train_data, task=config.task) 
        val_dataset = BaseDataset(val_data, task=config.task)  
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        logger.info("Datasets loaded successfully.")
    except Exception as e:
        logger.critical(f"Error loading data: {e}")
        return

    # Initialize models
    try:
        model = AdvancedModel(config.model_variant, config.device)
        ref_model = AdvancedModel(config.model_variant, config.device)
        ref_model.model.eval()
        for param in ref_model.model.parameters():
            param.requires_grad = False
        logger.info("Models initialized successfully.")
    except Exception as e:
        logger.critical(f"Error initializing models: {e}")
        return

    # Setup optimizer and scheduler
    try:
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
        total_steps = len(train_loader) * (config.num_epochs_stage_one + config.num_epochs_stage_two)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        logger.info("Optimizer and scheduler set up successfully.")
    except Exception as e:
        logger.critical(f"Error setting up optimizer and scheduler: {e}")
        return

    # Initialize trainer
    try:
        trainer = SCoReTrainer(
            model=model,
            ref_model=ref_model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config
        )
        logger.info("Trainer initialized successfully.")
    except Exception as e:
        logger.critical(f"Error initializing trainer: {e}")
        return

    # Start training and evaluation
    try:
        trainer.train()
        # trainer.evaluate()
    except Exception as e:
        logger.critical(f"Error during training/evaluation: {e}")
        return

    # Save the trained model
    try:
        model_save_path = os.path.join(config.output_dir, 'score_model.bin')
        torch.save(model.model.state_dict(), model_save_path)
        logger.info(f"Model saved to {model_save_path}.")
    except Exception as e:
        logger.critical(f"Error saving the model: {e}")
        return

    if trainer.use_wandb:
        try:
            wandb.finish()
            logger.info("Wandb run finished successfully.")
        except Exception as e:
            logger.error(f"Error finishing wandb run: {e}")


def load_model(model_path: str, model_variant: str, device: torch.device) -> AdvancedModel:
    """
    Load a saved model from disk.

    Args:
        model_path (str): Path to the saved model state dict.
        model_variant (str): Model variant identifier.
        device (torch.device): Device to load the model onto.

    Returns:
        AdvancedModel: Loaded model instance.
    """
    try:
        advanced_model = AdvancedModel(model_variant, device)
        advanced_model.model.load_state_dict(torch.load(model_path, map_location=device))
        advanced_model.model.to(device)
        advanced_model.model.eval()
        logger.info(f"Model loaded from {model_path} and moved to {device}.")
        return advanced_model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}") from e
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise RuntimeError(f"Failed to load model from {model_path}") from e


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        raise