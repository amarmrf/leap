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
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_community.llms import Ollama
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import radon.complexity as radon_complexity

import ast
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
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        logger.info(f"Seed set to {seed}.")
    except Exception as e:
        logger.error(f"Error setting seed: {e}")
        raise RuntimeError("Failed to set seed.") from e


@dataclass
class Config:
    """Configuration dataclass for evaluation parameters."""
    batch_size: int = 1
    max_seq_len: int = 4096
    max_new_tokens: int = 4096
    device: torch.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    seed: int = 42
    task: str = 'CODE'
    model_variant: str = 'qwen2.5-coder:1.5b'
    data_path: str = './data'
    output_dir: str = './outputs'
    num_workers: int = 1
    compute_cyclomatic_complexity: bool = False
    logging_steps: int = 10  # Added this line - log every 10 steps

    def validate(self) -> None:
        """
        Validate configuration parameters.
        """
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer.")
        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        if not os.path.isdir(self.output_dir):
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                logger.info(f"Created output directory at {self.output_dir}.")
            except Exception as e:
                logger.error(f"Failed to create output directory: {e}")
                raise


CODING_EXAMPLES = [
    {
        "problem": "Check if in given list of numbers, are any two numbers closer to each other than given threshold.",
        "tests": [
            "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True",
            "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False",
            "assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True"
        ],
        "solution": """def has_close_elements(numbers: List[float], threshold: float) -> bool:
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False"""
    },
    {
        "problem": "Separate nested parentheses groups into separate strings and return the list of those. Groups are balanced and not nested within each other.",
        "tests": [
            "assert separate_paren_groups('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']",
            "assert separate_paren_groups('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']",
            "assert separate_paren_groups('(()(())((())))') == ['(()(())((())))]'"
        ],
        "solution": """def separate_paren_groups(paren_string: str) -> List[str]:
    result = []
    current_string = []
    current_depth = 0

    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)

            if current_depth == 0:
                result.append(''.join(current_string))
                current_string.clear()

    return result"""
    },
    {
        "problem": "Given a positive floating point number, return its decimal part (leftover part always smaller than 1).",
        "tests": [
            "assert truncate_number(3.5) == 0.5",
            "assert abs(truncate_number(1.33) - 0.33) < 1e-6",
            "assert abs(truncate_number(123.456) - 0.456) < 1e-6"
        ],
        "solution": """def truncate_number(number: float) -> float:
    return number % 1.0"""
    }
]


def format_examples() -> str:
    """Format coding examples into a string."""
    examples_str = ""
    for i, example in enumerate(CODING_EXAMPLES, 1):
        examples_str += f"\nExample {i}:\n"
        examples_str += f"Problem: {example['problem']}\n"
        examples_str += f"Your code should pass these tests:\n"
        examples_str += "\n".join(example["tests"]) + "\n"
        examples_str += "Solution:\n```python\n"  # Using markdown code block syntax
        examples_str += example["solution"] + "\n"
        examples_str += "```\n"
    return examples_str


def get_code_first_turn_prompt(problem: str) -> str:
    """Generate the base prompt structure using Ollama's chat format."""
    return [
        {
            "role": "system",
            "content": f"""You are an expert Python programmer. Please understand the requirement and think step by step. Here are some examples of problems and their test cases:\n{format_examples()}"""
        },
        {
            "role": "user",
            "content": f"{problem}"
        }
    ]

def get_code_correction_prompt(problem: str, prev_attempt: str) -> str:
    """Generate the self-correction prompt using proper chat format."""
    return [
        {
            "role": "system",
            "content": f"You are an expert Python programmer. Please understand the requirement and think step by step. Here are some examples of problems and their test cases:\n{format_examples()}"
        },
        {
            "role": "user",
            "content": f"{problem}"
        },
        {
            "role": "assistant",
            "content": prev_attempt
        },
        {
            "role": "user",
            "content": "There might be an error in the code above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Only output the final correct Python program!"
        }
    ]

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
                return get_code_first_turn_prompt(item.get('text', item.get('prompt', '')))
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

class AdvancedModel(nn.Module):
    """
    Advanced model wrapper with Ollama integration.
    """

    def __init__(self, model_name: str, device: torch.device):
        super().__init__()
        try:
            self.model = Ollama(model=model_name)
            self.device = device
            
            # Add default generation parameters
            self.default_params = {
                "temperature": 0.0,
                "max_tokens": 4096,
                "top_p": 0.95,
            }
            
            logger.info(f"Ollama model initialized.")
        except Exception as e:
            logger.error(f"Error initializing Ollama model: {e}")
            raise RuntimeError(f"Failed to initialize Ollama model") from e

    def generate_text(self, prompt: str | List[Dict[str, str]], **kwargs) -> List[str]:
        """
        Generate text using Ollama with support for chat format.
        """
        try:
            # Log the input prompt
            # logger.info("\n" + "="*50)
            # logger.info("LLM INPUT:")
            # logger.info("-"*50)
            # logger.info(prompt)
            # logger.info("="*50)
            
            # Merge default params with any provided kwargs
            params = {**self.default_params, **kwargs}
            
            # Handle different prompt formats
            if isinstance(prompt, list):  # Chat format
                # Use Ollama's chat API directly
                response = self.model.invoke(
                    prompt,  # Pass the chat messages directly
                    **params
                )
            else:  # Legacy string format
                response = self.model.invoke(
                    prompt,
                    **params
                )
                
            # Log the output response
            # logger.info("\n" + "="*50)
            # logger.info("LLM OUTPUT:")
            # logger.info("-"*50)
            # logger.info(response)
            # logger.info("="*50 + "\n")
            
            return [response]

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


class Evaluate:
    """
    Trainer class for the SCoRe system.
    """

    def __init__(
        self,
        model: AdvancedModel,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config
    ):
        self.task = config.task
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.global_step = 0
        self.reward_history: List[float] = []
        self.edit_distance_ratios: List[float] = []
    
        self.checkpoint_file = os.path.join(self.config.output_dir, 'checkpoint-humaneval.json')
        self.last_completed_sample = self._load_checkpoint()

    def _load_checkpoint(self) -> int:
        """Load the last completed sample index from checkpoint file."""
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    logger.info(f"Resuming from sample {checkpoint['last_completed_sample'] + 1}")
                    return checkpoint['last_completed_sample']
            return -1
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return -1

    def _save_checkpoint(self, sample_index: int) -> None:
        """Save the current sample index to checkpoint file."""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump({'last_completed_sample': sample_index}, f)
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def _save_trace(self, trace_info: Dict) -> None:
        """
        Save trace information to a JSON file with pretty printing.
        """
        try:
            trace_file = os.path.join(self.config.output_dir, 'reward_traces_humaneval.json')
            with open(trace_file, 'a') as f:
                # Pretty print the JSON with indentation
                json_str = json.dumps(trace_info, indent=2)
                # Add a newline after each JSON object
                f.write(json_str + '\n\n')
        except Exception as e:
            logger.error(f"Error saving trace information: {e}")

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
                logger.debug("Cleaned:\n" + "\n".join(lines))
    
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
        """
        try:
            if isinstance(batch, dict):
                batch = [batch]
                
            if self.task == 'CODE':
                # Extract problem text and handle array wrapping
                problems = []
                for item in batch:
                    # Use prompt directly for HumanEval format
                    text = item.get('prompt', '')
                    if isinstance(text, list):
                        text = text[0] if text else ''
                    problems.append(text)
                
                # Handle array-wrapped correct answers
                correct = []
                for item in batch:
                    # Use canonical_solution for HumanEval format
                    solution = item.get('canonical_solution', '')
                    if isinstance(solution, list):
                        solution = solution[0] if solution else ''
                    correct.append(solution)
                
                # Convert test code to executable format
                test_lists = []
                for item in batch:
                    # For HumanEval, we need to construct test cases from the test field
                    test_code = item.get('test', '')
                    if test_code:
                        # Extract the assertions from the test code
                        test_lines = []
                        for line in test_code.splitlines():
                            if line.strip().startswith('assert'):
                                test_lines.append(line.strip())
                        test_lists.append(test_lines)
                    else:
                        test_lists.append([])
                
                if turn == 1:
                    inputs = [
                        get_code_first_turn_prompt(p) 
                        for p in problems
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


    def extract_function_name(self, code: str) -> str:
        """Extract the main function name from code using AST parsing."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except:
            return None
        return None

    def normalize_test_cases(self, test_cases: List[str], actual_func_name: str, expected_func_name: str) -> List[str]:
        """Replace function names in test cases to match the generated code."""
        return [test.replace(expected_func_name, actual_func_name) for test in test_cases]

    def evaluate(self) -> None:
        """
        Evaluate the model on the validation set with detailed validation steps.
        """
        self.model.eval()
        total_correct_t1, total_correct_t2, total_samples = 0.0, 0.0, 0
        delta_i_to_c, delta_c_to_i = 0, 0

        validation_metrics = {
            "first_attempt": {
                "total_samples": 0,
                "syntax_valid": 0,
                "execution_success": 0,
                "avg_execution_time": [],
                "cyclomatic_complexity": [],
            },
            "second_attempt": {
                "total_samples": 0,
                "syntax_valid": 0,
                "execution_success": 0,
                "avg_execution_time": [],
                "cyclomatic_complexity": [],
            }
        }

        try:
            logger.info("\n=== Starting Evaluation ===")
            dataset_size = len(self.val_loader.dataset)
            start_index = self.last_completed_sample + 1
            logger.info(f"Starting from sample {start_index}")
            for i in tqdm(range(start_index, dataset_size), desc="Evaluation"):
                batch = self.val_loader.dataset[i]
                logger.info(f"\n--- Processing Sample {i+1} ---")
                
                # Prepare single sample as a list
                inputs, correct, tests = self.prepare_batch([batch], turn=1)
                
                # Generate first attempt
                first = [self.model.generate_text(inputs[0])[0]]
                
                # Skip second attempt and mark it as false
                second = [""]
                
                trace_info = {
                    "sample_id": total_samples + 1,
                    "task_id": batch.get('task_id', None),
                    "problem": batch.get('prompt', ''),  # Changed from 'problem' to 'prompt'
                    "first_attempt": first[0],
                    "second_attempt": second[0],
                    "test_cases": tests[0] if tests else "",
                    "metrics": {},
                    "execution_status": {
                        "first_attempt": False,
                        "second_attempt": False
                    }
                }

                # Process first attempt
                logger.info("\n=== First Attempt ===")
                first_code = self._clean_code_response(first[0])
                
                # For HumanEval, we need to execute both the prompt and the solution
                exec_globals = {}
                all_tests_passed = True
                
                try:
                    # First execute the prompt (contains function signature and docstring)
                    exec(batch['prompt'], exec_globals)
                    
                    # Then execute the generated solution
                    exec(first_code, exec_globals)
                    logger.info("✓ Code execution successful")

                    # Execute test cases
                    test_cases = tests[0].split('\n') if tests else []
                    passed_tests = 0

                    actual_func_name = self.extract_function_name(first_code)
                    test_case = test_cases[0]
                    expected_func_name = test_case.split('(')[0].replace('assert ', '')
                    
                    if actual_func_name and actual_func_name != expected_func_name:
                        test_cases = self.normalize_test_cases(test_cases, actual_func_name, expected_func_name)
                        logger.info(f"Function name mismatch. Expected: {expected_func_name}, Got: {actual_func_name}")
            
                    
                    for j, test in enumerate(test_cases, 1):
                        if test.strip():
                            try:
                                exec(test, exec_globals)
                                passed_tests += 1
                                logger.info(f"Test {j}: ✓ {test}")
                            except AssertionError:
                                all_tests_passed = False
                                logger.info(f"Test {j}: × Failed assertion: {test}")
                            except Exception as e:
                                all_tests_passed = False
                                logger.info(f"Test {j}: × Failed with error: {str(e)}")
                
                    trace_info["execution_status"]["first_attempt"] = all_tests_passed
                    total_correct_t1 += 1 if all_tests_passed else 0
                    
                except Exception as e:
                    logger.info(f"× Code execution failed: {str(e)}")
                    trace_info["execution_status"]["first_attempt"] = False

                # Add metrics
                try:
                    edit_distance = 0  # Set to 0 since we're not doing second attempt
                    logger.info(f"\nEdit distance ratio: {edit_distance:.4f}")
                    trace_info["metrics"]["edit_distance"] = edit_distance
                    
                    if self.config.compute_cyclomatic_complexity:
                        first_complexity = self.compute_cyclomatic_complexity(first[0])
                        logger.info(f"First attempt complexity: {first_complexity:.2f}")
                        trace_info["metrics"].update({
                            "cyclomatic_first": first_complexity,
                            "cyclomatic_second": 0  # Set to 0 since we're not doing second attempt
                        })
                except Exception as e:
                    logger.error(f"Error computing metrics: {e}")
                    trace_info["metrics_error"] = str(e)

                # Save trace
                self._save_trace(trace_info)
                self._save_checkpoint(i)
                total_samples += 1

            # Log final metrics
            self._log_final_metrics(
                validation_metrics,
                total_samples,
                total_correct_t1,
                total_correct_t2,
                delta_i_to_c,
                delta_c_to_i
            )

        except Exception as e:
            logger.error(f"Error during evaluate function call: {e}")
            raise


    def _log_final_metrics(self, metrics: Dict, total_samples: int, 
                        total_correct_t1: float, total_correct_t2: float,
                        delta_i_to_c: int, delta_c_to_i: int) -> None:
        """Helper method to log final evaluation metrics."""
        logger.info("\n=== Final Validation Metrics ===")
        for attempt in ["first_attempt", "second_attempt"]:
            total = metrics[attempt]["total_samples"]
            if total > 0:
                syntax_rate = metrics[attempt]["syntax_valid"] / total * 100
                success_rate = metrics[attempt]["execution_success"] / total * 100
                avg_time = np.mean(metrics[attempt]["avg_execution_time"]) if metrics[attempt]["avg_execution_time"] else 0
                avg_complexity = np.mean(metrics[attempt]["cyclomatic_complexity"]) if metrics[attempt]["cyclomatic_complexity"] else 0
                
                logger.info(f"\n{attempt.replace('_', ' ').title()}:")
                logger.info(f"Total samples: {total}")
                logger.info(f"Syntax validation rate: {syntax_rate:.2f}%")
                logger.info(f"Test execution success rate: {success_rate:.2f}%")
                logger.info(f"Average execution time: {avg_time:.4f}s")
                if self.config.compute_cyclomatic_complexity:
                    logger.info(f"Average cyclomatic complexity: {avg_complexity:.2f}")

        # Compute final metrics
        accuracy_t1 = total_correct_t1 / total_samples if total_samples > 0 else 0.0
        accuracy_t2 = total_correct_t2 / total_samples if total_samples > 0 else 0.0
        delta = accuracy_t2 - accuracy_t1
        delta_i_to_c_frac = delta_i_to_c / total_samples if total_samples > 0 else 0.0
        delta_c_to_i_frac = delta_c_to_i / total_samples if total_samples > 0 else 0.0

        logger.info(f"\nOverall Metrics:")
        logger.info(f"Accuracy@t1: {accuracy_t1:.4f}")
        logger.info(f"Accuracy@t2: {accuracy_t2:.4f}")
        logger.info(f"Δ(t1,t2): {delta:.4f}")
        logger.info(f"Δ_i→c(t1,t2): {delta_i_to_c_frac:.4f}")
        logger.info(f"Δ_c→i(t1,t2): {delta_c_to_i_frac:.4f}")
def main():
    """Main function to parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="Code Evaluation System")
    parser.add_argument('--task', type=str, default='CODE', choices=['MATH', 'CODE'], help="Task type: MATH or CODE")
    parser.add_argument('--model_variant', type=str, default='qwen2.5-coder:1.5b', help="Model variant to use")
    parser.add_argument('--data_path', type=str, default='./data', help="Path to the data directory")
    parser.add_argument('--output_dir', type=str, default='./outputs', help="Directory to save outputs")
    parser.add_argument('--no_cyclomatic', action='store_false', dest='compute_cyclomatic_complexity', help="Disable cyclomatic complexity computation")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to the checkpoint file")
    args = parser.parse_args()

    # Initialize configuration
    config = Config(
        task=args.task,
        model_variant=args.model_variant,
        data_path=args.data_path,
        output_dir=args.output_dir,
        compute_cyclomatic_complexity=args.compute_cyclomatic_complexity,
    )


    try:
        config.validate()
        set_seed(config.seed)

        val_file = os.path.join(config.data_path, 'HumanEval.jsonl')
        val_data = load_json(val_file)
        val_dataset = BaseDataset(val_data, task=config.task)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )

        model = AdvancedModel(config.model_variant, config.device)
        model.eval()

        evaluator = Evaluate(
            model=model,
            optimizer=None,  # Not needed for inference
            scheduler=None,  # Not needed for inference
            train_loader=None,  # Not needed for inference
            val_loader=val_loader,
            config=config
        )

        evaluator.evaluate()
    except Exception as e:
        logger.critical(f"Error during evaluation: {e}")
        return


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        raise