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
    max_seq_len: int = 2048
    max_new_tokens: int = 1024
    device: torch.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    seed: int = 42
    task: str = 'CODE'
    model_variant: str = 'qwen2.5-coder:3b'
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


def format_examples() -> str:
    """Format coding examples into a string."""
    examples_str = ""
    for i, example in enumerate(CODING_EXAMPLES, 1):
        examples_str += f"\nExample {i}:\n"
        examples_str += f"Problem: {example['problem']}\n"
        examples_str += f"Your code should pass these tests:\n"
        examples_str += "\n".join(example["tests"]) + "\n"
        examples_str += "[BEGIN]\n"
        examples_str += example["solution"] + "\n"
        examples_str += "[DONE]\n"
    return examples_str


def get_code_first_turn_prompt(problem: str) -> str:
    """Generate the base prompt structure using Ollama's chat format."""
    return [
        {
            "role": "system",
            "content": f"You are an expert Python programmer. Here are some examples of problems and their test cases:\n{format_examples()}"
        },
        {
            "role": "user",
            "content": f"Now please solve this problem:\n{problem}"
        }
    ]

def get_code_correction_prompt(problem: str, prev_attempt: str) -> str:
    """Generate the self-correction prompt using proper chat format."""
    return [
        {
            "role": "system",
            "content": f"You are an expert Python programmer. Here are some examples of problems and their test cases:\n{format_examples()}"
        },
        {
            "role": "user",
            "content": f"Now please solve this problem:\n{problem}"
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
                "temperature": 0.7,
                "max_tokens": 2048,
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
                # Convert chat format to string
                formatted_prompt = ""
                for message in prompt:
                    role = message['role']
                    content = message['content']
                    if role == 'system':
                        formatted_prompt += f"System: {content}\n\n"
                    elif role == 'user':
                        formatted_prompt += f"User: {content}\n\n"
                    elif role == 'assistant':
                        formatted_prompt += f"Assistant: {content}\n\n"
                
                response = self.model.invoke(
                    formatted_prompt,
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

class Evaluate:
    """
    Trainer class for the SCoRe system.
    """

    def __init__(
        self,
        model: AdvancedModel,
        optimizer: Any,
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


    def evaluate(self) -> None:
        """
        Evaluate the model on the validation set with standard MBPP metrics and save detailed traces.
        """
        self.model.eval()
        metrics = {
            'total_samples': 0,
            'first_attempt': {
                'solved_problems': 0,
                'syntax_valid': 0,
                'avg_execution_time': [],
                'complexities': []
            },
            'second_attempt': {
                'solved_problems': 0,
                'syntax_valid': 0,
                'avg_execution_time': [],
                'complexities': []
            }
        }

        # Track state transitions
        correct_t1 = 0  # Correct on first attempt
        correct_t2 = 0  # Correct on second attempt
        delta_i_to_c = 0  # Incorrect to Correct transitions
        delta_c_to_i = 0  # Correct to Incorrect transitions

        logger.info("Starting evaluation...")
        
        try:
            for i in tqdm(range(len(self.val_loader.dataset)), desc="Evaluating samples"):
                batch = self.val_loader.dataset[i]
                metrics['total_samples'] += 1
                
                # Prepare trace info dictionary
                trace_info = {
                    'sample_id': i,
                    'problem': batch.get('text', batch.get('prompt', '')),
                    'attempts': []
                }
                
                # First attempt
                inputs, correct, tests = self.prepare_batch([batch], turn=1)
                test_cases = tests[0].split('\n') if tests else []
                
                first_response = self.model.generate_text(inputs[0])[0]
                first_results = self._validate_attempt(first_response, test_cases)

                # Track first attempt success
                first_attempt_correct = (first_results['syntax_valid'] and 
                                    first_results['test_results']['passed_tests'] == 
                                    first_results['test_results']['total_tests'])
                if first_attempt_correct:
                    correct_t1 += 1

                # Record first attempt in trace
                trace_info['attempts'].append({
                    'turn': 1,
                    'generated_code': first_results['code'],
                    'syntax_valid': first_results['syntax_valid'],
                    'test_results': first_results['test_results'],
                    'complexity': first_results['complexity'],
                    'is_correct': first_attempt_correct
                })

                # Update first attempt metrics
                if first_results['syntax_valid']:
                    metrics['first_attempt']['syntax_valid'] += 1
                    metrics['first_attempt']['avg_execution_time'].append(
                        first_results['test_results']['execution_time']
                    )
                    if first_results['complexity'] is not None:
                        metrics['first_attempt']['complexities'].append(first_results['complexity'])
                    if first_results['test_results']['passed_tests'] == first_results['test_results']['total_tests']:
                        metrics['first_attempt']['solved_problems'] += 1

                # Second attempt
                second_inputs, _, _ = self.prepare_batch([batch], turn=2, prev_attempts=[first_response])
                second_response = self.model.generate_text(second_inputs[0])[0]
                second_results = self._validate_attempt(second_response, test_cases)

                # Track second attempt success
                second_attempt_correct = (second_results['syntax_valid'] and 
                                    second_results['test_results']['passed_tests'] == 
                                    second_results['test_results']['total_tests'])
                if second_attempt_correct:
                    correct_t2 += 1

                # Track transitions
                if not first_attempt_correct and second_attempt_correct:
                    delta_i_to_c += 1
                elif first_attempt_correct and not second_attempt_correct:
                    delta_c_to_i += 1

                # Record second attempt in trace
                trace_info['attempts'].append({
                    'turn': 2,
                    'generated_code': second_results['code'],
                    'syntax_valid': second_results['syntax_valid'],
                    'test_results': second_results['test_results'],
                    'complexity': second_results['complexity'],
                    'is_correct': second_attempt_correct
                })

                # Add transition information to trace
                trace_info['transitions'] = {
                    'improved': not first_attempt_correct and second_attempt_correct,
                    'degraded': first_attempt_correct and not second_attempt_correct,
                    'first_correct': first_attempt_correct,
                    'second_correct': second_attempt_correct
                }

                # Save trace for this sample
                self._save_trace(trace_info)

                # Update second attempt metrics
                if second_results['syntax_valid']:
                    metrics['second_attempt']['syntax_valid'] += 1
                    metrics['second_attempt']['avg_execution_time'].append(
                        second_results['test_results']['execution_time']
                    )
                    if second_results['complexity'] is not None:
                        metrics['second_attempt']['complexities'].append(second_results['complexity'])
                    if second_results['test_results']['passed_tests'] == second_results['test_results']['total_tests']:
                        metrics['second_attempt']['solved_problems'] += 1

            # Calculate final metrics
            total_samples = metrics['total_samples']
            accuracy_t1 = correct_t1 / total_samples if total_samples > 0 else 0
            accuracy_t2 = correct_t2 / total_samples if total_samples > 0 else 0
            delta = accuracy_t2 - accuracy_t1

            # Log final metrics
            self._log_final_metrics(
                metrics=metrics,
                total_samples=total_samples,
                total_correct_t1=correct_t1,
                total_correct_t2=correct_t2,
                delta_i_to_c=delta_i_to_c,
                delta_c_to_i=delta_c_to_i
            )

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise  
    def _validate_attempt(self, code: str, tests: List[str]) -> Dict[str, Any]:
        """
        Pure validation logic for a single code attempt.
        
        Args:
            code: The code to validate
            tests: List of test cases
        
        Returns:
            Dict containing validation results:
            {
                'syntax_valid': bool,
                'test_results': {
                    'passed_tests': int,
                    'total_tests': int,
                    'execution_time': float,
                    'execution_error': Optional[str]
                },
                'complexity': Optional[float]
            }
        """
        results = {
            'syntax_valid': False,
            'test_results': {
                'passed_tests': 0,
                'total_tests': len(tests),
                'execution_time': 0.0,
                'execution_error': None,
                'test_details': []  # Add test-level details
            },
            'complexity': None,
            'code': self._clean_code_response(code) 
        }
        
        # Syntax validation
        try:
            ast.parse(results['code'])
            results['syntax_valid'] = True
        except SyntaxError:
            return results

        # Test execution
        start_time = time.time()
        exec_globals = {}
        
        try:
            exec(results['code'], exec_globals)
            
            # Run and track individual test results
            for test in tests:
                test_result = {
                    'test': test,
                    'passed': False,
                    'error': None
                }
                
                if test.strip():
                    try:
                        exec(test, exec_globals)
                        test_result['passed'] = True
                        results['test_results']['passed_tests'] += 1
                    except Exception as e:
                        test_result['error'] = str(e)
                        
                results['test_results']['test_details'].append(test_result)
                    
        except Exception as e:
            results['test_results']['execution_error'] = str(e)
        finally:
            results['test_results']['execution_time'] = time.time() - start_time

        if self.config.compute_cyclomatic_complexity:
            try:
                complexity = radon_complexity.cc_visit(results['code'])
                results['complexity'] = np.mean([block.complexity for block in complexity]) if complexity else 0.0
            except Exception:
                pass

        return results

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
    parser.add_argument('--model_variant', type=str, default='qwen2.5-coder:3b', help="Model variant to use")
    parser.add_argument('--data_path', type=str, default='./data', help="Path to the data directory")
    parser.add_argument('--output_dir', type=str, default='./outputs', help="Directory to save outputs")
    parser.add_argument('--no_cyclomatic', action='store_false', dest='compute_cyclomatic_complexity', help="Disable cyclomatic complexity computation")
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

        val_file = os.path.join(config.data_path, 'mbpp_test.jsonl')
        val_data = load_json(val_file ,10)
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