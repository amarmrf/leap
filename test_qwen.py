from langchain_ollama.llms import OllamaLLM
from datasets import load_dataset
import re


# Load Dataset 
humaneval = load_dataset("openai/openai_humaneval")["test"]
print(humaneval[0]["prompt"])

# Initialize Model
llm = OllamaLLM(model="qwen2.5-coder:0.5b")

# # t1 Eval
# prompt = humaneval[0]["prompt"]
# response = llm.invoke(prompt)
# print("Response: ", response)
# pattern = r"```python\s+(.*?)\s+```"
# # Use re.DOTALL to match across newlines
# python_blocks = re.findall(pattern, response, re.DOTALL)
# print("Python Code: ", python_blocks[0])

# # t2 Eval by passing t1's python code
# prompt2 = "This was the function you were previously asked to implement: \n"
# prompt2 += prompt 
# prompt2 += " and this was your solution: \n\n\n"
# prompt2 += python_blocks[0]
# prompt2 += "\n There may or may not be issues with your previous solution. Analyze it and generate new solution."
# print(prompt2)
# response = llm.invoke(prompt2)
# python_blocks = re.findall(pattern, response, re.DOTALL)
# print("Python Code: ", python_blocks[0])

# # t3 Eval by passing t2's python code
# prompt2 = "This was the function you were previously asked to implement: \n"
# prompt2 += prompt 
# prompt2 += " and this was your solution: \n\n\n"
# prompt2 += python_blocks[0]
# prompt2 += "\n There may or may not be issues with your previous solution. Analyze it and generate new solution."
# print(prompt2)
# response = llm.invoke(prompt2)
# python_blocks = re.findall(pattern, response, re.DOTALL)
# print("Python Code: ", python_blocks[0])

import re
from typing import Any

def evaluate_function(llm: Any, humaneval, iterations: int):
    """
    Evaluate a function by iterating through an analysis and improvement loop.

    Parameters:
        llm (Any): The initialized language model object.
        humaneval (list): List containing the prompts for evaluation.
        iterations (int): Number of times to iterate through the evaluation process.

    Returns:
        list: A list of Python code blocks generated during the iterations.
    """
    # Initialize results
    results = []

    # Initial setup
    prompt = humaneval[0]["prompt"]
    response = llm.invoke(prompt)
    pattern = r"```python\s+(.*?)\s+```"
    python_blocks = re.findall(pattern, response, re.DOTALL)
    if python_blocks:
        results.append(python_blocks[0])
    else:
        print("No valid Python code block found in the initial response.")
        return results

    # Iterative refinement loop
    for _ in range(iterations):
        # Build the evaluation prompt
        prompt2 = (
            f"This was the function you were previously asked to implement: \n{prompt}"
            f" and this was your solution: \n\n\n{results[-1]}"
            f"\n There may or may not be issues with your previous solution. Analyze it and generate a new solution. If you think your previous solution is wrong, it is ok to try a new approach"
        )
        response = llm.invoke(prompt2)
        python_blocks = re.findall(pattern, response, re.DOTALL)
        
        if python_blocks:
            results.append(python_blocks[0])
            print("Python Code (latest iteration): ", python_blocks[0])
        else:
            print("No valid Python code block found in the response during iteration.")
            break

    return results

def evaluate_with_humaneval(function_code, test_code, entrypoint_function):
    full_code = f"""
{function_code}

{test_code}

# Entrypoint
if __name__ == "__main__":
    check({entrypoint_function})
    print("All tests passed!")
"""
    print(full_code)
    # Execute the dynamically generated code
    try:
        exec(full_code)
        return "Passed"
    except AssertionError as e:
        print("Assertion failed:", e)
        return "Failed"
    except Exception as e:
        print("Execution error:", e)
        return "ExecutionFailed"

# iterations = 0 mean there is no turn2
# basically the function does 1 prompting by default 
# iterations add extra "interations" turns
results = evaluate_function(llm=llm, humaneval=humaneval, iterations=4)
final_eval = []
for res in results:
    final_eval.append(evaluate_with_humaneval(function_code=res, test_code=humaneval[0]["test"], entrypoint_function=humaneval[0]["entry_point"]))

print(final_eval)