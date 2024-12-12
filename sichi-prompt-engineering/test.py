import torch
from transformers import pipeline
import pandas as pd
from accelerate import Accelerator

accelerator = Accelerator()

print(accelerator.device)

#Importing the data
splits = {'train': 'full/train-00000-of-00001.parquet', 'test': 'full/test-00000-of-00001.parquet', 'validation': 'full/validation-00000-of-00001.parquet', 'prompt': 'full/prompt-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/google-research-datasets/mbpp/" + splits["train"])

#Append "Here is your task: " to every entry in the text column of the df
df['text'] = 'Here is your task: ' + df['text']

prompt_df = df[['text', 'test_list', ]]

def get_t1_prompts():
  prompts = []

  #for each item in the prompt dataframe
  for index, row in prompt_df.iterrows():
    test_list = ""
    #for each item in the test list
    for item in row['test_list']:
      test_list += item + "\n"

    test_string = 'Your code should pass these tests: \n' + test_list + '\n'
    prompts.append({"role": "user", "content": row['text'] + test_string})
  return prompts

def get_t2_prompts(prev_responses):
  prompts = []

  #for each item in the prompt dataframe
  for index, row in prompt_df.iterrows():
    test_list = ""
    #for each item in the test list
    for item in row['test_list']:
      test_list += item + "\n"

    test_string = 'Your code should pass these tests: \n' + test_list + '\n '
    prev_task = 'You were given this task previously: "' + row['text'] + test_string + '"'
    prev_result = ' and this was your response ' + prev_responses[index] + '\n '
    current_prompt = 'There might be an error in the code above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution in a format begining with  ```python. Only output the final correct Python program!'
    prompts.append({"role": "user", "content": prev_task + prev_result + current_prompt})
  return prompts

def get_responses(prompts):
    responses = []
    i = 0
    for prompt in prompts:
        messages = system_prompt + [prompt]
        print(i)
        outputs = pipe(
            messages,
            max_new_tokens=1000,
        )

        response = outputs[0]["generated_text"][-1]
        responses.append(response)
        i+=1
    return responses

# Intializing the model
model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Initializing the system prompt
system_prompt = [
    {"role": "system", "content": "You are an strict expert Python programmer. Only output the final correct Python program and don't give any example use-cases or other text. The code should begin with ```python. You get text input and output python only!"}
]

# Handle T1 responses and prompts
t1_prompts = get_t1_prompts()
t1_responses = get_responses(t1_prompts)
t1_prompt_content = [p['content'] for p in t1_prompts]
t1_response_content = [r['content'] for r in t1_responses]

# Handle T2 prompts and responses
t2_prompts = get_t2_prompts(t1_response_content)
t2_responses = get_responses(t2_prompts)
t2_prompt_content = [p['content'] for p in t2_prompts]
t2_response_content = [r['content'] for r in t2_responses]

print('t1_responses\n')
print(t1_responses)
print('t2_responses\n')
print(t2_responses)

prompt_response_df = pd.DataFrame({'t1_prompt': t1_prompt_content, 't1_response' : t1_response_content, 't2_prompt': t2_prompt_content, 't2_response' : t2_response_content})

print(prompt_response_df)
prompt_response_df.to_json('output.json', orient='records', lines=True)


