import json
import os

def extract_mbpp_dataset(input_file, output_file):
    """Extract MBPP dataset content to a new JSON file.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to save the extracted JSON file
    """
    # Read and parse JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Only create output directory if there's a directory path
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract relevant fields from each test case
    processed_data = []
    for item in data:
        test_case = {
            'task_id': item['row']['task_id'],
            'prompt': item['row']['prompt'],
            'code': item['row']['code'],
            'test_list': item['row']['test_list']
        }
        processed_data.append(test_case)
    
    # Write formatted JSON to output file
    with open(output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f'Dataset extracted to {output_file}')


if __name__ == '__main__':
    input_file = 'mbpp_dataset.json'
    output_file = 'sanitized_mbpp_dataset_train.json'
    
    extract_mbpp_dataset(input_file, output_file)