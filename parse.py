import json

def parse_json_file(input_file):
    valid_json_lines = []
    current_json = ""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        # Read the file line by line
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Check if this looks like the start of a new JSON object
            if line.startswith('{') and current_json:
                try:
                    json_obj = json.loads(current_json)
                    valid_json_lines.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON at line {line_num-1}: {str(e)}")
                    print(f"Problematic JSON: {current_json[:100]}...")
                current_json = line
                continue
                
            # Accumulate lines until we have a complete JSON object
            current_json += line
                
    if current_json:  # Handle any remaining JSON
        try:
            json_obj = json.loads(current_json)
            valid_json_lines.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"Error parsing final JSON object: {str(e)}")
            print(f"First 100 chars: {current_json[:100]}...")
            print(f"Last 100 chars: {current_json[-100:] if len(current_json) > 100 else current_json}")
                
    return valid_json_lines

def main():
    input_file = "outputs-leap/reward_traces_leap.jsonl"  # Adjust path as needed
    parsed_data = parse_json_file(input_file)
    
    # Print some stats about the parsed data
    print(f"\nParsing Summary:")
    print(f"Successfully parsed {len(parsed_data)} valid JSON objects")
    if parsed_data:
        print("Fields in first object:", list(parsed_data[0].keys()))
        print("Sample IDs found:", sorted(set(obj['sample_id'] for obj in parsed_data)))

if __name__ == "__main__":
    main()