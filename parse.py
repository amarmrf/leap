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

def export_second_attempts(parsed_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for obj in parsed_data:
            if 'second_attempt' in obj:
                f.write(obj['second_attempt'])
                f.write('\n\n---\n\n')  # Separator between entries

def main():
    from datetime import datetime
    input_file = "outputs-leap/reward_traces_leap.jsonl"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"second_attempts_{timestamp}.txt"
    
    parsed_data = parse_json_file(input_file)
    export_second_attempts(parsed_data, output_file)
    
    
    # Print summary
    print(f"\nParsing Summary:")
    print(f"Successfully parsed {len(parsed_data)} valid JSON objects")
    if parsed_data:
        print("Fields in first object:", list(parsed_data[0].keys()))
        print("Sample IDs found:", sorted(set(obj['sample_id'] for obj in parsed_data)))
        print(f"\nSecond attempts exported to {output_file}")

if __name__ == "__main__":
    main()