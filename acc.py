import json
import os

def analyze_results(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return None

    unique_results = {
        'Stage1_First': {'attempts': set()},
        'Stage1_Second': {'attempts': set()},
        'Stage2_First': {'attempts': set()},
        'Stage2_Second': {'attempts': set()}
    }

    results = {
        'Stage1_First': {'passed': 0, 'total': 0},
        'Stage1_Second': {'passed': 0, 'total': 0},
        'Stage2_First': {'passed': 0, 'total': 0},
        'Stage2_Second': {'passed': 0, 'total': 0}
    }
    
    current_json = ""
    brace_count = 0
    
    print("Starting file analysis...")
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            brace_count += line.count('{') - line.count('}')
            current_json += line
            
            if brace_count == 0 and current_json:  # We have a complete JSON object
                try:
                    result = json.loads(current_json)
                    if "test_cases" in result:
                        stage = result.get("stage", "")
                        attempt = result.get("attempt", "")
                        
                        # Only count as passed if ALL test cases passed
                        all_tests_passed = (result['test_cases']['passed'] == result['test_cases']['total'])
                        
                        if "stage_1" in stage:
                            stage_key = f"Stage1_{attempt.capitalize()}"
                            if stage_key in results:
                                results[stage_key]['passed'] += 1 if all_tests_passed else 0
                                results[stage_key]['total'] += 1
                        elif "stage_2" in stage:
                            stage_key = f"Stage2_{attempt.capitalize()}"
                            if stage_key in results:
                                results[stage_key]['passed'] += 1 if all_tests_passed else 0
                                results[stage_key]['total'] += 1
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                current_json = ""

    print("\nFinal counts:")
    for stage, data in results.items():
        if data['total'] > 0:
            print(f"{stage}: {data['passed']}/{data['total']} attempts with all tests passing")

    final_results = {
        'Stage1_First': {
            'percentage': results['Stage1_First']['passed'] / results['Stage1_First']['total'] if results['Stage1_First']['total'] > 0 else 0,
            'passed': results['Stage1_First']['passed'],
            'total': results['Stage1_First']['total']
        },
        'Stage1_Second': {
            'percentage': results['Stage1_Second']['passed'] / results['Stage1_Second']['total'] if results['Stage1_Second']['total'] > 0 else 0,
            'passed': results['Stage1_Second']['passed'],
            'total': results['Stage1_Second']['total']
        },
        'Stage2_First': {
            'percentage': results['Stage2_First']['passed'] / results['Stage2_First']['total'] if results['Stage2_First']['total'] > 0 else 0,
            'passed': results['Stage2_First']['passed'],
            'total': results['Stage2_First']['total']
        },
        'Stage2_Second': {
            'percentage': results['Stage2_Second']['passed'] / results['Stage2_Second']['total'] if results['Stage2_Second']['total'] > 0 else 0,
            'passed': results['Stage2_Second']['passed'],
            'total': results['Stage2_Second']['total']
        }
    }

    return final_results

if __name__ == "__main__":
    file_path = "outputs/reward_traces.jsonl"
    results = analyze_results(file_path)
    
    if results:
        print("\nFinal Results:")
        print(f"Stage 1 First Attempt: {results['Stage1_First']['percentage']:.2%} ({results['Stage1_First']['passed']}/{results['Stage1_First']['total']})")
        print(f"Stage 1 Second Attempt: {results['Stage1_Second']['percentage']:.2%} ({results['Stage1_Second']['passed']}/{results['Stage1_Second']['total']})")
        print(f"Stage 2 First Attempt: {results['Stage2_First']['percentage']:.2%} ({results['Stage2_First']['passed']}/{results['Stage2_First']['total']})")
        print(f"Stage 2 Second Attempt: {results['Stage2_Second']['percentage']:.2%} ({results['Stage2_Second']['passed']}/{results['Stage2_Second']['total']})")
    else:
        print("Failed to analyze results.")