import json

def calculate_accuracy():
    total_samples = 0
    first_attempt_correct = 0
    second_attempt_correct = 0
    improved = 0  # Count of cases that went from incorrect to correct
    degraded = 0  # Count of cases that went from correct to incorrect
    improved_tasks = []  # List to store improved task IDs
    degraded_tasks = []  # List to store degraded task IDs
    
    # Read the JSONL file line by line
    with open('outputs-leap/reward_traces_refine.jsonl', 'r') as file:
        content = file.read()
        # Split the content by closing braces followed by opening braces
        entries = content.split('}\n\n{')
        
        # Process each entry
        for i, entry in enumerate(entries):
            # Add back the braces except for first and last entries
            if i > 0:
                entry = '{' + entry
            if i < len(entries) - 1:
                entry = entry + '}'
                
            try:
                # Parse the JSON
                data = json.loads(entry)
                total_samples += 1
                
                first_success = data['execution_status']['first_attempt']
                second_success = data['execution_status']['second_attempt']
                task_id = data.get('task_id', 'unknown')  # Get task_id
                
                if first_success:
                    first_attempt_correct += 1
                    if not second_success:
                        degraded += 1
                        degraded_tasks.append(task_id)  # Store degraded task ID
                else:
                    if second_success:
                        improved += 1
                        improved_tasks.append(task_id)  # Store improved task ID
                
                if second_success:
                    second_attempt_correct += 1
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                continue
            except KeyError as e:
                print(f"Missing key: {e}")
                continue
    
    # Calculate accuracies
    if total_samples > 0:
        first_attempt_accuracy = (first_attempt_correct / total_samples) * 100
        second_attempt_accuracy = (second_attempt_correct / total_samples) * 100
        delta_accuracy = second_attempt_accuracy - first_attempt_accuracy
        
        # Calculate improvement and degradation rates
        improvement_rate = (improved / (total_samples - first_attempt_correct) * 100) if (total_samples - first_attempt_correct) > 0 else 0
        degradation_rate = (degraded / first_attempt_correct * 100) if first_attempt_correct > 0 else 0
        
        print(f"Total samples: {total_samples}")
        print(f"First attempt correct: {first_attempt_correct}")
        print(f"Second attempt correct: {second_attempt_correct}")
        print(f"First attempt accuracy: {first_attempt_accuracy:.2f}%")
        print(f"Second attempt accuracy: {second_attempt_accuracy:.2f}%")
        print(f"Delta accuracy (T2-T1): {delta_accuracy:.2f}%")
        print(f"Improved (incorrect→correct): {improved} ({improvement_rate:.2f}%)")
        print("Improved tasks:", improved_tasks)
        print(f"Degraded (correct→incorrect): {degraded} ({degradation_rate:.2f}%)")
        print("Degraded tasks:", degraded_tasks)
    else:
        print("No valid samples found")

if __name__ == "__main__":
    calculate_accuracy()