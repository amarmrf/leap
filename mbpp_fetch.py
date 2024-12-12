import requests
import json

def fetch_mbpp_dataset():
    url = "https://datasets-server.huggingface.co/rows"
    
    params = {
        "dataset": "google-research-datasets/mbpp",
        "config": "sanitized",
        "split": "test",
        "offset": 200,
        "length": 100
    }
    
    response = requests.get(url, params=params)
    
    # Check if request was successful
    if response.status_code == 200:
        # Save the response to a JSON file
        with open('mbpp_dataset.json', 'w') as f:
            json.dump(response.json(), f, indent=2)
        print("Dataset successfully saved to mbpp_dataset.json")
    else:
        print(f"Error fetching dataset: {response.status_code}")

if __name__ == "__main__":
    fetch_mbpp_dataset()