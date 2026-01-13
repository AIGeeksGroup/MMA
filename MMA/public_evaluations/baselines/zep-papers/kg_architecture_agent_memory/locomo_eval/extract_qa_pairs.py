import json
from typing import List, Tuple

def extract_qa_pairs(file_path: str) -> List[Tuple[str, int]]:
    """
    Extract all QA pairs from locomo.json and return a list of tuples (question, category)
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    qa_pairs = []
    
    # Iterate through each object in the array
    for item in data:
        if 'qa' in item:
            # Extract each QA pair from the qa array
            for qa in item['qa']:
                question = qa.get('question', '')
                category = qa.get('category', -1)  # Default to -1 if category is missing
                qa_pairs.append((question, category))
    
    return qa_pairs

def main():
    file_path = 'data/locomo.json'
    
    try:
        qa_pairs = extract_qa_pairs(file_path)
        
        print(f"Total QA pairs extracted: {len(qa_pairs)}")
        print("\nFirst 10 QA pairs:")
        for i, (question, category) in enumerate(qa_pairs[:10]):
            print(f"{i+1}. Question: {question}")
            print(f"   Category: {category}")
            print()
        
        # Save the results to a new file
        output_file = 'data/qa_pairs_list.json'
        with open(output_file, 'w') as f:
            json.dump(qa_pairs, f, indent=2)
        
        print(f"QA pairs saved to: {output_file}")
        
        # Also print some statistics
        categories = [category for _, category in qa_pairs]
        unique_categories = set(categories)
        print(f"\nUnique categories found: {sorted(unique_categories)}")
        
        category_counts = {}
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print("\nCategory distribution:")
        for category in sorted(category_counts.keys()):
            print(f"Category {category}: {category_counts[category]} questions")
            
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 