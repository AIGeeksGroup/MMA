import json
from typing import Dict, List, Tuple

def load_qa_pairs(file_path: str) -> Dict[str, int]:
    """
    Load QA pairs from qa_pairs_list.json and create a mapping from question to category
    """
    with open(file_path, 'r') as f:
        qa_pairs = json.load(f)
    
    # Create a dictionary mapping questions to categories
    question_to_category = {}
    for question, category in qa_pairs:
        question_to_category[question] = category
    
    return question_to_category

def add_categories_to_grades(grades_file_path: str, qa_pairs_file_path: str, output_file_path: str):
    """
    Add category information to zep_locomo_grades.json by matching questions
    """
    # Load the question to category mapping
    question_to_category = load_qa_pairs(qa_pairs_file_path)
    
    # Load the grades data
    with open(grades_file_path, 'r') as f:
        grades_data = json.load(f)
    
    # Track statistics
    total_questions = 0
    matched_questions = 0
    unmatched_questions = []
    
    # Process each user group
    for user_group, questions in grades_data.items():
        for question_data in questions:
            total_questions += 1
            question = question_data.get('question', '')
            
            # Try to find the category for this question
            if question in question_to_category:
                question_data['category'] = question_to_category[question]
                matched_questions += 1
            else:
                question_data['category'] = None
                unmatched_questions.append(question)
    
    # Save the updated data
    with open(output_file_path, 'w') as f:
        json.dump(grades_data, f, indent=2)
    
    # Print statistics
    print(f"Total questions processed: {total_questions}")
    print(f"Questions matched with categories: {matched_questions}")
    print(f"Questions without categories: {total_questions - matched_questions}")
    print(f"Match rate: {matched_questions/total_questions*100:.2f}%")
    
    if unmatched_questions:
        print(f"\nFirst 10 unmatched questions:")
        for i, question in enumerate(unmatched_questions[:10]):
            print(f"{i+1}. {question}")
    
    return grades_data

def main():
    grades_file_path = 'data/zep_locomo_grades.json'
    qa_pairs_file_path = 'data/qa_pairs_list.json'
    output_file_path = 'data/zep_locomo_grades_with_categories.json'
    
    try:
        print("Adding categories to zep_locomo_grades.json...")
        updated_data = add_categories_to_grades(grades_file_path, qa_pairs_file_path, output_file_path)
        
        print(f"\nUpdated data saved to: {output_file_path}")
        
        # Show a sample of the updated data
        print("\nSample of updated data:")
        for user_group, questions in list(updated_data.items())[:1]:  # Just show first user group
            print(f"\nUser Group: {user_group}")
            for i, question_data in enumerate(questions[:3]):  # Show first 3 questions
                print(f"  {i+1}. Question: {question_data['question']}")
                print(f"     Category: {question_data.get('category', 'Not found')}")
                print(f"     Grade: {question_data.get('grade', 'N/A')}")
                print()
            break
            
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 