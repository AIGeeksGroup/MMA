import json
from collections import defaultdict

def analyze_grades_by_category(file_path: str):
    """
    Analyze the distribution of grades by category
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Initialize counters
    category_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0})
    overall_stats = {'total': 0, 'correct': 0, 'incorrect': 0}
    
    # Process each user group
    for user_group, questions in data.items():
        for question_data in questions:
            category = question_data.get('category')
            grade = question_data.get('grade', False)
            
            if category is not None:
                category_stats[category]['total'] += 1
                overall_stats['total'] += 1
                
                if grade:
                    category_stats[category]['correct'] += 1
                    overall_stats['correct'] += 1
                else:
                    category_stats[category]['incorrect'] += 1
                    overall_stats['incorrect'] += 1
    
    # Print results
    print("Grade Distribution by Category:")
    print("=" * 50)
    
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"Category {category}:")
        print(f"  Total questions: {stats['total']}")
        print(f"  Correct: {stats['correct']}")
        print(f"  Incorrect: {stats['incorrect']}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print()
    
    # Overall statistics
    overall_accuracy = (overall_stats['correct'] / overall_stats['total']) * 100 if overall_stats['total'] > 0 else 0
    print("Overall Statistics:")
    print("=" * 50)
    print(f"Total questions: {overall_stats['total']}")
    print(f"Correct: {overall_stats['correct']}")
    print(f"Incorrect: {overall_stats['incorrect']}")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    
    return category_stats, overall_stats

def main():
    file_path = 'data/zep_locomo_grades_with_categories.json'
    
    try:
        print("Analyzing grades by category...")
        category_stats, overall_stats = analyze_grades_by_category(file_path)
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 