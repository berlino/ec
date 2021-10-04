"""
language_utilities.py | Author : Catherine Wong
Utility functions for handling language data.
"""
import json

def add_task_language_annotations(train_tasks, test_tasks, language_annotations_data_file):
    with open(language_annotations_data_file, "r") as f:
        language_annotations_data = json.load(f)
    
    num_annotated = 0
    for tasks in [train_tasks, test_tasks]:
        for task in tasks:
            if task.name in language_annotations_data:
                task.sentences = language_annotations_data[task.name]
                num_annotated += 1
    print(f"Loaded language annotations from {language_annotations_data_file} for {num_annotated}/{len(train_tasks) + len(test_tasks)} tasks.")
    return train_tasks, test_tasks
