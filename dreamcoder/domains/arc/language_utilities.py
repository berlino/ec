"""
language_utilities.py | Author : Catherine Wong
Utility functions for handling language data.
"""
import json
import pickle

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

def analysis_write_frontier_programs_tokens_csv(args):
    """Writes out programs and tokens from a pre-loaded frontier file to a CSV"""
    preload_frontiers_file = args["preload_frontiers"]
    with open(preload_frontiers_file, "rb") as f:
        preloaded_frontiers = pickle.load(f)
    tasks_to_preloaded_frontiers = {
        task.name : frontier
        for task, frontier in preloaded_frontiers.items() if not frontier.empty
    }
    print(f"Writing out tokens from {len(tasks_to_preloaded_frontiers)} frontiers.")
    output_csv = preload_frontiers_file + ".csv"
    DELIMITER = ","
    with open(output_csv, "w") as f:
        header = ["task_id", "program", "program_tokens"]
        f.write(DELIMITER.join(header)+ "\n")
        for task_name, frontier in sorted(tasks_to_preloaded_frontiers.items()):
            for entry in frontier.entries:
                program = entry.program
                tokens = " ".join(program.left_order_tokens())
                csv_row = [task_name, str(program), str(tokens)]
                f.write(DELIMITER.join(csv_row) + "\n")
    
    