"""
Batch driver for LOCOMO/ScreenshotVQA experiments. Spawns subprocesses to run
`run_instance.py` per global index to isolate memory and process state.
"""
import os
import json
import time
import argparse
import numpy as np
import subprocess
from tqdm import tqdm
from conversation_creator import ConversationCreator



def parse_args():
    """Parse CLI arguments for batch experiment driver."""
    parser = argparse.ArgumentParser(description="Multi-Modal Memory Illustration")
    parser.add_argument("--agent_name", type=str, choices=['gpt-long-context', 'mma', 'siglip', 'gemini-long-context'])
    parser.add_argument("--dataset", type=str, default="LOCOMO", choices=['LOCOMO', 'ScreenshotVQA'])
    parser.add_argument("--num_exp", type=int, default=100)
    parser.add_argument("--load_db_from", type=str, default=None)
    parser.add_argument("--num_images_to_accumulate", default=None, type=int)
    parser.add_argument("--global_idx", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="gpt-4.1", help="Model name to use for gpt-long-context agent")
    parser.add_argument("--config_path", type=str, default=None, help="Config file path for mma agent")
    parser.add_argument("--force_answer_question", action="store_true", default=False)
    parser.add_argument("--run_id", type=str, default=None, help="Unique run folder name")
    return parser.parse_args()

def run_with_chunks_and_questions_subprocess(args, global_idx, run_id):
    """Run a single-index experiment via subprocess to isolate memory/process state."""
    # Build command arguments
    cmd = [
        'python', 'run_instance.py',
        '--agent_name', args.agent_name,
        '--dataset', args.dataset,
        '--global_idx', str(global_idx),
        '--num_exp', str(args.num_exp),
        '--run_id', run_id,
    ]
    
    # Add optional arguments
    if args.model_name:
        cmd.extend(['--model_name', args.model_name])
    if args.config_path:
        cmd.extend(['--config_path', args.config_path])
    if args.force_answer_question:
        cmd.append('--force_answer_question')
    
    try:
        # Run the subprocess with real-time output
        print(f"Running subprocess for global_idx {global_idx}")
        print("=" * 50)
        
        process = subprocess.Popen(
            cmd, 
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')  # Print without adding extra newline
        
        # Wait for process to complete and get return code
        return_code = process.wait()
        
        print("=" * 50)
        print(f"Subprocess completed for global_idx {global_idx} with return code {return_code}")
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
            
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed for global_idx {global_idx} with return code {e.returncode}")
        raise
    except Exception as e:
        print(f"Unexpected error running subprocess for global_idx {global_idx}: {e}")
        raise

def main():
    """Batch over all indices (or selected), invoking subprocess runs per index."""
    
    args = parse_args()
    conversation_creator = ConversationCreator(args.dataset, args.num_exp)

    run_id = args.run_id or time.strftime("run-%Y%m%d-%H%M%S")

    if args.agent_name == 'gpt-long-context':
        with_instructions = False
    else: 
        with_instructions = True

    all_chunks = conversation_creator.chunks(with_instructions=with_instructions)
    all_queries_and_answers = conversation_creator.get_query_and_answer()

    for global_idx, (chunks, queries_and_answers) in enumerate(zip(all_chunks, all_queries_and_answers)):
        
        if args.global_idx is not None and global_idx != args.global_idx:
            continue
        
        run_with_chunks_and_questions_subprocess(args, global_idx, run_id)

if __name__ == '__main__':
    main()
