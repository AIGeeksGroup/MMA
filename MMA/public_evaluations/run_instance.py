#!/usr/bin/env python3

"""
Single-run driver for experiments: loads conversation chunks and Q&A pairs,
persists per-run state, and executes either memorization or answering phase.
"""

import os
import json
import argparse
import time
from tqdm import tqdm
from agent import AgentWrapper
from conversation_creator import ConversationCreator
import logging
logging.basicConfig(level=logging.INFO)

def parse_args():
    """Parse CLI arguments for a single experiment run."""
    parser = argparse.ArgumentParser(description="Run instance with chunks and questions")
    parser.add_argument("--agent_name", type=str, required=True, choices=['gpt-long-context', 'mma', 'siglip', 'gemini-long-context'])
    parser.add_argument("--dataset", type=str, default="LOCOMO", choices=['LOCOMO', 'ScreenshotVQA'])
    parser.add_argument("--global_idx", type=int, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4.1", help="Model name to use for gpt-long-context agent")
    parser.add_argument("--config_path", type=str, default=None, help="Config file path for mma agent")
    parser.add_argument("--force_answer_question", action="store_true", default=False)
    parser.add_argument("--num_exp", type=int, default=100, help="Number of experiments")
    parser.add_argument("--run_id", type=str, default=None, help="Unique run identifier to create a new results folder")
    return parser.parse_args()


def run_with_chunks_and_questions(
        args,
        global_idx,
        chunks, 
        queries_and_answers):
    """
    Run a single experiment:
    - Memorization phase: ingest chunks (text or images) into memory
    - Answering phase: answer questions against the stored memory

    Args:
        args: Parsed CLI arguments
        global_idx: Index selecting which conversation subset to run
        chunks: List of chunks for memorization phase
        queries_and_answers: List of [idx, question, answer, metadata]
    """

    base_dir = f"./results/{args.agent_name}_{args.dataset}/"
    if args.agent_name == 'gpt-long-context' or args.agent_name == 'gemini-long-context':
        base_dir = f"./results/{args.agent_name}_{args.dataset}-{args.model_name}/"

    run_dir_name = args.run_id if args.run_id else time.strftime("run-%Y%m%d-%H%M%S")
    run_base_dir = os.path.join(base_dir, run_dir_name)
    
    out_dir = os.path.join(run_base_dir, str(global_idx))
    
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_dir) and os.listdir(out_dir): # Check if directory is not empty
        agent = AgentWrapper(args.agent_name, load_agent_from=out_dir, model_name=args.model_name, config_path=args.config_path)
    else:
        if args.agent_name == 'mma':
            if os.path.exists(os.path.expanduser(f"~/.mma/sqlite.db")):
                # need to delete the existing db
                os.system(f"rm -rf ~/.mma/sqlite.db*")
        agent = AgentWrapper(args.agent_name, model_name=args.model_name, config_path=args.config_path)

    current_step_path = os.path.join(out_dir, "current_step.txt")
    chunks_path = os.path.join(out_dir, "chunks.json")
    results_path = os.path.join(out_dir, "results.json")

    if os.path.exists(current_step_path):
        with open(current_step_path, "rb") as f:
            current_step = int(f.read().decode())
    else:
        current_step = -1

    if os.path.exists(chunks_path):
        with open(chunks_path, "r") as f:
            existing_chunks = json.load(f)
    else:
        existing_chunks = []

    for idx, next_chunk in tqdm(enumerate(chunks), total=len(chunks)):
        if idx <= current_step or args.force_answer_question:
            continue

        if args.dataset == 'ScreenshotVQA':
            image_uris, timestamp = [x[0] for x in next_chunk], [x[1] for x in next_chunk]
            response = agent.send_message(message=None, 
                                          image_uris=image_uris, 
                                          memorizing=True,
                                          timestamp=timestamp)
            existing_chunks.append({
                'image_uri': image_uris,
                'response': response
            })
        else:
            prompt = next_chunk
            response = agent.send_message(prompt, memorizing=True)
            existing_chunks.append({
                'message': prompt,
                'response': response
            })

        if args.agent_name == 'mma':
            agent.save_agent(out_dir)
            # CHANGED: Use full path
            with open(current_step_path, "wb") as f:
                f.write(str(idx).encode())
            # CHANGED: Use full path
            with open(chunks_path, "w") as f:
                json.dump(existing_chunks, f, indent=2)

    agent.save_agent(out_dir)
    agent.prepare_before_asking_questions()

    if os.path.exists(results_path):
        existing_results = json.load(open(results_path, "r"))
    else:
        existing_results = []
    
    existing_results = [x for x in existing_results if x['response'] != 'ERROR']
    all_questions = [x['question'] for x in existing_results]

    for item in queries_and_answers:
        if (item[3]['question'] if len(item) > 3 else item[1]) in all_questions:
            item_idx = all_questions.index(item[3]['question'] if len(item) > 3 else item[1])
            if 'metadata' not in existing_results[item_idx]:
                existing_results[item_idx]['metadata'] = item[3]
                # CHANGED: Use full path
                with open(results_path, "w") as f:
                    json.dump(existing_results, f, indent=2)
            continue
        print("Question [{} / {}]: ".format(len(existing_results), len(queries_and_answers)), item[3]['question'] if len(item) > 3 else item[1])

        response = agent.send_message(item[1], memorizing=False)

        existing_results.append(
            {
                'question': item[3]['question'] if len(item) > 3 else item[1],
                'response': response,
                'answer': item[2],
                'metadata': item[3] if len(item) > 3 else None
            }
        )
        
        with open(results_path, "w") as f:
            json.dump(existing_results, f, indent=2)
        
        agent = AgentWrapper(args.agent_name, load_agent_from=out_dir, model_name=args.model_name, config_path=args.config_path)

def main():
    """Entrypoint orchestrating chunk creation and a single-index run."""
    args = parse_args()
    
    # Load chunks and queries using ConversationCreator like in main.py
    conversation_creator = ConversationCreator(args.dataset, args.num_exp)

    if args.agent_name == 'gpt-long-context':
        with_instructions = False
    else: 
        with_instructions = True

    all_chunks = conversation_creator.chunks(with_instructions=with_instructions)
    all_queries_and_answers = conversation_creator.get_query_and_answer()
    
    # Get the specific chunks and queries for this global_idx
    chunks = all_chunks[args.global_idx]
    queries_and_answers = all_queries_and_answers[args.global_idx]
    
    run_with_chunks_and_questions(args, args.global_idx, chunks, queries_and_answers)


if __name__ == '__main__':
    main()
