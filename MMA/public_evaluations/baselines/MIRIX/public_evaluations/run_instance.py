#!/usr/bin/env python3

from dotenv import load_dotenv
load_dotenv()

import os
import json
import argparse
from tqdm import tqdm
from agent import AgentWrapper
from conversation_creator import ConversationCreator
import logging
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Run instance with chunks and questions")
    parser.add_argument("--agent_name", type=str, required=True, choices=['gpt-long-context', 'mirix', 'siglip', 'gemini-long-context'])
    parser.add_argument("--dataset", type=str, default="LOCOMO", choices=['LOCOMO', 'ScreenshotVQA'])
    parser.add_argument("--global_idx", type=int, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4.1", help="Model name to use for gpt-long-context agent")
    parser.add_argument("--config_path", type=str, default=None, help="Config file path for mirix agent")
    parser.add_argument("--force_answer_question", action="store_true", default=False)
    parser.add_argument("--num_exp", type=int, default=100, help="Number of experiments")
    return parser.parse_args()


def run_with_chunks_and_questions(
        args,
        global_idx,
        chunks, 
        queries_and_answers):

    out_dir = f"./results/{args.agent_name}_{args.dataset}/"
    if args.agent_name == 'gpt-long-context' or args.agent_name == 'gemini-long-context':
        out_dir = f"./results/{args.agent_name}_{args.dataset}-{args.model_name}/"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_dir = out_dir + f"{global_idx}"

    if os.path.exists(out_dir):
        
        agent = AgentWrapper(args.agent_name, load_agent_from=out_dir, model_name=args.model_name, config_path=args.config_path)
    
    else:

        if args.agent_name == 'mirix':
            if os.path.exists(os.path.expanduser(f"~/.mirix/sqlite.db")):
                # need to delete the existing db
                os.system(f"rm -rf ~/.mirix/sqlite.db*")

        agent = AgentWrapper(args.agent_name, model_name=args.model_name, config_path=args.config_path)

    if os.path.exists(f"{out_dir}/current_step.txt"):
        with open(f"{out_dir}/current_step.txt", "rb") as f:
            current_step = int(f.read().decode())
    else:
        current_step = -1

    if os.path.exists(f"{out_dir}/chunks.json"):
        with open(f"{out_dir}/chunks.json", "r") as f:
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

        if args.agent_name == 'mirix':
            agent.save_agent(out_dir)

            with open(f"{out_dir}/current_step.txt", "wb") as f:
                f.write(str(idx).encode())

            with open(f"{out_dir}/chunks.json", "w") as f:
                json.dump(existing_chunks, f, indent=2)


    agent.save_agent(out_dir)

    agent.prepare_before_asking_questions()

    if os.path.exists(f"{out_dir}/results.json"):
        existing_results = json.load(open(f"{out_dir}/results.json", "r"))
    else:
        existing_results = []
    
    existing_results = [x for x in existing_results if x['response'] != 'ERROR']

    all_questions = [x['question'] for x in existing_results]

    for item in queries_and_answers:

        if (item[3]['question'] if len(item) > 3 else item[1]) in all_questions:
            item_idx = all_questions.index(item[3]['question'] if len(item) > 3 else item[1])
            if 'metadata' not in existing_results[item_idx]:
                existing_results[item_idx]['metadata'] = item[3]
                with open(f"{out_dir}/results.json", "w") as f:
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

        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(existing_results, f, indent=2)
        
        agent = AgentWrapper(args.agent_name, load_agent_from=out_dir, model_name=args.model_name, config_path=args.config_path)

def main():
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
