import os
import json
import argparse
import time
import math

import openai
from tqdm import tqdm

from meta_eval_summeval import parse_output

def scoring(scores: list) -> float:
    probs = [math.exp(p) for p in scores]
    total = sum(probs)
    probs = [p / total for p in probs]
    final_score = sum([(i + 1) * p for i, p in enumerate(probs)])
    return final_score

def query(cur_prompt: str, model: str, client: openai.OpenAI) -> float:
    scores_options = ["1", "2", "3", "4", "5"]
    _response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": cur_prompt}],
        temperature=0, # 1 or 2 for GPT-4
        max_tokens=5,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        logprobs=True, # False for GPT-4
        top_logprobs=20, # not relevant for GPT-4
        n=1 # 20 for GPT-4
    )
    # time.sleep(0.5)
    if model.startswith("gpt-4"):
        all_responses = [_response.choices[i].message.content 
                         for i in range(len(_response.choices))]
        all_scores = [parse_output(x) for x in all_responses]
        final_score = sum(all_scores) / len(all_scores)
    else:
        scores = [0, 0, 0, 0, 0]
        for r in _response.choices[0].logprobs.content[0].top_logprobs:
            if r.token in scores_options:
                scores[int(r.token) - 1] = r.logprob
        final_score = scoring(scores)
    return final_score


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--prompt_dir_fp", type=str, default="prompts/freud")
    argparser.add_argument("--save_fp", type=str, default="results/freud/experiment.json")
    argparser.add_argument("--data_fp",
                           type=str,
                           default="data/freud/generated_responses_basic_sample.json")
    argparser.add_argument("--key", type=str, required=True)
    argparser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    args = argparser.parse_args()

    openai.api_key = args.key
    client = openai.OpenAI(api_key=args.key)

    with open(args.data_fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    ct, ignore = 0, 0
    aggregated_scores = {}

    for prompt_fp in tqdm(os.listdir(args.prompt_dir_fp)):
        aspect = prompt_fp.split(".")[0]
        with open(os.path.join(args.prompt_dir_fp, prompt_fp), "r", encoding="utf-8") as f:
            prompt = f.read()

        all_scores = []
        for instance in data:
            dialog = f"""USER: {instance['utt_1']}\nBOT: {instance['utt_2']}\n
                         USER: {instance['utt_3']}\nBOT: {instance['generated_response']}"""
            cur_prompt = prompt.replace("{{Dialog}}", dialog)
            instance["prompt"] = cur_prompt
            while True:
                try:
                    final_score = query(cur_prompt, args.model, client)
                    all_scores.append(final_score)
                    if "scores" in instance:
                        instance["scores"][aspect] = final_score
                    else:
                        instance["scores"] = {aspect: final_score}
                    ct += 1
                    break
                except Exception as e:
                    print(e)
                    if ("limit" in str(e)):
                        time.sleep(2)
                    else:
                        ignore += 1
                        print("ignored", ignore)
                        break
        aggregated_scores[aspect] = sum(all_scores) / len(all_scores)

    print("ignored total", ignore)
    print("Aggregated scores:", aggregated_scores)
    with open(f"{args.save_fp.split('.')[0]}_aggregated.json", "w", encoding="utf-8") as f:
        json.dump(aggregated_scores, f, indent=4)
    with open(args.save_fp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
