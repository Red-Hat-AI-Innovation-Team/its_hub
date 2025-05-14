import os
import re
import json
import torch
import argparse
import numpy as np
import concurrent.futures
import concurrent
from tqdm import tqdm
from openai import OpenAI
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import warnings
import logging
import uuid


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def extract_answer(text):
    """
    Extract the answer from text that contains a LaTeX boxed expression.
    Properly handles nested braces and extracts the content inside \boxed{}.
    """
    if not text:
        return "ERROR"

    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()

    # Look for \boxed pattern
    if "\\boxed" not in text_lower:
        if "## step 3:" in text_lower:
            return text_lower.split("## step 3:")[-1].strip()
        elif "step 3:" in text_lower:
            return text_lower.split("step 3:")[-1].strip()
        elif "\n\n" in text:
            return text.split("\n\n")[-1].strip()
        else:
            return text

    # Find all instances of \boxed
    boxed_indices = [i for i, _ in enumerate(text_lower) if text_lower[i:i+6] == "\\boxed"][-1:]

    for start_idx in boxed_indices:
        # Find the opening brace after \boxed
        open_brace_idx = text.find("{", start_idx)
        if open_brace_idx == -1:
            continue

        # Track nested braces to find the matching closing brace
        brace_count = 1
        for i in range(open_brace_idx + 1, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                
            if brace_count == 0:
                # Extract content between braces
                content = text[open_brace_idx + 1:i].strip()
                if content:
                    return content.replace("\%", "%")
    
    return "ERROR"


def model_evaluate_answer(judge_model_name, question, real_answer, generated_answer, openai_client=None):
    """Use the model to evaluate if an answer is correct"""
    if len(real_answer.split()) < 3 and ("$" in real_answer or is_number(real_answer)):
        input_text = (
            f"Question: {question}\nCorrect Answer: {real_answer}\nProposed Solution: {generated_answer}\n\n"
            "Given the above correct answer and proposed solution, follow the below steps to evaluate the proposed solution:\n"
        f"[Step 1] First analyze the Proposed Solution using the correct answer as a reference.\nFollow the question regarding the precision/decimal requirement of the answer;\nif not specified, you can ignore decimal differences. You should output Correct if the Proposed Solution is exactly the same as the Correct Answer, or when the question does not specify decimal precision, integer digits are the same. If decimal precision is required, you should output Partial if the Proposed Solution is only decimals difference from the Correct Answer. Otherwise, you should output Incorrect."
            f"[Step 2] Output your final judgement in the following format: \\boxed{{Correct}} or \\boxed{{Incorrect}} or \\boxed{{Partial}}.\n"
        )
    else:
        input_text = (
            f"Question: {question}\nCorrect Answer: {real_answer}\nProposed Solution: {generated_answer}\n\n"
            "Given the above correct answer and proposed solution, follow the below steps to evaluate the proposed solution:\n"
            f"[Step 1] First analyze the Proposed Solution using the correct answer as a reference.\n"
            f"Check if the final conclusion in the Proposed Solution aligns with the correct answer.\n"
            f"You should output Correct if the Proposed Solution reaches the same conclusion.\n"
            f"Output Partial if the Proposed Solution contains contradictory information against the correct answer, but still gives a correct conclusion.\n"
            f"Output Incorrect if the Proposed Solution does not reach the same conclusion.\n"
            f"[Step 2] Output your final judgement in the following format: \\boxed{{Correct}} or \\boxed{{Incorrect}} or \\boxed{{Partial}}.\n"
        )

    response = openai_client.chat.completions.create(
        model=judge_model_name,
        messages=[{"role": "user", "content": input_text}],  # Send one message at a time
        temperature=0.0,
        max_tokens=4000
    )
    response = response.choices[0].message.content
    extracted_text = extract_answer(response).lower()

    if "partial" in extracted_text:
        return 0.5, response
    elif "incorrect" in extracted_text:
        return 0, response
    elif "correct" in extracted_text:
        return 1, response
    else:
        return 0, response
    

def is_correct(judge_model_name, question, extracted_answer, real_answer, openai_client=None):
    """Check if the extracted answer is correct using model evaluation"""
    for attempt in range(3):
        try:
            model_evaluation, text = model_evaluate_answer(
                judge_model_name, question, real_answer, extracted_answer, openai_client
            )
            break
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise
            print(f"Attempt {attempt+1} failed: {e}. Retrying...")
    
    return model_evaluation, text



if __name__ == "__main__":
    judge_openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    judge_model_name = "gpt-4o"
    file_path = "/new_data/shiv/neurips2025/financebench/qwen7b_bon_results.jsonl"
    file_path = "/new_data/shiv/neurips2025/financebench/llama_p8_beam_search_results.jsonl"
    # file_path = "/new_data/shiv/neurips2025/financebench/llama_p8_particle_filtering_results.jsonl"
    file_path = "/new_data/shiv/neurips2025/financebench/llama_p8_self_consistency_results.jsonl"

    # load the test set with the predicted_response column. 
    test_set = load_dataset("json", data_files=file_path)["train"]

    # append the predicted_response to the test set
    responses = [example["response"] for example in test_set]
    # candidate_answers = [example["completions"] for example in test_set]

    gpt4o_judge, gpt4o_judge_text = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Map preserves order of input iterable
        future_results = list(executor.map(
            lambda x: is_correct(
                judge_model_name, 
                x[1]['question'],  # x[1] is the example dict, x[0] is the index
                extract_answer(responses[x[0]]), 
                x[1]['answer'],  # x[1] is the example dict
                judge_openai_client
            ),
            [(i, example) for i, example in enumerate(test_set)]
        ))
        
        for score, judge_text in tqdm(future_results, desc="Processing results"):
            gpt4o_judge.append(score)
            gpt4o_judge_text.append(judge_text)
    
    correct_count = 0
    results = []
    for i, example in enumerate(tqdm(test_set, desc="Evaluating")):
        correct_count += gpt4o_judge[i]
            
        results.append({
            'question_id': i,
            'question': example['question'],
            'correct_answer': example['answer'],
            'model_response': responses[i],
            # 'candidate_answers': candidate_answers[i],
            'extracted_answer': extract_answer(responses[i]),
            'is_correct': gpt4o_judge[i],
            'evaluation_judgement': gpt4o_judge_text[i]
        })
        
        print(f"Question {i+1}/{len(test_set)}: {'✓' if gpt4o_judge[i]==1 else '1/2' if gpt4o_judge[i]==0.5 else '✗'} (Accuracy so far: {correct_count/(i+1):.2%})")
        print(f"Model Answer: {extract_answer(responses[i])}")
        print(f"Correct Answer: {example['answer']}")
    
    # save the results
    # filepath + evaluated_results.jsonl
    with open(file_path.replace(".jsonl", "_evaluated_results.jsonl"), "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    # print the accuracy
    print(f"Accuracy: {correct_count/150:.2%}")
