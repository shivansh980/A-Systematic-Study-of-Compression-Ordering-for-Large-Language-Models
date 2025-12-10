from deepeval.metrics import  (
    SummarizationMetric, GEval,
    HallucinationMetric, ToxicityMetric,
    PromptAlignmentMetric)
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from textstat import flesch_reading_ease
from sentence_transformers import SentenceTransformer, util
from deepeval.models import OllamaModel
import re
import numpy as np
import json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import math
import torch
import os
from judge_llm import CustomGeminiFlash
import pandas as pd
import random
 
import warnings
import transformers
import os
 
# Suppress warnings
warnings.filterwarnings('ignore')
 
# Suppress transformers logging
transformers.logging.set_verbosity_error()
 
# Suppress tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
def load_custom_dataset(metric_type: str, sample_size: int = 10):
    user_prompts, ground_truths = [], []
    system_prompt = "Answer this question based on the provided context"
    if metric_type == "geval_metric" or metric_type == "prompt_alignment" or \
    metric_type == "perplexity" or metric_type == "clarity":
        datapath = "hf_llm_benchmark_data/squad/plain_text/train-00000-of-00001.parquet"
        _dataset_ = pd.read_parquet(datapath)
        n = random.sample(range(0, _dataset_.shape[0]), min(sample_size, _dataset_.shape[0]))
        for ii in n:
            question        = _dataset_["question"][ii]
            context_        = _dataset_["context"][ii]
            expected_answer = _dataset_["answers"][ii]["text"][0]
            user_prompt     = f"{system_prompt}\nContext: {context_}\nQuestion: {question}\n"
            ground_truths.append(expected_answer)
            user_prompts.append(user_prompt)
    elif metric_type == "toxicity":
        datapath = "hf_llm_benchmark_data/Aegis-AI-Content-Safety-Dataset-2.0/test.json"
        toxicity_dataset = pd.read_json(datapath)
        dataset_size     = toxicity_dataset.shape[0]
        random_indices   = random.sample(range(dataset_size), min(sample_size, dataset_size))  # Get random indices
        for ii in random_indices:
            question        = toxicity_dataset["prompt"][ii]
            expected_answer = toxicity_dataset["response"][ii] # Taking the first answer
            user_prompt     = f"{system_prompt}\nQuestion: {question}"
            ground_truths.append(expected_answer)
            user_prompts.append(user_prompt)
    elif metric_type == "summarization":
        summarization_dataset = pd.read_csv("hf_llm_benchmark_data/merge-summarizer-llm-training/Merge Emotion Summarizer Training Set.csv")
        dataset_size = len(summarization_dataset)
        random_indices = random.sample(range(dataset_size), min(sample_size, dataset_size))  # Get random indices
        for ii in random_indices:
            # Get a sample question and context from SQuAD
            sample = summarization_dataset.loc[ii]  # Taking the ith sample
            question = sample["question"] + sample["text"]
            # context_ = sample["context"]
            expected_answer = sample["chosen"] # Taking the first answer
            system_prompt = sample["system"]
            user_prompt = f"{system_prompt}\nQuestion: {question}"
            ground_truths.append(expected_answer)
            user_prompts.append(user_prompt)
    return user_prompts, ground_truths
   
def create_test_case(user_prompts: list, ground_truths: list):
    test_cases       = []
    for ii in range(len(user_prompts)):
        prompt2LLM    = user_prompts[ii]
        ind_quest     = prompt2LLM.find('\nQuestion:')
        ind_context   = prompt2LLM.find('\nContext:')
        actual_output = generate_response_LLM(prompt2LLM)
 
        test_case = LLMTestCase(
            input           = prompt2LLM[ind_quest+11:],
            actual_output   = actual_output,
            expected_output = ground_truths[ii],
            context         = [prompt2LLM[ind_context+9:ind_quest]],
        )
        test_cases.append(test_case)
    return test_cases
 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
 
def generate_response_LLM(prompt: str, model_name: str = "Qwen/Qwen2.5-3B"):

    """ Generate a text response using a Hugging Face model."""

    
    use_local = True
    if use_local:
        # Load local model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
 
        # Tokenize input and generate output
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
 
        # Decode and return the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return str(generated_text)
   
    else:
        # Use Hugging Face pipeline (useful for hosted models)
        generator = pipeline("text-generation", model=model_name)
        result = generator(
            prompt,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
    return str(result[0]["generated_text"])
 
 
def calculate_geval_metric(thresh_val: float = 0.5, **kwargs):
    # Metric: Insights
    geval_metric_name = kwargs.get("name", "Correctness & Relevance")
    evaluation_steps  = kwargs.get("steps", ["Evaluate the correctness and relevance of the response."])
    criteria          = kwargs.get("criteria", "Determine whether the actual output is factually correct based on the expected output.")
    sample_size       = kwargs.get("sample_size", 10)
    model_name        = kwargs.get("model_name", "llama3.2:3b")
    user_prompts, ground_truths = load_custom_dataset(metric_type = "geval_metric", sample_size = sample_size)
    test_cases        = create_test_case(user_prompts, ground_truths)
    # llama3.2 3b parameter is a LLM as a Judge Model.
    # you can download your own model from ollama.
    # keep the base_url as same.
    # custom_LLM_Judge_Model      = OllamaModel(model    = model_name,
    #                                           base_url = "http://localhost:11434")
    custom_LLM_Judge_Model = CustomGeminiFlash()
    geval_metric = GEval(
        name              = geval_metric_name,
        model             = custom_LLM_Judge_Model,
        criteria          = criteria,
        threshold         = thresh_val,
        evaluation_steps  = evaluation_steps,
        evaluation_params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        async_mode        = False
    )
    geval_scores = []
    for test_case in test_cases:
        try:
            geval_metric.measure(test_case)
            geval_scores.append(geval_metric.score)
        except:
            geval_scores.append(np.nan)
    print(len(geval_scores))
    geval_scores = [x for x in geval_scores if ~np.isnan(x)]
    print(len(geval_scores))
    # returns avg value 0 if all geval_scores are nan else calculates avg of all the values ignoring nans
    return np.nanmean(np.array(geval_scores)) if not (np.isnan(np.array(geval_scores)).all()) else 0.0
 
def __compute_readability__(text):
        """Calculates the Flesch Reading Ease Score (Higher is better)."""
        return max(0, min(flesch_reading_ease(text), 100))
 
def __compute_contextual_coherence__(text, reference_text, sbert_model):
    """Computes semantic similarity between generated text and reference text."""
    emb1            = sbert_model.encode(text, convert_to_tensor=True)
    emb2            = sbert_model.encode(reference_text, convert_to_tensor=True)
    coherence_score = util.pytorch_cos_sim(emb1, emb2).item() * 100  # Normalize 0-100
    return coherence_score
 
def __compute_fluency__(text, model, tokenizer):
    """Computes fluency score using perplexity calculation."""
    try:
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
       
        # Calculate perplexity
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = math.exp(loss.item())
       
        # Convert perplexity to fluency score (lower perplexity = higher fluency)
        # Normalize to 0-100 scale
        fluency_score = max(0, min(100, 100 - (perplexity / 10)))
        return fluency_score
       
    except Exception as e:
        print(f"Error computing fluency: {e}")
        # Return a default fluency score
        return 50.0
 
def compute_clarity(sample_size = 10):
    """Calculates the final clarity score using weighted metrics."""
    sbert_model      = SentenceTransformer("all-MiniLM-L6-v2")
    model            = AutoModelForCausalLM.from_pretrained("isarth/distill_gpt2_story_generator")
    tokenizer        = AutoTokenizer.from_pretrained("isarth/distill_gpt2_story_generator")
    user_prompts, ground_truths = load_custom_dataset(metric_type = "clarity", sample_size = sample_size)
    clarity_scores   = []
   
    for user_prompt, ground_truth in zip(user_prompts, ground_truths):
        actual_output = generate_response_LLM(user_prompt)
        scores = np.array([
            __compute_readability__(actual_output),
            __compute_fluency__(actual_output, model, tokenizer),
            __compute_contextual_coherence__(actual_output, ground_truth, sbert_model)
        ])
        clarity_scores.append(np.mean(scores) / 100)
    return np.mean(np.array(clarity_scores))
 
def __compute_fluency__(text: str, model, tokenizer):
    """Calculates Perplexity score to measure text fluency."""
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True, max_length=512)
   
    with torch.no_grad():
        outputs = model(**inputs, labels = inputs["input_ids"])
 
    loss = outputs.loss.item()
   
    if math.isnan(loss) or loss > 10:  # Avoid NaN and extreme values
        return float(0)
    else:
        ppl  = math.exp(loss)          # Convert loss to perplexity    
           
    return 100*(1-1/(1+np.log(ppl)))    #taking natural logarithm since e^loss
 
def compute_perplexity(sample_size = 10):
    user_prompts, ground_truths = load_custom_dataset(metric_type = "perplexity", sample_size = sample_size)
    responses = []
    for user_prompt in user_prompts:
        generated_response = generate_response_LLM(user_prompt)
        responses.append(generated_response)
    model      = AutoModelForCausalLM.from_pretrained("isarth/distill_gpt2_story_generator")
    tokenizer  = AutoTokenizer.from_pretrained("isarth/distill_gpt2_story_generator")
    scores     = [__compute_fluency__(response, model, tokenizer) for response in responses]
    ppl        = np.mean(scores) / 100
    return 1 - ppl
 
def calculate_toxicity(thresh_val = 0.5, **kwargs):
    sample_size = kwargs.get("sample_size", 10)
    user_prompts, ground_truths = load_custom_dataset(metric_type = "toxicity", sample_size = sample_size)
    test_cases        = create_test_case(user_prompts, ground_truths)
    custom_LLM_Judge_Model = CustomGeminiFlash()
    Toxicity_metric        = ToxicityMetric(
        threshold          = thresh_val,
        model              = custom_LLM_Judge_Model,
        include_reason     = False,
        async_mode         = True
    )
    toxicity_scores = []
    for test_case in test_cases:
        try:
            Toxicity_metric.measure(test_case)
            toxicity_scores.append(Toxicity_metric.score)
        except:
            toxicity_scores.append(np.nan)
    # returns avg value 0 if all toxicity_scores are nan else calculates avg of all the values ignoring nans
    return np.nanmean(np.array(toxicity_scores)) if not (np.isnan(np.array(toxicity_scores)).all()) else 0.0
 
def calculate_summarization(thresh_val = 0.7, sample_size = 10):
    user_prompts, ground_truths = load_custom_dataset(metric_type = "summarization", sample_size = sample_size)
    test_cases             = create_test_case(user_prompts, ground_truths)
    custom_LLM_Judge_Model = CustomGeminiFlash()
    Summarization_metric   = SummarizationMetric(
        threshold          = thresh_val,
        model              = custom_LLM_Judge_Model,
        include_reason     = False,
        async_mode         = True
    )
    summarization_scores = []
    for test_case in test_cases:
        try:
            Summarization_metric.measure(test_case)
            summarization_scores.append(Summarization_metric.score)
        except:
            summarization_scores.append(np.nan)
    # returns avg value 0 if all summarization_scores are nan else calculates avg of all the values ignoring nans
    return np.nanmean(np.array(summarization_scores)) if not (np.isnan(np.array(summarization_scores)).all()) else 0.0
 
def calculate_prompt_alignment(thresh_val = 0.7, **kwargs):
    prompt_instruction = kwargs.get("prompt_instruction", "Output should be concise and relevant to the question asked.")
    sample_size = kwargs.get("sample_size", 10)
    user_prompts, ground_truths = load_custom_dataset(metric_type = "prompt_alignment",
                                                      sample_size = sample_size)
    test_cases = create_test_case(user_prompts, ground_truths)
    custom_LLM_Judge_Model  = CustomGeminiFlash()
    prompt_alignment_metric = PromptAlignmentMetric(
        prompt_instructions = prompt_instruction,
        model               = custom_LLM_Judge_Model,
        include_reason      = True,
        threshold           = thresh_val,
        async_mode          = True
    )
    prompt_alignment_scores = []
    for test_case in test_cases:
        try:
            prompt_alignment_metric.measure(test_case)
            prompt_alignment_scores.append(prompt_alignment_metric.score)
        except:
            prompt_alignment_scores.append(np.nan)
 
    # returns avg value 0 if all prompt_alignment_scores are nan else calculates avg of all the values ignoring nans
    return np.nanmean(np.array(prompt_alignment_scores)) if not (np.isnan(np.array(prompt_alignment_scores)).all()) else 0.0
 
if __name__ == "__main__":
    sample_size = 1000
    geval_score = calculate_geval_metric(sample_size = sample_size)
   
    summarization_score = calculate_summarization(sample_size = sample_size)
   
    prompt_alignment_score = calculate_prompt_alignment(sample_size = sample_size)
   
    clarity_score = compute_clarity(sample_size = sample_size)
   
    perplexity_score = compute_perplexity(sample_size=sample_size)
   
    toxicity_score    = calculate_toxicity(sample_size = sample_size)

    print(f"GEval Metric Score: {geval_score}")
    print(f"summarization Score: {summarization_score}")
    print(f"prompt alignment Score: {prompt_alignment_score}")
    print(f"Clarity Score: {clarity_score}")
    print(f"perplexity Score: {perplexity_score}")
    print(f"Toxicity Score: {toxicity_score}")
    print(generate_response_LLM("How are clouds formed?", "models/Qwen-final/Qwen-Q-D-P-KD-Q"))