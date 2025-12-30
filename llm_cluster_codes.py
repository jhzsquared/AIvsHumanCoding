# code for LLM experiment pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import sys
import json
import csv
import time
import logging
import os
import pandas as pd
from datetime import datetime
import travail_code.llm_topic as lt
from travail_code.memory_profile import Monitor
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    '''Main function to iterate through interviews and prompts
    Reads in settings from a config file
        output: folder to save output
        model_id: name of HF model
        interview_path: folder to iterate through
        model_kwargs: True if we are manually setting kwargs
            temp, top_k, top_p, max_new_tokens: model specific settings
        data_proc: data processing function used
    Saves LLM output and metadata to a tab delineated csv
    '''
    # define settings
        # define settings
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        settings = json.load(f)
    output_path = settings['output_path'] # where to save cluster output
    output_file = settings['output_file'] # which raw output file to read
    input_path = settings['input_path']
    model_id = settings['model_id']
    model_path = settings['model_path']
    model_source = model_id.split('/')[0]
    filename = output_file.split('/')[-1]
    cluster_file = f"{output_path}/cluster_{filename}"
    context_file = settings['context_file']
    output = pd.read_csv(output_file, delimiter = "\t")
     # initiate output file
    if os.path.exists(cluster_file)!=True:
        columns = ['data_proc',
                'model_name', 
                'prompt_name', 
                'identity', 
                'context', 
                'output_cluster',
                'time_elapsed',
                'cluster_prompt',
                'max_new_tokens',
                'temperature',
                'top_k',
                'top_p'
                ]
        with open(cluster_file, 'a') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(columns)
    logger.info('cluster output file %s',cluster_file)
       
    # load model
    tokenizer = AutoTokenizer.from_pretrained(f'{model_path}/{model_id}', 
                                              device_map = "auto",
                                            local_files_only = True)
    model = AutoModelForCausalLM.from_pretrained(f'{model_path}/{model_id}',
                                                  device_map = "auto",
                                                local_files_only = True)  # check it is on cuda
    if model.device.type == "cuda":
        logging.info('Model is on GPU')
    else:
        logging.error('Model is not on GPU')
        # quit()
    # define model settings
    if settings['model_kwargs']:
        model_kwargs = {
            'temp': settings['temp'],
            'top_k': settings['top_k'],
            'top_p': settings['top_p']
        }
        pipeline_kwargs = {
            'max_new_tokens': settings['max_new_tokens']
        }
    else:
        model_kwargs = {
                'temp': model.generation_config.temperature,
                'top_k': model.generation_config.top_k,
                'top_p': model.generation_config.top_p}
        pipeline_kwargs = {
            'max_new_tokens': model.generation_config.max_new_tokens,
        }
    # load model
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text = False,
        pad_token_id=tokenizer.eos_token_id)
    llm_pipeline = HuggingFacePipeline(pipeline=pipe, 
                                    pipeline_kwargs = pipeline_kwargs,
                                    model_kwargs = model_kwargs)
    # set up templates
    with open(f'{input_path}/prompts/model_templates.json' , 'r') as f:
            model_templates = json.load(f)
    with open(f'{input_path}/{context_file}' , 'r') as f:
            contexts = json.load(f) 
    try:
        model_template = model_templates[model_source]
    except: 
         logger.error('{} does not exist in model templates. Using default blanks'.format(model_source))
         model_template = model_templates['mistralai']
    # initialize initial metadata (will be updated each iteration)
    metadata = pipeline_kwargs
    metadata.update(model_kwargs)
    metadata.update({'model_id': model_id})

    # setup  prompts
    with open(f'{input_path}/prompts/prompts_cluster_codes.json' , 'r') as f:
        prompt_templates = json.load(f)
    total_output = len(output)
    track_cluster = 0
    # iterate through prompts (every prompt clusters with aligned type)
    for prompt_type in prompt_templates.keys():
        logger.info('Running {}'.format(prompt_type))
        if prompt_type =="base_theme":
            sub_output = output[output['prompt_name']=='base_theme'].copy()
        elif prompt_type == "base_t":
            sub_output = output[output['prompt_name'].str.endswith('_t')].copy()
        elif prompt_type == "base_c":
            sub_output = output[output['prompt_name'].str.endswith('_c')].copy()
        if len(sub_output)==0:
            logger.info('Prompt type {} not present'.format(prompt_type))
            continue
        track_cluster += len(sub_output)
        # select prompt and compile into pipe
        metadata['cluster_prompt'] = prompt_type
        prompt_template = prompt_templates[prompt_type]
        lt.cluster_output(sub_output, llm_pipeline, 
                    model_template, prompt_template, contexts,
                    cluster_file, metadata)
        logger.info('prompt iteration complete, clustered {}'.format(len(sub_output)))
    logger.info('{}/{} of output clustered'.format(track_cluster, total_output))        
if __name__ == "__main__":
    monitor = Monitor(600)
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    logger.info('Script complete after {:.4f} seconds'.format(end_time-start_time))
    monitor.stop()