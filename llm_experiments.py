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
from torch.utils.data import DataLoader
from datetime import datetime
import travail_code.llm_topic as lt
from travail_code.clean_data import InterviewDataset # needed or else dataset won't open
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
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        settings = json.load(f)
    model_id = settings['model_id']
    model_source = model_id.split('/')[0]
    model_path = settings['model_path']
    output_path = f"{settings['output_path']}/output_{model_id.split('/')[1]}_{datetime.today().strftime('%Y%m%d%H%M')}.csv"
    input_path = settings['input_path']
    context_file = settings['context_file']
     # initiate output file
    if os.path.exists(output_path)!=True:
        columns = ['filename', # from dataset item (corresponds to interview file or question)
                'dataset_id', # numerical id of dataset item
                'model_name', 
                'prompt_name', 
                'output',  # actual output of llm
                'identity', 
                'context', 
                'time_elapsed',
                'max_new_tokens',
                'temperature',
                'top_k',
                'top_p', 
                'data_proc' # data processing method
                ]
        with open(output_path, 'a') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(columns)
    logger.info('output_path %s',output_path)
    # load model
    if 'gguf' in settings:
        gguf_file = settings['gguf']
    else:
        gguf_file = None
    tokenizer = AutoTokenizer.from_pretrained(f'{model_path}/{model_id}', 
                                              device_map ="auto", 
                                              gguf_file = gguf_file,
                                            local_files_only = True)
    model = AutoModelForCausalLM.from_pretrained(f'{model_path}/{model_id}',
                                                  device_map = "auto",
                                                  gguf_file = gguf_file,
                                                local_files_only = True)
    # check it is on cuda
    if model.device.type == "cuda":
        logging.info('Model is on GPU')
        logging.debug(model.hf_device_map)
    else:
        logging.error('Model is not on GPU')
        logging.debug(model.hf_device_map)
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
    
    for path_dict in settings['paths']:
        logger.info('Running \n {}'.format(path_dict))
        interview_path = path_dict['interview_path']
        prompt_file = path_dict['prompt_file']
        metadata['data_proc'] = path_dict['data_proc']
        metadata['batch_size'] = path_dict['batch_size']
        metadata['batch_restart'] = path_dict.get('batch_restart', 0)
        if metadata['batch_restart']!=0:
             logger.info('Restarting prompt at batch id {}'.format(metadata['batch_restart']))
        # load dataset (could use dataloader but None values):
        dataset = torch.load(interview_path, weights_only = False)
        dataloader = DataLoader(dataset, collate_fn = lt.langchain_collate,
                                batch_size = path_dict['batch_size'], shuffle = False)
   
        # setup  prompts
        with open(f'{input_path}/{prompt_file}' , 'r') as f:
            prompt_templates = json.load(f)

        # iterate through prompts
        for prompt_type in prompt_templates.keys():
            logger.info('Running prompt {}'.format(prompt_type))
            # select prompt and compile into pipe
            metadata['prompt_type'] = prompt_type
            prompt_template = prompt_templates[prompt_type]
            lt.iterate_prompt(dataloader, llm_pipeline, 
                        model_template, prompt_template, contexts,
                        output_path, metadata)
        logger.info('{} interview iteration complete'.format(path_dict['interview_path']))
        
if __name__ == "__main__":
    monitor = Monitor(60)
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    logger.info('Script complete after {:.4f} seconds'.format(end_time-start_time))
    monitor.stop()
