import pandas as pd
from tqdm import tqdm
from langchain_core.prompts import  PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
import pickle
from datetime import datetime
import travail_code.llm_topic as lt
from travail_code.memory_profile import Monitor
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# from bertopic https://github.com/MaartenGr/BERTopic/blob/master/bertopic/representation/_langchain.py
# and https://github.com/MaartenGr/BERTopic/blob/master/bertopic/representation/_llamacpp.py
DEFAULT_PROMPT = """
This is a list of texts where each collection of texts describe a topic. After each collection of texts, the name of the topic they represent is mentioned as a short-highly-descriptive title
---
Topic:
Sample texts from this topic:
    - {DOCUMENTS}
Keywords: {KEYWORDS}
Topic name:"""

DEFAULT_SYSTEM_PROMPT = "You are an assistant that extracts high-level topics from texts. Only return the topic name."


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
    logger.info(f'Running {settings}')
    output_path = settings['output_path'] # where to save output
    topic_files = settings['topic_files'] # which topic dataframes to read
    model_id = settings['model_id']
    model_path = settings['model_path']
    input_path = settings['input_path']
    model_source = model_id.split('/')[0]
    
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
    try:
        model_template = model_templates[model_source]
    except: 
        logger.error('{} does not exist in model templates. Using default blanks'.format(model_source))
        model_template = model_templates['mistralai']
    

    template = lt.make_prompt(model_template, {'system': DEFAULT_SYSTEM_PROMPT,
                                               'user': DEFAULT_PROMPT})
    full_prompt = PromptTemplate(
        input_variables = ['DOCUMENTS', 'KEYWORDS'],
        template = template)
    
    llm_chain = full_prompt | llm_pipeline  | StrOutputParser()
    for file in topic_files:
        logger.info('Running {}'.format(file))
        with open(file, 'rb') as f:
            topic_info  = pickle.load(f)
        documents = topic_info['Representative_Docs'].apply(lambda x: '\n    - '.join(x)) # new line and a bullet
        docexamples = topic_info['Representative_Examples'].apply(lambda x: '\n    - '.join([f'{k}: {v}' for k, v in x.items()]))
        keywords = topic_info['Representation'].apply(lambda x: ', '.join(x))
        filename = file.split('/')[-1]
        output_file = f'{output_path}/new_{filename}'

        #initiate outputfile:
        if os.path.exists(output_file)!=True:
            columns = ['Topic', 'Count', 'Name', 
                    'Representation', 'Representative_Docs', 'Representative_Examples', 'LLM_name', 'LLM_namewex']
            with open(output_file, 'ab') as f:
                pickle.dump(columns, f)
        for i, doc_kw in enumerate(tqdm(list(zip(documents, docexamples, keywords)))):
            input_text = {'DOCUMENTS': doc_kw[0],
                        'KEYWORDS': doc_kw[2]}
            output = llm_chain.invoke(input_text)
            results = list(topic_info.iloc[i])
            results.append(output.strip())
            # redo with doc and example
            input_text = {'DOCUMENTS': doc_kw[1],
                        'KEYWORDS': doc_kw[2]}
            output2 = llm_chain.invoke(input_text)
            results.append(output2.strip())
            with open(output_file, 'ab') as f:
                pickle.dump(results, f)
    logger.info('Renaming complete')

if __name__ == "__main__":
    monitor = Monitor(60)
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    logger.info('Script complete after {:.4f} seconds'.format(end_time-start_time))
    monitor.stop()