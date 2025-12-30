# utility functions to run LLM experiments
import re
import time
import csv
from tqdm import tqdm
from langchain_core.prompts import  PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
logger = logging.getLogger(__name__)

def make_prompt(model_t, prompt_t):
    '''Combine various model formats with prompt template
    model_t: dict of model specific prefix/suffix
    prompt_t: dict of system, user, and/or assistant
    Alternative: use tokenizer.apply_chat_template.
    But then you can't apply  leading 'assistant' terms
    '''
    base_template = f"{model_t['text']['prefix']}"
    if 'system' in prompt_t.keys():
        base_template = f"{base_template}{model_t['system']['prefix']}{prompt_t['system']}{model_t['system']['suffix']}"
    if 'user' in prompt_t.keys():
        base_template = f"{base_template}{model_t['user']['prefix']}{prompt_t['user']}{model_t['user']['suffix']}"
    if 'assistant' in prompt_t.keys():
        base_template = f"{base_template}{prompt_t['assistant']}"
    return base_template


def langchain_collate(batch):
    # pytorch otherwise by default flattens dicts
    return [item for item in batch]
    
def iterate_prompt(dataloader, llm_pipeline, 
                    model_template, prompt_template, contexts,
                    output_path, metadata):
    '''Go through prompt and save results to output file (iterate through contexts if relevant)
    Parameters:
        dataset: list of dicts of input text values
        llm_pipeline: HuggingFacePipeline object
        model_template: model specific inputs
        prompt_template: prompt specific inputs
        contexts: info on identities and context the llm should adopt
        output_path: where output is saved
        metadata: various metadata variables
    Returns:
        None- saves output to file
    '''
    # select prompt and compile into pipe
    template = make_prompt(model_template, prompt_template)
    full_prompt = PromptTemplate(
        input_variables = ['QUESTION', 'INTERVIEW', 'IDENTITY', 'CONTEXT'],
        template = template)
    llm_chain = full_prompt | llm_pipeline  | StrOutputParser()

    # iterate through dataset 
    for batch_id, batch_text in enumerate(tqdm(dataloader, 
                                               desc = "Inference",
                                               miniters = 5)):
        # if batch_id == 20: # for benchmarking
        #     break
        if batch_id < metadata.get('batch_restart', 0):
             logger.debug('skipping batch id: {}'.format(batch_id))
             continue  # skip until you reach a certain num
        # iterate through context options
        for identity in contexts['IDENTITIES'].keys():
            for context in contexts['CONTEXTS'].keys():
                # get output
                if 'IDENTITY' in full_prompt.input_variables:
                    [input_text.update({'IDENTITY': contexts['IDENTITIES'][identity], 
                                    'CONTEXT': contexts['CONTEXTS'][context]}) 
                                    for input_text in batch_text] # updated batch text
                iterate_start = time.perf_counter()
                batch_output = llm_chain.batch(batch_text)
                iterate_end = time.perf_counter()
                avg_time = (iterate_end-iterate_start)/len(batch_text)
                # save output
                results = []
                for idx, output in enumerate(batch_output): 
                    results.append([
                                batch_text[idx]['filename'], # datasetfilename (or question)
                                batch_id * metadata['batch_size'] + idx, # id of dataset 
                                metadata['model_id'], 
                                metadata['prompt_type'], 
                                output, 
                                identity,
                                context,
                                '{:.4f}'.format(avg_time),
                                metadata['max_new_tokens'],
                                metadata['temp'],
                                metadata['top_k'],
                                metadata['top_p'],
                                metadata['data_proc']])
                with open(output_path, 'a') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerows(results)

def cluster_output(output, llm_pipeline,
                    model_template, prompt_template,
                    contexts,
                    output_path, metadata):
    '''Have LLM cluster outputs into bigger categories
    Parameters:
        output: results of LLM output (not parsed)
        llm_pipeline: HuggingFacePipeline object
        model_template: model specific inputs
        prompt_template: prompt specific inputs
        contexts: info on identities and context the llm should adopt
        output_path: where output is saved
        metadata: various metadata variables
    Returns:
        None- saves output to file
    '''
    # select prompt and compile into pipe
    template = make_prompt(model_template, prompt_template)
    full_prompt = PromptTemplate(
        input_variables = ['OUTPUT', 'IDENTITY', 'CONTEXT'],
        template = template)
    llm_chain = full_prompt | llm_pipeline  | StrOutputParser()
    output_split = parse_outputs(output)
    # iterate through each set of output
    output_keys = ['data_proc', 'model_name', 'prompt_name', 'identity', 'context']
    for keys, group_output in tqdm(output_split.groupby(output_keys, 
                                                        dropna=False), 
                                                    desc="Inference"):
        # use the identity and context from original prompt
        iterate_start = time.perf_counter()
        if 'IDENTITY' in full_prompt.input_variables:
            identity = contexts['IDENTITIES'][keys[3]]
            context = contexts['CONTEXTS'][keys[4]]
        else:
            context = ""
            identity = ""
        input_text = {'OUTPUT': '; '.join(list(group_output['code'].dropna())),
            # 'OUTPUT': '; '.join(list(set(group_output['code'].dropna()))),
                        'IDENTITY': identity, 
                        'CONTEXT': context}
        output = llm_chain.invoke(input_text)
        iterate_end = time.perf_counter()
        # save output
        results = list(keys) + [output, 
                        '{:.4f}'.format(iterate_end-iterate_start),
                        metadata['cluster_prompt'],
                        metadata['max_new_tokens'],
                        metadata['temp'],
                        metadata['top_k'],
                        metadata['top_p']]
        with open(output_path, 'a') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(results)
            
def split_examples(text):
    result = (None, None)
    if type(text) is str:
        split_text = re.split(r'(\*\*:|:\*\*|:|\n)', text) # usually code is in **code** 
        if type(split_text) is list:
            code = split_text[0]
            example = ' '.join(split_text[1:])
            if len(split_text)==1:
                # check if code is preceded by example ("example" (code))
                splitparentext = re.split(r'" \(([^\)]+)\)$', text) # assumes example comes first, code is in ()
                splitquotetext = re.split(r'" - ', text) # assumes "code" - example
                splitboldtext = re.split(r"\*\* ", text) # assumes code** example
                if len(splitparentext)>2:
                    example = ' '.join(splitparentext[:-2])
                    code = splitparentext[-2]
                    result = (code, example)
                elif len(splitquotetext) > 1:
                    example = ' '.join(splitquotetext[1:])
                    code = splitquotetext[0] 
                elif len(splitboldtext) > 1:
                    code = splitboldtext[0]
                    example = ' '.join(splitboldtext[1:])
                else:
                    splitdash = re.split(r' - ', text)
                    code = splitdash[0]
                    example = ' '.join(splitdash[1:])
            code = re.sub('Code [1-9]*', '', code)       
            result = (code, example)
    return result

def parse_outputs(df, output_col="output"):
    '''Parse output csv into list (remove summary outputs beforehand)
    '''
    #remove refusals (even if they provide a code later on)
    df = df.dropna(subset=[output_col]).copy()
    df = df[~df[output_col].str.contains('I cannot provide')].copy()
    # separate out code from examples
    df['split_output'] = df[output_col].apply(lambda x: re.split( 
                                        r"^[(Theme)|(Code)]*\s*\d+[\.|:]\s", x,
                                        flags = re.MULTILINE)[1:]) # split along numbers, skip first one (either blank or will be fluff)
    exp_df = df.explode('split_output').reset_index(drop=True)
    exp_df[['code', 'examples']] = exp_df['split_output'].apply(
        lambda x: split_examples(x)).tolist()
    exp_df['code'] = exp_df['code'].str.strip('**').str.lower()
    exp_df.dropna(subset = "code", inplace = True)
    exp_df = exp_df[exp_df['code']!=""].copy()
    return exp_df