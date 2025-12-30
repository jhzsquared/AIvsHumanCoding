import sys
import os
import pandas as pd
import pickle
import logging
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from travail_code import evaluators as et
from travail_code import llm_topic as lt
from travail_code.memory_profile import Monitor
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    '''Loop through an output file to calculate evaluation metrics
    Input:
        json config file with:
            threshold: similarity threshold
            output_files: filepaths with output
            gold_file: filepath for gold codes
            result_path: folder path to save evaluation results
            result_filename: filename of results
            test_reliable: true to conduct reliability of threshold checks
            uid: columns to group results with for analysis
    Return:
        Saves 1 csv and 1 pickle with:
dd
        CSV Statistics - by model/prompt/context/identity/dataproc
        Similarity related
        - perc_coverage: percent of gold codes with a match
        - perc_relevant_codes: percent of created codes that match a gold code
        - avg/std_sim: average and standard similarity score for matches
        - total_output: total number of output codes
        - total_gold: total number of gold codes
        - num_newcodes: number of new codes

        Sentiment related
        - count_switch: count where sentiment of output and gold are different
        - count_best/worst/near_switch: count where sentiment of output and gold are different 
                    # (only for best/worst/near matches for given code regardless of threshold)
        - avg/std_diff: avg/std of difference in sentiment for matches (gold-output, 1 is positive)
        - avg/std_best/worst/near_diff: same as above except for outputs with highest/lowest/near scores corresponding to each gold code
        - count_pos: total number of positive output codes
        - count_pos_switch:  count where sentiment of gold is positive, and it switched 
        - count_nonmatch_pos: number of positive output codes that are new
        - count_pos_gold: total number of positive gold codes

        Pickle  leftover data: 
            dict (keys are a tuple of experiment settings;
                    values are split by parent vs subcode results)
                
        Similarity related
        - count_coverage: dict with count of created codes that match gold code
        - all_matches: list of tuples (out index, gold index) that meet threshold
        - new_codes_i: list of output indices that do not align with any gold code
        - new_codes: the actual new code words
        - best_match: similarity indices for the output code that 
            best aligns with given gold code  (len <= # gold codes)
        - worst_match: same as above, but for least alignment (matches are not included)
        - near_match: same as above, but closest to threshold if a match (len <=# gold codes)
        - best/worst/near_gold_tuple: (gold, output, similarity score)
        - best/near_out_tuple: (output, gold, abs value of similarity diff), regardless of match
        Sentiment related
        - output_sentiment: sentiment of output codes
        - gold_sentiment: sentiment of gold codes
        - sent_diff: difference in sentiment log prob between matching output and gold
    '''
    config_path= sys.argv[1]
    with open(config_path, 'r') as f:
        settings = json.load(f)
    logger.info(settings)
    output_files = settings['output_files']
    threshold = settings['threshold']
    gold_file = settings['gold_file']
    result_path = settings['result_path']
    result_filename = settings['result_filename']
    test_reliable = settings['test_reliable']
    uid = settings['uid']
    # if a path, it needs to be the path of a model that has been manually saved
    sim_model_id = settings['sim_model'] 
    sent_model_id = settings['sent_model']

    # generate result files:
    result_csv_path = os.path.join(result_path, result_filename) # mathematical results
    result_pickle_path = result_csv_path.replace('csv', 'pickle') # for raw matched names/words
    # initial output cleaning 
    output_list = []
    for file in output_files:
        df = pd.read_csv(file, delimiter = "\t")
        output_list.append(df)
    output = pd.concat(output_list)
    output = output[output['prompt_name']!='summary']  # we'll deal with summaries later

    # parse out output:
    exp_df = lt.parse_outputs(output, settings['output_col'])
    exp_df.dropna(subset = 'code', inplace = True)
    gold_codes = pd.read_excel(gold_file, skiprows=[0,1])
    clean_gold = []
    for code, group in gold_codes.groupby('Code'):
        clean_gold.append([code, 0, None, list(group.index)])    
    for subcode, group in gold_codes.groupby('Initial Code'):
        clean_gold.append([subcode, 1, list(set(group['Code'])), list(group.index)])    
    clean_gold_df = pd.DataFrame(clean_gold, columns = ['code', 'sub_yes', 'parent', 'indices'])
    gold_parent_codes = clean_gold_df[clean_gold_df['sub_yes']==0]['code'].reset_index(drop=True)
    gold_subcodes = clean_gold_df[clean_gold_df['sub_yes']==1]['code'].reset_index(drop=True)
    # evaluate per set of interviews
    sim_model = SentenceTransformer(sim_model_id)
    sent_model= pipeline("sentiment-analysis",
                                model = sent_model_id)
    full_results = {}
    parent_sent = None
    parent_embed = None
    subcode_sent = None
    subcode_embed = None
    count =0
    for id, group in exp_df.groupby(uid, dropna=False):
        count+=1
        # if count > 5:
        #     break
        logger.info('evaluating {}'.format(id))
        full_results[id] = {}
        output_codes = pd.Series(list(set(group['code']))) #deduplicate as best able
        # run for parent codes
        evaluator = et.Evaluate_Pipeline(output_codes, gold_parent_codes, threshold,
                                        sim_model, sent_model,
                                        gold_embed = parent_embed,
                                        gold_sentiment = parent_sent)
        results = evaluator()
        full_results[id]['parent_results'] = results
        # update sentiment/embed for gold codes so they're not recalculated
        parent_sent = evaluator.gold_sentiment
        parent_embed = evaluator.gold_embed
        # test subcodes
        full_results[id]['subcode_results'] = evaluator.rerun_subcode(gold_subcodes, subcode_embed, subcode_sent)
        subcode_sent = evaluator.gold_sentiment
        subcode_embed = evaluator.gold_embed
        full_results[id]['output_codes'] = output_codes
        full_results[id]['raw_output_codes'] = group['code']

        if test_reliable: # redo with threshold shifted up and down
            # only do it for subcodes (since least generic text)
            shift = evaluator.similarities.std()
            # TODO: calculate quantile of current threshold
            threshold_high = threshold+shift
            threshold_low = threshold-shift
            full_results[id]['high_results'] = evaluator.rerun_threshold(threshold_high)
            full_results[id]['low_results'] = evaluator.rerun_threshold(threshold_low)
            
    logger.info('Evaluation pipeline complete, saving results out')
    # aggregate results into csv
    csv_keys = ['perc_coverage', 'perc_relevant_codes', 
            'avg_sim', 'std_sim', 
            'count_switch', 'avg_diff', 'std_diff',
            'count_best_switch', 'avg_best_diff', 'std_best_diff',
            'count_worst_switch', 'avg_worst_diff', 'std_near_diff',
            'count_near_switch', 'avg_near_diff', 'std_near_diff',
            'count_pos', 'count_pos_switch', 
            'count_nonmatch_pos', 'count_pos_gold',
            'num_newcodes', 'total_output', 'total_gold']
    # everything else gets pickled as a dict
    metadata_keys = [i for i in results.keys() if i not in csv_keys]
    # compile results
    numer_results = []
    for id in full_results.keys():
        result_id = list(id)
        for k, result_dict in full_results[id].items():
            if k=="output_codes" or k=='raw_output_codes':
                continue
            else:
                result = result_id.copy()
                result.append(k) #parent or subcode or high/low threshold
                result += [result_dict[key] for key in csv_keys]
                numer_results.append(result)
    numer_cols = uid
    numer_cols.append('sub_key')
    numer_cols += csv_keys
    result_df = pd.DataFrame(numer_results, columns = numer_cols)
    logger.info(result_df.shape)
    result_df.to_csv(result_csv_path, index = False)
    
    if settings['save_pickle']:
        # get aggregated stats, and save with metadata
        metadata_file = open(result_pickle_path, 'ab')
        for id in full_results.keys():
            result_id = list(id)
            id_metadata = {id: {}}
            for k, result_dict in full_results[id].items():
                if k=="output_codes" or k=='raw_output_codes':
                    id_metadata[k]=result_dict
                else:
                    id_metadata[k] = {key: 
                                            result_dict[key] for key in metadata_keys}
            pickle.dump({id: id_metadata}, metadata_file)
        pickle.dump({'gold_parent': gold_parent_codes}, metadata_file)
        pickle.dump({'gold_subcode': gold_subcodes}, metadata_file)
        metadata_file.close()
    logger.info('Aggregation of evaluation completed and saved to {}'.format(result_path))

if __name__ == "__main__":
    import time
    monitor = Monitor(600)
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    monitor.stop()