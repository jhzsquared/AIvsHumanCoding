# functions to evaluate output of a given set of results
import pandas as pd
import re
import logging
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
from collections import Counter
logger = logging.getLogger(__name__)
class Evaluate_Pipeline():
    '''Automated pipeline to evaluate codes using sentence transformers
    '''
    def __init__(self, output_codes, gold_codes, threshold, 
                 sim_model, sent_model = None,
                 gold_embed = None, gold_sentiment = None,
                ):
         '''
         Args:
            output_codes: list of code strings
            gold_codes: list of gold code strings,
            threshold: similarity threshold
            gold_embed: embedding for gold codes (optional)
            gold_sentiment: sentiment for gold codes (optional)
            sim_model: model to calculate cosine similarity 
            sent_model: model to calculate sentiment 
                (do not recommend changing this given potential variation in output format)
         '''
         self.output_codes = output_codes
         self.gold_codes = gold_codes
         self.threshold = threshold
         self.sim_model = sim_model
         self.sent_model = sent_model
         # values that could be provided as input or generated later
         self.gold_embed = gold_embed
         self.gold_sentiment = gold_sentiment
         # values to be generated later
         self.output_embed = None
         self.similarities = None
         self.output_sentiment = None

    def __call__(self):
        '''Run through all similarity and sentiment evaluation
        '''    
        logger.debug('calculating similarity')
        self.similarities = self.calc_similarity()
        logger.debug('evaluating similarity')
        results = self.evaluate_similarity(self.similarities)
        logger.debug('evaluating sentiment')
        if self.sent_model:
            sent_results = self.evaluate_sentiment(results)
            results.update(sent_results)
            for key in ['best', 'worst', 'near']:
                results[f'{key}_gold_tuple'] = self.get_match_triplets(results[f'{key}_match'])
        results[f'best_out_tuple'] = self.get_similarity_triplets(results[f'max_similarity_score'])
        results[f'near_out_tuple'] = self.get_similarity_triplets(results[f'near_similarity_score'])
 
        return results


    def rerun_threshold(self, new_threshold):
        '''Rerun with different threshold values
        (used saved similarity values + sentiment embeddings)
        '''
        self.threshold = new_threshold
        results = self.evaluate_similarity(self.similarities)
        sent_results = self.evaluate_sentiment(results)
        results.update(sent_results)
        for key in ['best', 'worst', 'near']:
            results[f'{key}_gold_tuple'] = self.get_match_triplets(results[f'{key}_match'])
        results[f'best_out_tuple'] = self.get_similarity_triplets(results[f'max_similarity_score'])
        results[f'near_out_tuple'] = self.get_similarity_triplets(results[f'near_similarity_score'])
        return results
    
    def rerun_subcode(self, gold_codes, gold_embed = None, gold_sent = None):
        '''Rerun with new set of gold codes (subcode)
        (used saved output embeddings)
        if gold embeddings and/or sentiment is not provided it will recalculate them
        '''
        self.gold_codes = gold_codes
        #reset values
        self.gold_embed = gold_embed
        self.gold_sentiment = gold_sent
        self.similarities = self.calc_similarity()
        results = self.evaluate_similarity(self.similarities)
        if self.sent_model:
            sent_results = self.evaluate_sentiment(results)
            results.update(sent_results)
            for key in ['best', 'worst', 'near']:
                results[f'{key}_gold_tuple'] = self.get_match_triplets(results[f'{key}_match'])
        results[f'best_out_tuple'] = self.get_similarity_triplets(results[f'max_similarity_score'])
        results[f'near_out_tuple'] = self.get_similarity_triplets(results[f'near_similarity_score'])
        return results

    def calc_similarity(self):
        '''Calculate cosine similarity
        Parameters:
            self:
                output_codes: list of output code strings
                gold_codes: list of gold code strings
            model_id: sentence transformer model id
        Returns
            2D tensor of similarity metric
        '''
        if self.output_embed is None:
            self.output_embed = self.sim_model.encode(list(self.output_codes))
        if self.gold_embed is None:
            self.gold_embed= self.sim_model.encode(list(self.gold_codes))
        similarities = self.sim_model.similarity(self.output_embed, self.gold_embed)
        return similarities

    def evaluate_similarity(self, similarities):
        '''Evaluate similarity by coverage and relevance of codes
        Parameters:
            self:
                thresholds: float (percent similarity min)
            similarities: tensor (output codes, gold codes)
        Returns:
            results: dict 
                max_similarity_score: values and indices of highest similarity score of each output code (index of the gold code)
                perc_coverage: percent of gold codes with a match
                perc_relevant_codes: percent of created codes that match a gold code
                avg/std_sim: average and standard similarity score for matches
                count_coverage: dict with count of created codes that match gold code
                all_matches: list of tuples (out index, gold index) that meet threshold
                new_codes_i: list of output indices that do not align with any gold code
                new_codes: the actual new code words
                best_match: similarity indices for the output code that 
                    best aligns with given gold code  (len <= # gold codes)
                worst_match: same as above, but for least alignment (matches are not included)
                near_match: same as above, but closest to threshold if a match (len <=# gold codes)
                total_output: total number of output codes
                total_gold: total number of gold codes
        '''
        matches = torch.where(similarities>self.threshold)
        gold_matches = matches[1].tolist()
        # count of how many times output matched each gold code (key)
        count_coverage = Counter(gold_matches) 
        # get codes that don't match any gold codes
        new_codes_i = torch.where(torch.all(similarities <= self.threshold, dim =1))[0].tolist()
        new_codes = self.output_codes[new_codes_i]
        # get best match for each gold code
        best = similarities.max(dim=0)
        best_gold = torch.where(best.values > self.threshold)[0]
        best_match = (best.indices[best_gold], best_gold)
        # get worst non match for each gold code
        worst = similarities.min(dim=0)
        worst_gold = torch.where(worst.values <= self.threshold)[0]
        worst_match = (worst.indices[worst_gold], worst_gold)
        # get matches closest to threshold (only if a match)
        match_near = torch.where(similarities >= self.threshold,  similarities,0)
        near_diff = torch.abs(match_near-self.threshold)
        near = near_diff.min(dim=0)
        near_gold = torch.where(near.values!= self.threshold)[0] # only if a match 
        near_match = (near.indices[near_gold], near_gold)

        results = {'max_similarity_score': similarities.max(dim=1), # best score for each new code
                'near_similarity_score': torch.abs(similarities-self.threshold).min(dim=1),
                'perc_coverage': len(count_coverage)/similarities.size()[1],
                'perc_relevant_codes': 1-(len(new_codes)/similarities.size()[0]),
                'avg_sim': torch.mean(similarities[matches]).item() ,
                'std_sim': torch.std(similarities[matches]).item(),
                'count_coverage': count_coverage,
                'all_matches': matches,
                'new_codes_i': new_codes_i,
                'new_codes': new_codes,
                'best_match': best_match,
                'worst_match': worst_match,
                'near_match': near_match,
                'num_newcodes': len(new_codes),
                'total_output': similarities.size()[0],
                'total_gold': similarities.size()[1]
            }
        return results
    
    def calc_sentiment(self, data):
        # calculate sentiment of given data (list of text)s
        sentiment = []
        sentiment = self.sent_model(list(data))
        df = pd.DataFrame(sentiment, columns = ['label', 'score'])
        df['codes'] =  data
        # normalize scores (1 is positive, 0 is negative)
        df['pos_score'] = df.apply(lambda x: 1-x['score'] if x['label']=="NEGATIVE" else x['score'], axis = 1)
        return df
       
    def evaluate_sentiment(self, sim_results):
        ''' Get sentiment of output codes and compare difference
        Parameters:
            self:
                output_codes: list of codes
                gold_codes: list of gold codes
            sim_results:
                all_matches: from similarity evaluation- list of tensor tuples corresponding to indices of output and the gold code they match
                best_match: same as above, but only the tuples for the output codes with the highest scores corresponding to each gold code
                worst_match: same as above but only the tuples for the output codes with the lowest scores corresponding to each gold code
                near_match: same as above but only the tuples for the output codes with the scores closest to the threshold to each gold code
                new_codes: new code indices
            update: True if update class results
        Returns:
            results: dict
                count_switch: count where sentiment of output and gold are different
                count_best/worst/near_switch: count where sentiment of output and gold are different 
                            # (only for best/worst/near matches for given code regardless of threshold)
                count_pos_switch:  count where sentiment of gold is positive, and it switched negative (for all matches)
                output_sentiment: sentiment of output codes
                gold_sentiment: sentiment of gold codes
                sent_diff: difference in sentiment log prob between matching output and gold
                avg/std_diff: avg/std of difference in sentiment for matches
                count_pos: total number of positive output codes
                count_nonmatch_pos: number of positive output codes that are new
                count_pos_gold: total number of gold positive output codes
        # can report avg, std, and plot histogram of sent_diff

        '''
        
        if self.output_sentiment is None:
            out_df = self.calc_sentiment(self.output_codes)
            self.output_sentiment = out_df
        else:
            out_df = self.output_sentiment
        if self.gold_sentiment is None:
            gold_df = self.calc_sentiment(self.gold_codes)
            self.gold_sentiment = gold_df
        else:
            gold_df = self.gold_sentiment

        all_matches = sim_results['all_matches']
        best_match = sim_results['best_match']
        worst_match = sim_results['worst_match']
        near_match = sim_results['near_match']
        # calculate differences from all the matches
        out_match_index = all_matches[0]
        gold_match_index = all_matches[1]
        out_matches = out_df.loc[out_match_index]
        out_nonmatches = out_df.loc[sim_results['new_codes_i']]
        gold_matches = gold_df.loc[gold_match_index]
        out_matches.reset_index(inplace = True)
        gold_matches.reset_index(inplace = True)
        matches_df = out_matches.merge(gold_matches, suffixes = ('_out', '_gold'), left_index = True, right_index = True )
        sent_diff = matches_df['pos_score_gold'] - matches_df['pos_score_out']

        #get actual count of different sentiment labels
        mismatch =  matches_df['label_gold'] != matches_df['label_out']
        count_switch = sum(mismatch)
        # get count where positive og switched
        mismatch_df = matches_df[mismatch]
        count_pos_switch = len(mismatch_df[mismatch_df['label_gold']=='POSITIVE'])

        # track differences from best/worst/near matches (sanity check, bc shouldn't be too different)
        results = {}
        for subset_match, name in zip([best_match, worst_match, near_match], ['best', 'worst', 'near']):
            results[f'count_{name}_switch'] = sum(out_df.loc[subset_match[0]].reset_index()['label'] != 
                                                  gold_df.loc[subset_match[1]].reset_index()['label'] )
            results[f'{name}_diff'] = out_df.loc[subset_match[0]].reset_index()['pos_score'] - \
                gold_df.loc[subset_match[1]].reset_index()['pos_score'] 
            results[f'avg_{name}_diff'] = results[f'{name}_diff'].mean()
            results[f'std_{name}_diff'] = results[f'{name}_diff'].std()

        results.update( {'count_switch': count_switch, # count where sentiment of output and gold are different 
                    'count_pos_switch': count_pos_switch, # count where sentiment of gold is positive, and it switched negative (for all matches)
                'output_sentiment': out_df, # sentiment of output codes
                'gold_sentiment': gold_df, # sentiment of gold codes
                'sent_diff': sent_diff, # difference in sentiment log prob between matching output and gold
                'avg_diff': sent_diff.mean(),
                'std_diff': sent_diff.std(),
                'count_pos': len(out_df[out_df['label']=='POSITIVE']),
                'count_nonmatch_pos': len(out_nonmatches[out_nonmatches['label']=='POSITIVE']),
                'count_pos_gold': len(gold_df[gold_df['label']=='POSITIVE'])
            }) 
        return results

    def get_match_triplets(self, match_indices):
        '''Gets list of best/worst/near match indices of each gold code for a sanity check
        Parameters:
            self
            match_indices: a tuple of output index, gold index
        Returns:
            list of tuple (gold, output, similarity score)
        '''
        return list(zip(self.gold_sentiment.loc[match_indices[1]]['codes'],
                 self.output_sentiment.loc[match_indices[0]]['codes'], 
                 self.similarities[match_indices]))

    def get_similarity_triplets(self, index_value):
        '''Gets list of best/worst/near similar indices of each output code for a sanity check
        Parameters:
            self
            match_indices: 
        Returns:
            list of tuple (output, gold, similarity score)
        '''
        return list(zip(self.output_codes, 
                 self.gold_codes.loc[index_value.indices], 
                 index_value.values))