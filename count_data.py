import pickle
import os
import pandas as pd
import numpy as np
from travail_code.clean_data import InterviewFile, InterviewLine

folder = 'data/cleaned_data/'
datasets = [
    'pg_Wave 1']


def print_filedata(filepath):
    with open(filepath, 'rb') as f:
        i_file = pickle.load(f)
    print(i_file.filename)
    print(i_file.interviewer_names)
    print(i_file.interviewee_names)
    num_words, num_responsewords = count_words(i_file)
    num_turns = len(i_file.clean_lines)
    print('num_lines:', num_turns)
    print('num_words:', num_words)
    return num_words, num_responsewords, num_turns

def count_words(i_file):
    # get a rough count of the number of words per file
    # 24-25k was the max
    df = pd.DataFrame(i_file.full_text)
    text = ' '.join(df[df['useful_text']]['raw_text'])
    num_words = len(text.split())
    interviewee_count = df[df['interviewee']]['raw_text'].apply(lambda x: len(x.split()))
    return num_words, list(interviewee_count)


def check_ifiles():
    # print metadata of interview files 
    # check that values make sense
    num_words = []
    num_turns = []
    num_words_response = []
    for d in datasets:
        files = os.listdir(f'{folder}/{d}')
        for file in files:
            filepath = f'{folder}/{d}/{file}'
            nw, nrw, nt = print_filedata(filepath)
            num_words.append(nw)
            num_words_response.extend(nrw)
            num_turns.append(nt)    
    print(f'Overall stats:  \
    num words: {np.mean(num_words)}, {np.std(num_words)} \
    num words response: {np.mean(num_words_response)}, {np.std(num_words_response)} \
    num turns: {np.mean(num_turns)}, {np.std(num_turns)}') 

if __name__ == "__main__":
    check_ifiles()