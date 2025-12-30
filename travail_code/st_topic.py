# Code for setting up data and setting up sentence embedding models
# deprecated
import nltk
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import vis_documents
from clean_data import InterviewFile
# nltk.download('punkt_tab')

def get_lines_by_speaker(data: InterviewFile):
    '''Extract the text from pre-cleaned InterviewFile object
    Parameters:
        data: InterviewFile object
    Returns:
        dict of each line by speaker type
    '''
    clean_lines = [data.full_text[i] for i in data.clean_lines] 
    interviewee_lines = [line.text for line in clean_lines if line.interviewee]
    interviewer_lines = [line.text for line in clean_lines if line.interviewer]
    credible_lines = [line.text for line in clean_lines if line.credible_messenger]
    unknown_lines = [line.text for line in clean_lines if
                      ((line.interviewer is False) 
                       and (line.interviewee is False) and 
                       (line.credible_messenger is False))]
    return {'interviewee': interviewee_lines, 
            'interviewer': interviewer_lines, 
            'credible_messenger': credible_lines, 
            'unknown': unknown_lines}

def sent_tokenizer(line: str):
    '''Given a line, separate it into sentences using NLTK punkt_tab model
    will need to have run nltk.download('punkt_tab')
    '''
    return nltk.tokenize.sent_tokenize(line, language = "english")

def get_sentences_by_speaker(data: InterviewFile):
    '''Extract list of sentences from pre-cleaned InterviewFile object
    Parameters:
        data: InterviewFile object 
    Returns:
        dict of each line split into sentences by speaker type
    '''
    clean_lines = [data.full_text[i] for i in data.clean_lines] 
    interviewee_lines = [sent for line in clean_lines if line.interviewee 
                         for sent in sent_tokenizer(line.text) ]
    interviewer_lines = [sent for line in clean_lines if line.interviewer 
                         for sent in sent_tokenizer(line.text)]
    credible_lines = [sent for line in clean_lines if line.credible_messenger 
                      for sent in sent_tokenizer(line.text)]
    unknown_lines = [sent for line in clean_lines if 
                     ((line.interviewer is False) and 
                      (line.interviewee is False) and 
                      (line.credible_messenger is False)) 
                        for sent in sent_tokenizer(line.text)]
    return {'interviewee': interviewee_lines, 
            'interviewer': interviewer_lines, 
            'credible_messenger': credible_lines, 
            'unknown': unknown_lines}

def get_chunks(lines:list, max_count = 512, chunk_overlap = 0):
    count_words = lambda x: len(x.split())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = max_count, 
                                                   chunk_overlap = chunk_overlap,
                                                   length_function =  count_words, 
                                               separators = ["\n\n", "\n", ".", "?", "!"])
    punct_trim = re.compile('^(\.|\?|!|\n)')
    chunks = text_splitter.create_documents(lines)
    clean_chunks = [re.sub(punct_trim, '', chunk.page_content).strip() for chunk in chunks]
    return clean_chunks

def get_chunks_by_speaker(data: InterviewFile, **kwargs):
    '''Extract list of sentences from pre-cleaned InterviewFile object
    Parameters:
        data: InterviewFile object 
        **kwargs: optional parameters for `get_chunks`
    Returns:
        dict of each line split into sentences by speaker type
    '''    
    clean_lines = [data.full_text[i] for i in data.clean_lines] 
    interviewee_lines = [line.text for line in clean_lines if line.interviewee]
    interviewer_lines = [line.text for line in clean_lines if line.interviewer]
    credible_lines = [line.text for line in clean_lines if line.credible_messenger]
    unknown_lines = [line.text for line in clean_lines if
                      ((line.interviewer is False) 
                       and (line.interviewee is False) and 
                       (line.credible_messenger is False))]
    full_lines = {'interviewee': interviewee_lines, 
            'interviewer': interviewer_lines, 
            'credible_messenger': credible_lines, 
            'unknown': unknown_lines}
    return {k: get_chunks(v, **kwargs) for k, v in full_lines.items()}
    
def get_paired_chunks(data: InterviewFile, **kwargs):
    '''Extract list of chunks with interviewer appended to following interviewer statement
    Parameters:
        data: InterviewFile object 
        **kwargs: optional parameters for `get_chunks`
    Returns:
        dict of each line split into sentences by speaker type
    '''
    clean_lines = [data.full_text[i] for i in data.clean_lines] 
    combo_lines = {'combo':[],
                'interviewee': [],
                'interviewer': [], 
                'credible_messenger': [], 
                'unknown': []}
    skip_loop = False
    for num, line in enumerate(clean_lines):
        if skip_loop: # line was already saved 
            skip_loop = False
            continue
        if num+1 < len(clean_lines):
            if line.interviewer and clean_lines[num+1].interviewee:
                speaker = 'combo'
                combo_lines['combo'].append(line.text + ' ' + clean_lines[num+1].text)
                skip_loop = True # skip the next loop
        else:
            if line.interviewer:
                speaker = 'interviewer'
            elif line.interviewee:
                speaker = 'interviewer'
            elif line.credible_messenger:
                speaker = 'credible_messenger'
            else:
                speaker = 'unknown'
            combo_lines[speaker].append(line.text)
    combo_chunks = {}
    for speaker, lines in combo_lines.items():
        combo_chunks[speaker] = get_chunks(lines, **kwargs)
    return combo_chunks
    
def senttrans_pipeline(data_list: List[str], topic_model, 
                      sentence_model, 
                      dim_model, 
                      doc_type: List[str] = False):
    '''Run sentence embedding, clustering, and visualization
    Parameters:
        data_list: list of sentences to embed
        topic_model: Sentence transformer model object
        sentence_model: SentenceTransformer model object
        dim_model: UMAP (or other dimension reduction) model object
        doc_type: optional, list of category to set marker shape with
    '''
    embeddings = sentence_model.encode(data_list)
    reduced_embeddings = dim_model.fit_transform(embeddings) # for visualization
    topics, probs = topic_model.fit_transform(data_list,  
                                            embeddings = embeddings) # return assignment of topics per doc
    topic_model.get_topic_info()
    fig = vis_documents.visualize_documents(topic_model, data_list, 
                                            reduced_embeddings = reduced_embeddings,
                                            doc_type = doc_type)
    return topic_model, fig, topics, probs
        