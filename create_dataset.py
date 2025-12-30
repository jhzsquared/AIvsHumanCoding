# create dataset file from pickled text
import torch
import logging
import os
from travail_code.clean_data import InterviewFile, InterviewLine, InterviewProc, InterviewDataset
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



if __name__ == "__main__":
    protocol_file = 
    folder_path = 
    save_path = 

    max_count = 256
    chunk_overlap = 0

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    files = os.listdir(folder_path)
    filepaths = [os.path.join(folder_path, f) for f in files]
    proc_pipe = InterviewProc(filepaths)
    proc_pipe.read_files()

    ## get full text dataset

    data_full = proc_pipe.get_full_text()
    ds_full = InterviewDataset(data_full)
    filepath = f'{save_path}/full_text.pt'
    torch.save(ds_full, filepath)
    logger.info( filepath)
    logger.info(len(data_full))

    # get chunked data

    for method in ['all']: # 'all', 'paired', 'interviewee',
        data_chunks = proc_pipe.get_chunks(method = method, 
                                           max_count = max_count,
                                           chunk_overlap = 0 )
        print(len(data_chunks))
        ds_chunks = InterviewDataset(data_chunks)
        filepath = f'{save_path}/chunk_{method}_count{max_count}_overlap{chunk_overlap}.pt'
        torch.save(ds_chunks,filepath )
        logger.info(filepath)
        logger.info(len(ds_chunks))

    # # get question based

    # full q chunks
    data_ques = proc_pipe.get_question_chunks(protocol_file=protocol_file)
    ds_questions = InterviewDataset(data_ques, qchunk = True)
    filepath = f'{save_path}/questions_full.pt'
    torch.save(ds_questions, filepath)
    logger.info(filepath)
    logger.info(len(ds_questions))
    for max_count in [256, 512, 2048, 4096]:
        data_qchunk = {k : proc_pipe._chunk_text([v], max_count = max_count, 
                                        chunk_overlap = chunk_overlap) for k, v in data_ques.items()}
        ds_qchunks = InterviewDataset(data_qchunk, qchunk = True)
        filepath = f'{save_path}/questions_count{max_count}_overlap{chunk_overlap}.pt'
        torch.save(ds_qchunks, filepath)
        logger.info(filepath)
        logger.info(len(ds_qchunks))