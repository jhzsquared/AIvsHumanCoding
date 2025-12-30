# clean data from raw docx files
from travail_code import clean_data
import pickle
import os

def process_files(filepath, savepath, ph):
        if filepath[-4:]!='docx':
            return
        if 'Participants' in filepath:
            return
        if 'Focus Groups' in filepath:
            focus_group = True
        else:
            focus_group = False
        i_file = clean_data.extract_file(filepath, focus_group, ph)
        savepath = savepath.replace('docx', 'pickle')
        with open(savepath, 'wb') as f:
            pickle.dump(i_file, f)


if __name__ == '__main__':
    folder = 'data/raw_data'
    datasets = [folderpath]
    for d in datasets:
        files = os.listdir(f'{folder}/{d}')
        savefolder = f'data/cleaned_data/{d}'
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        for file in files:
            filepath = f'{folder}/{d}/{file}'
            if os.path.isdir(filepath): # PG data is in nested folder
                savefolder = f'data/cleaned_data/{d}_{file}'
                if not os.path.exists(savefolder):
                    os.makedirs(savefolder)
                for subfile in os.listdir(filepath):
                    process_files(f'{filepath}/{subfile}', f'{savefolder}/{subfile}', ph)
            else:
                process_files(filepath, f'{savefolder}/{file}', ph)
           
