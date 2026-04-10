# "Can LLMs Understand the Impact of Trauma? 
Last updated: 10APR2026

Repo for work accepted to 2026 ACL Findings: "Can LLMs Understand the Impact of Trauma? Costs and Benefits of LLMs
Coding the Interviews of Firearm Violence Survivors" 

## Repo setup
- Python version 3.10.12
- highly recommend using a virtual environment 
    - python -m venv .name_of_venv (use something that is meaningful to the project)
    - `source .name_of_venv/bin/activate` for linux to activate the venv before installing packages
- `pip install -r requirements.txt` (install packages)
    - may need to first run `apt-get install python3-dev` for hdbscan to install


## Research question:
How well can topic modeling capture key and novel themes (i.e. codes) in gun violence interviews? 

## Method:
### Clean Data  
Process word docs into python/model friendly structure and chunk as appropriate per the desired context
- `clean_raw.py`: pre-processes the raw word doc into a pickle file of `InterviewFile` and `InterviewLine` objects (object defined in  `travail/clean_data.py`)
-  `create_dataset.py`: breaks up the pre-processed data and restructures into torch dataset objects (defined as `InterviewDataset` in `travail/clean_data.py`) 
    - uses `InterviewProc()` function to chunk data appropriately
        - pairing respondant's statement to previous interviewer


### Modeling
1. Prompt with Generative Instruct models( Open source decoder only large language models (e.g. llama, gemma, or mistral)) using `llm_experiments.py` (calls `travail_code/llm_topic.py`)
    - limited by resource constraints (24GB of VRAM on BSWIFT, Zaratan compute limits)
    - Test various prompting strategies, different context, and identities
2. Aggregate results of prompt or embedding based codes
    - using generative LLMs: `llm_cluster_codes.py`
    - cluster with BERTopic: use `notebooks/cluster_output.py`

### Evaluation:
Evaluate on topic/code accuracy + relevancy

1. `evaluate_llm.py`: calls `travail_code/evaluators.py` (basic metrics defined there)
    - looks at cosine similarity and sentiment difference

## Desired End state:
1.	A better understanding of the interviews and interventions in gun violence.
2.	A better understanding of if/how AI can help with coding.


## Limitations:
-	Some of the interviews were auto transcribed. AI -transcriptions may propagate their own biases against AAVE (e.g. systematic inaccuracies). However, I believe most of these transcriptions were manually checked.
-	Coding is inherently a subjective task. While the transcripts have manually curated codes, there is no true “correct” measurement. However, we will have access to subject matter experts who can validate the relevance of the model generated codes. 
