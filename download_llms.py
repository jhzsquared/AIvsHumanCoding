from huggingface_hub import snapshot_download
import logging
logging.basicConfig(level=logging.DEBUG)
with open(tokenpath, 'r') as file:
    token = file.read().strip()
model_path = '../models'
model_ids = [
    # "HuggingFaceTB/SmolLM-135M",
    # 'meta-llama/Llama-3.2-1B-Instruct',
    # "meta-llama/Llama-3.1-8B-Instruct",    
    # "google/gemma-3-12b-it",
    # "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    # "mistralai/Ministral-8B-Instruct-2410",
    # "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    # "google/gemma-3-27b-it-qat-q4_0-gguf",
    # "allenai/OLMo-2-1124-13B-Instruct",
    # "allenai/OLMo-2-1124-7B-Instruct",
    "sentence-transformers/all-mpnet-base-v2"
    ]
for model_id in model_ids:
    try:
        snapshot_download(repo_id = model_id, repo_type = "model", 
                          token = "token",
                          local_dir = f'{model_path}/{model_id}')
        print(f'{model_id} download complete')
    except Exception as e:
        print(model_id, "--failed")
        print(e)
        continue


