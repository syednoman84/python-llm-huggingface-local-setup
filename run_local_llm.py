# run_local_llm.py

import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

import time
from datetime import datetime
# -----------------------------
# 1. Download GGUF model from Hugging Face
# -----------------------------
MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q6_K.gguf"

print("Downloading model from Hugging Face...")
model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILENAME,
    local_dir="models",
    local_dir_use_symlinks=False
)
print(f"Model downloaded to: {model_path}")

# -----------------------------
# 2. Load the model with llama-cpp
# -----------------------------
print("Loading model into Llama...")
llm = Llama(
    model_path=model_path,
    n_ctx=768,
    n_batch=128,
    n_gpu_layers=0  # Set to 0 for CPU; Metal GPU is auto-used on Mac
)

# -----------------------------
# 3. Define response function
# -----------------------------
def generate_response(query, max_tokens=512, temperature=0.7, top_p=0.95, top_k=50):

    print(f"\n[Start] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()

    print("Prompt is ready, calling model...")
    
    output = llm(
        prompt=f"Q: {query}\nA:",
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )

    print("Model responded!")
    
    end_time = time.time()
    print(f"[End]   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Duration] {end_time - start_time:.2f} seconds")
    return output["choices"][0]["text"].strip()

# -----------------------------
# 4. Define multiple queries
# -----------------------------
queries = [
    "What is the capital of France?"
    #"What is the protocol for managing sepsis in a critical care unit?"
    #"What are the common symptoms for appendicitis, and can it be cured via medicine? If not, what surgical procedure should be followed to treat it?",
    #"What are the effective treatments or solutions for addressing sudden patchy hair loss, commonly seen as localized bald spots on the scalp, and what could be the possible causes behind it?"
]

# -----------------------------
# 5. Run all queries and display results
# -----------------------------
if __name__ == "__main__":
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Q: {query}")
        answer = generate_response(query)
        print(f"A: {answer}")

