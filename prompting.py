import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import os
import pandas as pd

os.environ['HF_TOKEN']='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'

# Global Variables
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 200
TOP_K = 50
TOP_P = 0.9
PENALTY = 1.2

def read_movies_desc(path):
   movies = pd.read_csv(path, delimiter='::', names=['movieId', 'title', 'year', 'genres','description'], engine='python')
   return movies

def generateBasicDescriptions(model,tokenizer, movies, batch_size, output_file):
    mov_descriptions = [{"movieId": mov["movieId"], "description": mov["description"]} for mov in movies]
    tokenizer.pad_token = tokenizer.eos_token
    basic_desc = []

    print(f"Generating basic descriptions...")

    with open(output_file, "w") as f:
      for i in range(0, len(mov_descriptions), batch_size):
          torch.cuda.empty_cache()
          print(f"Processing batch {i//batch_size+1}/{len(mov_descriptions)//batch_size}")
          batch_input = [
            f'''Summarize the following movie description in under 50 words. Keep all essential details. Do not add or remove details.
                {desc["description"]}
            ''' for desc in mov_descriptions[i:i+batch_size]
          ]
          inputs = tokenizer(batch_input, padding=True, truncation=True, return_tensors="pt").to("cuda")
          outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=PENALTY,
          )
          results = []
          for i, output in enumerate(outputs):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            cleaned = decoded.replace(batch_input[i], "").strip()
            results.append(cleaned)

          for result in results:
              f.write(result + "\n::\n")
              basic_desc.append(result)

    print(f"Paraphrased descriptions saved to {output_file}")

    return basic_desc

def generateRecDrivenDescriptions(model,tokenizer, movies, batch_size, output_file):
    mov_descriptions = [{"movieId": mov["movieId"], "description": mov["description"]} for mov in movies]
    tokenizer.pad_token = tokenizer.eos_token

    recDriven_desc = []

    print(f"Generating descriptions...")

    with open(output_file, "w") as f:
      for i in range(0, len(mov_descriptions), batch_size):
          torch.cuda.empty_cache()
          print(f"Processing batch {i//batch_size+1}/{len(mov_descriptions)//batch_size}")
          batch_input = [
            f'''Use the movie description and state what you would say to someone to recommend the movie to them.
                {desc["description"]}
            ''' for desc in mov_descriptions[i:i+batch_size]
          ]
          inputs = tokenizer(batch_input, padding=True, truncation=True, return_tensors="pt").to("cuda")
          outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=PENALTY,
          )
          results = []
          for i, output in enumerate(outputs):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            cleaned = decoded.replace(batch_input[i], "").strip()
            results.append(cleaned)

          for result in results:
              recDriven_desc.append(result)
              f.write(result + "\n::\n")

    print(f"Paraphrased descriptions saved to {output_file}")

    return

if __name__ == "__main__":
    print("Starting prompting...")
    start = time.time()
    
    # Setting up LLM to enhance descriptions - must have HF token to access
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    end = time.time()
    print(f"Model loaded in {(end-start):.2f} seconds.")

    # Read movie_desc.dat to extract descriptions
    path = "movielens-1m/movies_desc.dat"
    movies = read_movies_desc(path)

    batch_size = 2 # higher batch sizes lead to CUDA Out of Memory Errors

    # Generate basic prompting results and save to basic_prompting.txt
    basic_desc = generateBasicDescriptions(model, tokenizer, movies, batch_size)

    # Generate recommendation driven prompting results and save them to recDriven_prompting.txt
    recDriven_desc = generateRecDrivenDescriptions(model, tokenizer, movies, batch_size)

    for mov in movies:
       for d1, d2 in zip(basic_desc, recDriven_desc):
          mov["description"] += (d1 + d2)
    
    # Save enhanced descriptions to movies_enhanced_desc.dat
    save_path = "movielens-1m/movies_enhanced_desc.dat"
    with open(save_path, "w", encoding="utf-8") as f:
        for mov in movies:
            f.write(mov["movieID"] + "::" + mov["title"] + "::" + mov["year"] + "::" + mov["genres"] + "::" + mov["description"] + "\n")