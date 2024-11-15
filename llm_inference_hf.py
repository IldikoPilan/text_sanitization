""" Loading LLMs and using them for inference.
"""

import os
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Configurations for quantization
bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_use_double_quant=True, # nested quantization for more memory efficiency (quantization constants from the first quantization are quantized again)
                                bnb_4bit_quant_type="nf4",            # quantization data type in the bnb.nn.Linear4Bit layers
                                bnb_4bit_compute_dtype=torch.bfloat16 # default torch.float32 reduced for speedup
                                )

def load_tokenizer_and_model(base_model_name, adapter_model_name=None, q_config={}):
    """ 
    Loads a tokenizer and a base model from Hugging Face, and optionally 
    merges it with a trained adapter.
    
    Parameters:
    - base_model_name (str): The name of the base model to be loaded.
    - adapter_model_name (str, optional): The name of the adapter model to be loaded and 
        merged with the base model. Defaults to None.
    - bnb_config (dict, optional): Configuration settings for quantization. If empty, 
        the model will not be quantized. Defaults to an empty dictionary.
    
    Returns (tuple):
    - tokenizer: The loaded tokenizer.
    - model: The loaded (and possibly merged) model.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, add_eos_token=True)

    # Load the base model with optional quantization configuration
    model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=q_config, device_map="auto")
    if adapter_model_name:
        print("Loading and merging adapter weights")
        model = PeftModel.from_pretrained(model, adapter_model_name)

    return tokenizer, model

def inference(chat, tokenizer, model):
    """
    Use a pre-trained tokenizer and model to perform inference using the provided chat input 
    with a template applied to it matching the model.
    
    Parameters:
    - chat (list): The input chat for which a response is to be generated. Should be a list
                     of dictionaries with 'role' ('user' or 'assistant') and 'content' (the message) keys.
    - tokenizer (object): A pre-trained tokenizer object capable of tokenizing input chats.
    - model (object): A pre-trained model capable of generating responses based on input chats.
    
    Returns (str):
    - The generated response for the input chat.
    """
    start = time.time()
    input_ids =  tokenizer.apply_chat_template(chat, return_tensors="pt").to("cuda") 
    outputs = model.generate(input_ids, 
                             max_new_tokens=256,
                             do_sample=True,
                             temperature=0.3, 
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.eos_token_id,
                             use_cache=True) 
                             
    
    # Extract the generated tokens
    input_length = input_ids.shape[1]
    # Retain only the generated tokens and omit the repeated input
    generated_tokens = outputs[:, input_length:]  
    response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    #print("Inference time =", round(time.time() - start, 2))

    return response

if __name__ == "__main__":
    llm_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer, model = load_tokenizer_and_model(llm_name, bnb_config_4bit)
    print(f"Small LLM inference demo with {llm_name}")
    chat = [{"role":"user", "content": "Guess who this sentence is about: 'She was born on 17 January 1955 and she is a Hungarian-American biochemist.'"}]
    output = inference(chat, tokenizer, model)
    print(output)



