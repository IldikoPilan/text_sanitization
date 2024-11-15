import json
import random
import argparse
import spacy
import copy
import re
import ast
import copy
from datetime import datetime
from transformers import set_seed
import prompt_w_examples
import llm_inference_hf


##################################
# Utils for LLM-based replacements
##################################

def split_into_sentences(text, nlp_pipeline):
    """ Splits the input 'text' into individual sentences using the SpaCy NLP pipeline 
    and returns a list of tuples containing the sentence's character offsets and the sentence text.

    Parameters:
    - text: The input text to split into sentences.
    - nlp_pipeline: The SpaCy NLP pipeline for processing the text.

    Returns:
    - A sorted list of tuples, each containing:
      - start_char: The start character index of the sentence.
      - end_char: The end character index of the sentence.
      - sentence_text: The text of the sentence.
    """
    # Process the text with spaCy
    doc = nlp_pipeline(text)
    sentences_with_indices = []
    for sent in doc.sents:
        start_char = sent.start_char
        end_char = sent.end_char
        sentence_text = sent.text
        sentences_with_indices.append((start_char, end_char, sentence_text))
    return sorted(sentences_with_indices)

def format_example_output(example_output, bullet='- '):
    """ 
    Parameters:
    - example_output: A list of strings to be formatted.
    - bullet: The symbol to prepend to each list item. Default is '- '.

    Returns:
    - A string where each item in example_output is preceded by the specified bullet point.
    """
    return "\n".join([bullet + item for item in example_output])

def prepare_prompt(instance, span_text, ent_type):
    """ 
    Prepares the prompt template for an LLM model, filling it with an instance, span text, and entity type, 
    along with example replacements based on the entity type.

    Parameters:
    - instance: The dataset instance for which the prompt is being prepared.
    - span_text: The span of text within the instance that corresponds to an entity.
    - ent_type: The entity type (e.g., "PERSON", "DATE").

    Returns:
    - A list of messages, with each message formatted according to the prompt template and entity-specific examples.
    """
    messages = copy.deepcopy(prompt_w_examples.messages)
    ex_bracketed_ent = prompt_w_examples.examples[ent_type]["in"].split("[[")[-1].split("]]")[0]
    messages[0]['content'] = messages[0]['content'].format(prompt_w_examples.examples[ent_type]["in"], ex_bracketed_ent)
    messages[1]['content'] = messages[1]['content'].format(format_example_output(prompt_w_examples.examples[ent_type]["out"]))
    messages[2]['content'] = messages[2]['content'].format(instance, f"[[{span_text}]]")
    return messages

def remove_parenthesis(text):
    """ 
    Parameters:
    - text: The input text from which parentheses content will be removed.

    Returns:
    - The input text with all content within parentheses removed.
    """
    pattern = r"\([^)]*\)"
    result = re.sub(pattern, "", text)
    return result.strip()

def extract_list(output_str, mention_id, bullet="-"):
    """ 
    Extracts a list of replacements or guesses from the LLM output. It checks if the output is in a list format, 
    handles bulleted and numbered lists, and removes any comments in parentheses.

    Parameters:
    - output_str: The string containing the LLM output.
    - mention_id: A unique identifier for the entity mentioned in the output.
    - bullet: The bullet symbol used in the list. Default is "- ".

    Returns:
    - A list of replacements or guesses extracted from the output string, with a maximum of five items, 
      or the entire response as a string if no list is detected.
    """
    replacements = []
    lines = list(filter(None, [l.strip() for l in output_str.split("\n")]))

    # Check if response appears to have a Python list formating and parse if possible.
    if output_str[0] == "[" and output_str[-1] == ["]"] and not output_str[1].isdigit():
        try:
            replacements = ast.literal_eval(output_str)
        except SyntaxError:
            pass

    # Look for bulleted list entries in output lines
    else:
        for line in lines:
            if len(line) > 3:
                if line.startswith(bullet) and line != '---': # exclude cases like '---'
                    replacements.append(line.replace(bullet, "", 1).lstrip())
                elif line[0].isdigit() and line[1] in [".", ")", " ", ":"]:     # handle numbered lists with different separators
                    replacements.append(line[2:].lstrip())
                elif line[0] == "[" and line[1].isdigit() and line[2] == "]":   # handle [1] numbered list formating (later addition in July 2024)
                    replacements.append(line[3:].lstrip())
                elif line.lower().startswith("guess") and ":" in line:          # Format 'Guess 1:'
                    replacements.append(line.split(":")[1].lstrip())
                elif line.startswith("Note:"):                                  # Skip final notes
                    continue

    # Take only the first five items, sometimes more are generated
    if replacements:
        if len(replacements) > 5:
            replacements = replacements[:5] 
        # Remove comments in parenthesis
        return [remove_parenthesis(repl) for repl in replacements]

    # Return whole response as string if a list cannot be detected     
    else:
        print("NO LIST: ", mention_id, output_str)
        return output_str

def match_casing(orig_text, adjusted_start_ix, repl_list):
    """ 
    Adjusts the casing of replacements based on the original text and the position of the entity in the sentence.

    Parameters:
    - orig_text: The original text in which the entity appears.
    - adjusted_start_ix: The adjusted starting index of the entity in the sentence.
    - repl_list: The list of potential replacements for the entity.

    Returns:
    - A list of replacements with the casing adjusted to match the original text's style.
    """
    repl_list_reformatted = []
    for repl in repl_list:
        if orig_text.islower() or adjusted_start_ix > 1:
            first_word = repl.split(" ")[0]
            if first_word in ["A", "An", "The", "With", "In", "About"]:
                repl = repl[0].lower() + repl[1:]
        repl_list_reformatted.append(repl)
    return repl_list_reformatted


###################################
# Utils for rule-based replacements
###################################

def parse_date(date_str, date_len):
    """ 
    Parses the date string based on its length and attempts to match it to various date formats.

    Parameters:
    - date_str: The date string to parse.
    - date_len: The number of components in the date string (e.g., year, month, day).

    Returns:
    - A tuple containing the parsed date object and the matching date format string if successful,
      or None if parsing fails.
    """
    if date_len < 2: # year only
        date_formats = ['%Y', '%y']
    elif date_len < 3: # month year only
        date_formats = ['%m.%Y', '%m %Y', '%m.%y', '%m %y',  '%m/%Y', '%m/%y', '%B %Y', '%m/%Y', '%b %Y', '%b/%Y',
                        '%d %B', '%d %b', '%d. %B', '%d. %b']
    else:
        date_formats = ['%d %m %Y', '%d %m %y', '%d.%m.%Y', '%d.%m.%y', '%d/%m/%Y', '%d/%m/%y', 
                        '%d %B %Y', '%d. %B %Y.', '%d %B, %Y', '%d/%B/%Y', '%d/%B/%y', 
                        '%d %b %Y', '%d. %b %Y.', '%d %b, %Y', '%d/%b/%Y', '%d/%b/%y']
    for d_format in date_formats:
        try:
            return (datetime.strptime(date_str, d_format), d_format)
        except ValueError:
            pass

def get_date_replacements(date_str):
    """ 
    Parse and generalize a date into different levels of abstraction.
    This function breaks down a date string into its components and creates various 
    generalized forms of the date based on the level of detail provided in the original date.

    Parameters:
    - date_str: A string representing the date to be parsed (e.g., "15/06/2023").

    Returns:
    - A list of strings representing different generalizations of the input date. 
      The generalizations range from specific (e.g., "June 2023") to broad (e.g., "21st century").
      If the date is not parsable, returns None.
    """
    date_elems = re.split('\.|/|\s', date_str)
    date_len = len(date_elems)

    # Try to recognize and parse date format    
    try:
        date_obj, date_format = parse_date(date_str, date_len)
    except TypeError: # date not parsable
        return

    # Create date generalizations
    month_year = date_obj.strftime('%B %Y')
    year = date_obj.strftime('%Y')
    if date_obj.day >= 15:
        part_of_month_binary = f"second half of {date_obj.strftime('%B')}"
    else:
        part_of_month_binary = f"first half of {date_obj.strftime('%B')}"
    part_of_month = f"{['beginning', 'middle', 'end', 'end'][date_obj.day//10]} of {date_obj.strftime('%B')}"
    season = ['winter', 'spring', 'summer', 'autumn'][date_obj.month % 12 // 3]
    part_of_year = f"{season} of {year}"
    decade = f"{date_obj.year - (date_obj.year % 10)}s"
    part_of_decade = f"{['beginning', 'middle', 'end'][int(((date_obj.year % 10)-1) / 3)]} of {decade}"
    century = f"{date_obj.year // 100 + 1}th century"
    part_of_century = f"{['beginning', 'middle', 'end', 'end'][int(((date_obj.year % 100)-1) / 30)]} of {century}"
    
    # Compare the parsed date to the current date
    present = datetime.now()
    if date_obj <= present:
        past_or_future = "a date in the past"
    else:
        past_or_future = "a future date"
    
    # Return generalizations whose specificity reflect the specificity of the original date 
    # (e.g. if only year given, generalize up to century; if even day given, generalize only up to decade) 
    if date_len == 1: # year only
        return [part_of_decade, decade, part_of_century, century, past_or_future]
    elif date_len == 2: 
        if "y" in date_format.lower(): # month year
            return [part_of_year, year, part_of_decade, decade, part_of_century]
        else: # day month
            return [part_of_month, part_of_month_binary, date_obj.strftime('%B'), season, f"{date_obj.strftime('%d')} of an unknown month"]
    else: # day month year
        return [month_year, part_of_year, year, part_of_decade, decade]

############################
# Main replacement functions
############################

def get_replacements(entity, running_ids, model_type, span_text, current_sent, 
                     current_sent_start, tokenizer, model):
    """ 
    Obtain replacements for a given entity using either rule-based or LLM-based methods.
    This function checks if a suitable rule-based replacement is available for the entity type.
    If not, it prepares the input for a large language model (LLM) to generate the replacements.
    
    Parameters:
    - entity (dict): information about the entity, including 'entity_type' (e.g., "PERSON") and offsets.
    - running_ids (dict): a tracking dictionary that assigns unique identifiers to entities by type.
    - model_type (str): the type of LLM used.
    - span_text (str): the specific words of the entity within the current sentence.
    - current_sent (str): the complete text of the current sentence containing the entity.
    - current_sent_start (int): starting index of the current sentence relative to the full text.
    - tokenizer (object): the tokenizer to use for LLM input preparation.
    - model: (object): the model used to generate replacements.

    Returns:
    - replacements (list): the generated or rule-based replacements for the entity.
    - replacement_type (str): indicates whether the replacement is "rulebased" or generated by the model_type.
    - running_ids (dict): updated dict with incremented counters for applicable entity types.
    """
    # Rule-based replacement with entity label and a running number
    if entity['entity_type'] in ["PERSON", "CODE"]:
        replacements = [f"{entity['entity_type']}_{running_ids[entity['entity_type']]}"]
        running_ids[entity['entity_type']] += 1
        return replacements, "rulebased", running_ids
    else:
        # Try to get rule-based replacements for common date formats
        if entity['entity_type'] == "DATETIME":
            replacements = get_date_replacements(span_text)
            if replacements:
                running_ids[entity['entity_type']] += 1
                return replacements, "rulebased", running_ids
        else:
            replacements = []

        # Get LLM replacements for entity if no rule-based ones yet
        if not replacements:

            # Calculate the entity's position within the current sentence for accurate placement
            ent_start_ix = entity["start_offset"] - current_sent_start 
            ent_end_ix = entity["end_offset"] - current_sent_start

            # Prepare LLM input with sentences with square bracketed entities. 
            bracketed_sent = f"{current_sent[:ent_start_ix]}[[{span_text}]]{current_sent[ent_end_ix:]}"
            bracketed_sent = bracketed_sent.replace("\n\n", " ")  
            print("SENT: ", bracketed_sent)
            
            # LLM processing
            prompt = prepare_prompt(bracketed_sent, span_text, entity['entity_type'])
            output = llm_inference_hf.inference(prompt, tokenizer, model)
            replacements = extract_list(output, entity["entity_mention_id"])
            replacements = match_casing(span_text, ent_start_ix, replacements)
            print("LLM repl: ", replacements)

            return replacements, model_type, running_ids

def add_replacements(fname, model_type, tokenizer, model, limit=None, outpath="repl.json"):
    """ For each document within 'limit' amount, pick a random annotator and 
    for each unique entity, add sorted replacement candidates (with rules or LLM).
    The context for the LLM is a sentence, split with Spacy. 
    Saves output to a file.

    Parameters:
    - fname (str): path to a TAB-style corpus file in JSON
    - model_type (str): shorthand for model used as keyword in JSON attribute, e.g. 'mistral'
    - tokenizer (object): A pre-trained tokenizer object capable of tokenizing input chats.
    - model (object): A pre-trained model capable of generating responses based on input chats.
    - limit (int): maximum number of documents to process 
    - outpath (str): path of output JSON file

    Returns:
    - A list with the (subset of the) document objects for which replacements were added.
    """
    # Load the English Spacy language processing pipeline
    nlp_pipeline = spacy.load("en_core_web_sm", exclude=["ner"])
    random.seed(1234)
    nr_no_repl_ents = 0
    no_lookups = []
    with open(fname, "r", encoding="utf-8") as outfile:
        all_data = json.load(outfile)
        updated_data = [] # list of doc obj w replacements (unaltered doc objects not included)
        # when adding replacements with additional LLMs, do not overwrite
        if not updated_data:
            if 'repl' not in fname: 
                with open(outpath, "w") as f:
                    json.dump(updated_data, f, ensure_ascii=False, indent=4)
        if limit:
            all_data = all_data[:limit]

        for doc_obj in all_data:
            doc_id = doc_obj["doc_id"]
            print(f"Processing {doc_id}...")
            updated_doc_obj = copy.deepcopy(doc_obj) 
            running_ids = {k:0 for k in prompt_w_examples.examples}
            ent_id_to_repl = {} # keeps track of replacements per unique entity in text for coreference
            orig_text = doc_obj["text"]
            
            # Process text w spacy & get list of sents w char ix (the context for each entity will be the sentence they occur in.)
            sents_w_ix = split_into_sentences(orig_text, nlp_pipeline)   
            current_sent_ix = 0
            
            # Only take 1 randomly seleceted annotator for docs with multiply annotated versions 
            random_annotator = random.choice(list(doc_obj["annotations"].keys()))
            for ent_mention_ix, entity in enumerate(updated_doc_obj["annotations"][random_annotator]["entity_mentions"]):
                ent_copy = copy.deepcopy(updated_doc_obj["annotations"][random_annotator]["entity_mentions"][ent_mention_ix])
                current_sent_start, current_sent_end, current_sent = sents_w_ix[current_sent_ix]
                ent_start_ix = entity["start_offset"]
                ent_end_ix = entity["end_offset"]
                span_text = entity["span_text"]
                
                # Check if entity in current sent
                while ent_end_ix > current_sent_end:
                    current_sent_ix += 1
                    current_sent_start, current_sent_end, current_sent = sents_w_ix[current_sent_ix]
                    # Skip spans that are multiple sentences long
                    if ent_end_ix <= current_sent_end and ent_start_ix < current_sent_start:
                        print("multisent span: ", ent_start_ix, ent_end_ix, current_sent_start, current_sent_end)
                        no_lookups.append(entity["entity_id"])
                        continue

                # Get replacements if previous mentions of the same entity do not yet have replacements
                if entity["entity_id"] not in ent_id_to_repl: 
                    replacements, replacement_type, running_ids = get_replacements(entity, running_ids, model_type, span_text, current_sent, 
                                                                                   current_sent_start, tokenizer, model)
                    if replacement_type == model_type:
                        entity[f"context_{model_type}"]  = {'start_offset':current_sent_start, 'end_offset':current_sent_end}
                        if type(replacements) != list: 
                            nr_no_repl_ents += 1
                    ent_id_to_repl[entity["entity_id"]] = (replacements, replacement_type)                         
                 # Get already collected replacements for this entity, don't look up this mention again
                else:
                    replacements, replacement_type = ent_id_to_repl[entity["entity_id"]]
                entity[f"replacements_{replacement_type}"] = replacements
                updated_doc_obj["annotations"][random_annotator]["entity_mentions"][ent_mention_ix] = entity

            # Remove annotators & their annotations that were not used for replacement lookups
            for annotator in updated_doc_obj["annotations"].copy():
                if annotator != random_annotator:
                    del updated_doc_obj["annotations"][annotator]
            updated_data.append(updated_doc_obj)
            
            with open(outpath, "w") as f:
                json.dump(updated_data, f, ensure_ascii=False, indent=4)

            # Print some statistics
            print()
            print('Nr docs updated:', len(updated_data))
            print("Unparsable repl lists (unique entities):", nr_no_repl_ents)
            print("No lookups (multispan entities):", len(set(no_lookups)))
            print()

    return updated_data


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--data_path", type=str, 
                           help="Path to TAB dataset.", required=True)
    argparser.add_argument("-llm", "--llm_name", type=str, 
                            help="Huggingface name for the LLM to use.", 
                            default="mistralai/Mistral-7B-Instruct-v0.2") 
    argparser.add_argument("-max", "--max_doc", type=int, 
                            help="Maximum number of documents to process.", 
                            default=None)
    argparser.add_argument("-out", "--out_path", type=str, 
                            help="Path to file where to write the output.", 
                            default="TAB_repl.json")
    args = argparser.parse_args()

    set_seed(99)

    # Load tokenizer and model
    model_type = args.llm_name.split('/')[-1].split('-')[0].lower()
    tokenizer, model = llm_inference_hf.load_tokenizer_and_model(args.llm_name, q_config=llm_inference_hf.bnb_config)
    
    # Find LLM-based replacement suggestions for TAB
    add_replacements(args.data_path, model_type, tokenizer, model, args.max_doc, args.out_path) 

# Example call for adding LLM replacements to TAB
# python replacements.py -d text-anonymization-benchmark/echr_dev.json -max 10
