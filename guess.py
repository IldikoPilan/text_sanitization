# Inference attack and matching between LLM guesses and original text span

import argparse
import random, json, copy, re
from transformers import set_seed
import spacy
from collections import Counter
from num2words import num2words
import calendar
import llm_inference_hf
import replacements as rep
import prompt_w_examples

MONTHS = [m for m in list(calendar.month_name) if m]

def convert2numeral(token):
    """
    Convert a token representing a number (in string form) into a spelled-out word if possible.

    Parameters:
    - token (str): The token (string) representing a number, which may include commas or spaces.

    Returns (str):
    - The token converted to a word representation of the number, or the original token if conversion fails.
    """
    token = token.replace(" ", "").replace(",", "").replace("%", "")
    try:
        # Transform string into a numeral
        if "." in token:
            numeral = float(token)
        else:
            numeral = int(token)
        # Spell out numeral with words
        return num2words(numeral)
    except:
        return token 

def fix_repl_list(replacements):
    """
    Re-extract items from numbered lists with square bracket delimiter (e.g. [1])
    (A format not handled during replacement candidate generation.)

    Parameters:
    - replacements (list of str): A list of replacement strings, potentially containing numbered items.

    Returns (list):
    - A list of cleaned replacement items without numbering prefixes.
    """
    new_repl_list = []
    orig_output = "".join(replacements)
    lines = list(filter(None, [l.strip() for l in orig_output.split("\n")]))
    for line in lines:
        if line[0] == "[" and line[1].isdigit() and line[2] == "]":
            new_repl_list.append(line[3:].lstrip())
    return new_repl_list

def convert_to_acronym(tkn_list):
    """
    Detects acronym in a list of tokens based on titlecasing information.
    
    Parameters:
    - tkn_list (list of str): A list of words (tokens) from which an acronym is to be created.

    Returns (str):
    - The acronym created from the title-cased tokens, or None if no acronym is detected.
    """
    acronym = ""
    for tkn in tkn_list:
        if tkn.istitle() and tkn not in ["A", "An", "The"]:
            acronym += tkn[0]
    if len(acronym) > 2:
        return acronym

def post_process_replacements(replacements, orig_span):
    """ 
    Remove replacement candidates that leak information from the original span:
    if the set of original and replacement lemmas is the same or if these strings 
    contain identical numeric information regardless of whether spelled out or in numerals.
    
    Parameters:
    - replacements (list of str): List of replacement candidate strings.
    - orig_span (str): The original text span for comparison.

    Returns (list):
    - Filtered list of replacements that are different from the original span.
    """
    # Filter replacements
    filtered_repl = replacements
    orig_lemmas = get_lemma_set(nlp_pipeline(orig_span)) 
    orig_lemmas = [convert2numeral(w) for w in orig_lemmas if w.replace(" ", "").replace(",", "").isdigit()]
    if orig_lemmas: 
        for repl in replacements:
            repl_lemmas = get_lemma_set(nlp_pipeline(repl))
            repl_lemmas = [convert2numeral(w) for w in repl_lemmas if w.replace(" ", "").replace(",", "").isdigit()] 

            # Check whether the set of original and replacement lemmas is the same
            if sorted(orig_lemmas) == sorted(repl_lemmas):
                filtered_repl.remove(repl)

            # Check number equivalents between digit and written-out versions
            else:
                for word in repl_lemmas:
                    if word.replace(" ", "").replace(",", "").isdigit():
                        conv_word = convert2numeral(word)
                        if conv_word in orig_lemmas:
                            if repl in filtered_repl:
                                filtered_repl.remove(repl) 
                for word in orig_lemmas:
                    if word.replace(" ", "").replace(",", "").isdigit():
                        conv_word = convert2numeral(word)
                        if conv_word in repl_lemmas:
                            if repl in filtered_repl:
                                filtered_repl.remove(repl) 
    return filtered_repl

def get_most_freq_lemmas(fname, n=100):
    """ 
    Create an save a list of the most frequent lemmas from a collection of texts.
    Excludes named entities, stopwords and non-alphabetical tokens. All list 
    items are lowercased.

    Parameters:
    - fname (str): File path of the JSON file containing document data.
    - n (int): Number of top most frequent lemmas to extract. Default is 100.

    Returns (list):
    - A list of tuples where each tuple contains a lemma and its frequency.
    """
    lemmas = [] 
    with open(fname, "r", encoding="utf-8") as outfile:
        all_data = json.load(outfile)
        for doc_obj in all_data:
            orig_text = doc_obj["text"]
            print("Processing ", doc_obj["doc_id"])

            # Process text w Spacy to get lemmas excluding stopwords
            proc_text = nlp_pipeline(orig_text)
            for tkn in proc_text:
                if not tkn.is_stop and tkn.is_alpha and tkn.ent_type_ == "": 
                    lemmas.append(tkn.lemma_.lower())
    
    # Get most freq lemmas
    lemma_counter = Counter(lemmas)
    top_most_common = lemma_counter.most_common(n)
    
    # Save resulting list as JSON
    with open("freq_words.json", "w") as f:
        json.dump([w for w,c in top_most_common], f, ensure_ascii=False, indent=4)
    return top_most_common

def get_guesses(bracketed_text, replacement, entity):
    """
    Create a prompt for LLM inference and retrieve guesses for re-identifying an entity.

    Parameters:
    - bracketed_text (str): The input text with brackets around the target entity.
    - replacement (str): Replacement text for the entity.
    - entity (dict): Entity details such as ID and type.

    Returns (list):
    - A list of re-identification guesses generated by the model.
    """
    # Prepare prompt
    prompt = copy.deepcopy(prompt_w_examples.messages_reidentif_1shot)
    prompt[-1]['content'] = prompt[-1]['content'].format(bracketed_text, f"{replacement}")

    # Get guesses
    output = llm_inference_hf.inference(prompt, tokenizer, model)
    guesses = rep.extract_list(output, entity["entity_mention_id"])
    return guesses

def get_ngrams(proc_text, ngram_size, only_ner=False):
    """ 
    Generate a set of unique n-grams for all tokens of a given preprocessed string,
    skipping the most frequent tokens.
    
    Parameters:
    - proc_text (spacy.tokens.Doc): object with linguistic annotations.
    - ngram_size (int): size of the ngrams to use.
    - only_ner (bool): whether to only extract ngrams if the token is a named entity.
    
    Returns (set of str):
    - A set of unique n-grams generated from the text.
    """
    ngrams = set()
    for token in proc_text:
        if not only_ner or token.ent_type_ not in ["", "DATE"]: # TODO: add and not token.replace(",","").isdigit()
            # Skip the most frequent tokens
            if token.lemma_.lower() not in ALLOWED_WORDS:
                for i in range(len(token.text) - ngram_size + 1):
                    ngrams.add(token.text[i:i+ngram_size])
    return ngrams

def get_lemma_set(proc_text):
    """
    Extract a set of lemmas from a processed text, excluding stopwords and punctuation.

    Parameters:
    - proc_text (spacy.tokens.Doc): A processed text object with token annotations.

    Returns (set of str):
    - A set of lemmas from the processed text.
    """
    return set([t.lemma_.lower() for t in proc_text if not t.is_stop and not t.is_punct])

def ordinal_to_cardinal(s):
    """
    Convert ordinal strings to their cardinal equivalents (e.g., '1st' -> '1').

    Parameters:
    - s (str): A string that may represent an ordinal number.

    Returns (str):
    - The converted cardinal number, or the original string if no conversion was made.
    """
    if len(s) > 2:
        if s[-2:] in ["st", "nd", "rd", "th"] and s[0].isdigit():
            return s.replace(s[-2:], "")
        else:
            return s
    return s

def proc_date(date_str):
    """
    Process a date string by splitting into elements and parsing its format.

    Parameters:
    - date_str (str): The input date string to be processed.

    Returns (tuple or None):
    - A tuple of the parsed date object and format, or None if parsing fails.
    """
    date_elems = re.split('\.|/|\s', date_str)
    date_len = len(date_elems)
    return rep.parse_date(date_str, date_len)

def is_guessed(guesses, original_span, orig_ent_type, replacement, ngram_size=4):
    """ 
    Check wether the original span is guessed based on either exact match, 
    lemma overlap or ngram overlap for named entities. Text is normalized to 
    lowercase in all comparisons. Saves the output in JSON with selected replacements
    added to all the information from the input file.

    Parameters:
    - guesses (list of str): List of guesses generated by the model.
    - original_span (str): Original span text to compare against guesses.
    - orig_ent_type (str): Type of the original entity (e.g., PERSON, CODE, DATETIME).
    - replacement (str): Suggested replacement text for the entity.
    - ngram_size (int): Length of n-grams for comparison. Default is 4.

    Returns (bool):
    - True if a match is found based on exact, lemma, or n-gram overlap; False otherwise.
    """
    # Check if any of the guesses is an exact match of the original span
    for guess in guesses:
        if guess.lower() == original_span.lower():
            print("GUESSED: exact match")
            return True
    
    processed_orig = nlp_pipeline(original_span)
    orig_lemmas = get_lemma_set(processed_orig)

    # Expand lemma list with automatically detected acronyms:
    orig_acronym = convert_to_acronym([tkn.text for tkn in processed_orig])
    if orig_acronym:
        orig_lemmas.add(orig_acronym)
    
    for guess in guesses:

        # Check lemma overlap and consider as guessed if the overlapping lemma is alphabetic, not a stop word, 
        # part of a named entity and not in the list of most frequent (and hence) allowed lemmas.
        processed_guess = nlp_pipeline(guess)
        guess_lemmas = get_lemma_set(processed_guess)
        guess_lemmas = set([l for l in guess_lemmas if l not in ALLOWED_WORDS])
        
        # Expand lemma list with automatically detected acronyms:
        _acronym = convert_to_acronym([tkn.text for tkn in processed_orig])
        guess_acronym = convert_to_acronym([tkn.text for tkn in processed_guess]) 
        if guess_acronym:
            guess_lemmas.add(guess_acronym)

        # Special handling of DATETIME 
        if orig_ent_type == "DATETIME": 

            # Try parsing dates and check whether they match
            orig_date = proc_date(original_span)
            guessed_date = proc_date(guess)
            if orig_date and guessed_date and (orig_date == guessed_date):
                print("GUESSED: parsed dates match")
                return True

            # Only considered guessed of all guess and original lemmas match (~ all the info guessed)  
            guess_lemmas = [ordinal_to_cardinal(l) for l in guess_lemmas]
            orig_lemmas = [ordinal_to_cardinal(l) for l in orig_lemmas]
            if sorted(orig_lemmas) == sorted(guess_lemmas):
                print("GUESSED: same set of orig and guess date lemmas")
                return True
        else:
            lemma_overlap = orig_lemmas.intersection(guess_lemmas)
            if lemma_overlap:
                print(f"GUESSED: LEMMA OVERLAP {lemma_overlap}")
                return True
            
            # Check if longer original lemmas contained in guessed lemmas and viceversa
            for o_lemma in orig_lemmas:
                if len(o_lemma) > 3:
                    for g_lemma in guess_lemmas:
                        if len(g_lemma) > 3:
                            if o_lemma in g_lemma or g_lemma in o_lemma:
                                print(f"GUESSED: LEMMA CONTAINED {lemma_overlap}")
                                return True

            # Check character ngram overlap for named entities e.g. Turkey - Turkish
            orig_ngrams = get_ngrams(processed_orig, ngram_size, only_ner=True)
            guess_ngrams = get_ngrams(processed_guess, ngram_size, only_ner=True)
            ngram_overlap = orig_ngrams.intersection(guess_ngrams)
            if ngram_overlap:
                print(f"GUESSED: NRGAM-OVERLAP for NER {ngram_overlap}")
                return True 
                    
def retrieve_replacements(entity_mention):
    """
    Retrieve replacements suggested by the LLM or rule-based methods for a given entity mention.

    Parameters:
    - entity_mention (dict): Dictionary with keys representing entity mention data, including replacement options.

    Returns (list of str):
    - List of replacements if found, or None if no replacements are present.
    """
    for k in entity_mention.keys():
        # Find the LLM or rulebased replacements
        if k.startswith("replacements"):
            return entity_mention[k]

def get_text_with_replacements(char_ixs_to_mention, selected_replacements, entity_to_guess_ixs, 
                                replacement, orig_text):
    """ 
    Prepare text with all annotated spans replaced by their 1st replacement
    then replace these incrementally with the selected replacements.

    Parameters:
    - char_ixs_to_mention (dict): Maps character index pairs to specific entity mentions.
    - selected_replacements (dict): Mapping of previously selected replacements for each entity span.
    - entity_to_guess_ixs (tuple): Character indices (start, end) of the span for re-identification.
    - replacement (str): Replacement text for the span.
    - orig_text (str): The original text containing entity spans.

    Returns (str):
    - Text with all mentions replaced, bracketed to identify entities under guessing. 
    """
    # Reconstruct the whole text plugging in replacements instead of the original spans
    bracketed_text_w_repl = ""
    current_end_ix = 0
    offset = 0
    for (start_ix, end_ix), ent_mention in sorted(char_ixs_to_mention.items()):

        # Take the original text from the beginning or between the previous span and the current 
        bracketed_text_w_repl += orig_text[current_end_ix:start_ix]
        
        # Plug in replacement instead of the original span
        if (start_ix, end_ix) == entity_to_guess_ixs:
            # Add brackets around the replacement to guess
            bracketed_text_w_repl += "[[" + replacement + "]]"
            
        else:
            # Check if mention has a selected replacement already, 
            # if yes use that
            if (start_ix, end_ix) in selected_replacements:
                bracketed_text_w_repl += selected_replacements[(start_ix, end_ix)]
            # if no selected replacement yet, choose most specific (first) replacement, if any
            else:
                other_mentions_repl = retrieve_replacements(char_ixs_to_mention[(start_ix, end_ix)])
                try: 
                    bracketed_text_w_repl += other_mentions_repl[0]
                except:
                    # Use entity type as replacement if there are no replacement suggestions
                    bracketed_text_w_repl += char_ixs_to_mention[(start_ix, end_ix)]["entity_type"]

        current_end_ix = end_ix
    
    # Add any remaining original text behind the last span
    if current_end_ix < len(orig_text):
        bracketed_text_w_repl += orig_text[current_end_ix:]
    
    return bracketed_text_w_repl

def select_replacements(fname, limit, tokenizer, model, model_type, outpath):
    """
    For each annotated span in TAB file passed, it selects a replacement 
    out of the 5 (LLM- or rule-based) suggestions. Selection is based on guesses by an LLM 
    based on the whole text where spans to process are selected in random order. Spans not yet 
    guessed are set to the first (most specific) replacement option.

    Parameters:
    - fname (str): Path to the input JSON file containing entity annotations.
    - limit (int): Maximum number of documents to process. If None, process all.
    - tokenizer (object): Tokenizer used for text processing.
    - model (object): Model used for text inference.
    - model_type (str): Type of the model (e.g., LLM, rule-based).
    - outpath (str): Path to save the output JSON file with selected replacements.

    Returns (None):
    - The function writes the updated document with replacements to the specified output path.
    """
    random.seed(1234)
    doc_lens = []
    nr_unparsed_guesses = 0
    with open(fname, "r", encoding="utf-8") as outfile:
        all_data = json.load(outfile)
        updated_data = [] # list of doc obj w replacements (unaltered doc objects not included)
        if limit:
            all_data = all_data[:limit]

        for doc_obj in all_data:
            doc_id = doc_obj["doc_id"]
            print(f"Processing {doc_id}...")
            updated_doc_obj = copy.deepcopy(doc_obj) # will contain the orig info + replacements (+ sent info?)
            orig_text = doc_obj["text"]
            #print("ORIG TEXT: ", orig_text)

            # For each entity mention, try to guess original span with an LLM             
            entity_ids = []
            for annotator in updated_doc_obj["annotations"]:
                all_mentions_in_doc = updated_doc_obj["annotations"][annotator]["entity_mentions"]
                char_ixs_to_mention = {(entm["start_offset"], entm["end_offset"]) : entm for entm in all_mentions_in_doc}
                mention_ixs = list(char_ixs_to_mention.keys())
                # Shuffle mention indices to process them in random order 
                random.shuffle(mention_ixs) 
                selected_replacements = {}
                ent_ids_for_running_ids = []

                for mix, entity_to_guess_ixs in enumerate(mention_ixs):
                    entity_to_guess = char_ixs_to_mention[entity_to_guess_ixs]
                    mention_ix_to_update = all_mentions_in_doc.index(entity_to_guess)
                    print(f"Guessing mention nr {mix+1}: ", entity_to_guess["span_text"])
                    replacements = retrieve_replacements(entity_to_guess)

                    # Apply bug fix and re-extract replacements
                    if len(replacements) > 5:
                        replacements = fix_repl_list(replacements)

                    # Post process replacements to filter out some information leaking ones
                    if entity_to_guess["entity_type"] not in ["PERSON", "CODE", "DATETIME"]:
                        replacements = post_process_replacements(replacements, entity_to_guess["span_text"])
                        
                        if not replacements:
                            if entity_to_guess["entity_id"] not in ent_ids_for_running_ids:
                                ent_ids_for_running_ids.append(entity_to_guess["entity_id"])
                                selected_repl = entity_to_guess["entity_type"] + "_" + str(ent_ids_for_running_ids.index(entity_to_guess["entity_id"]))
                    
                    # Set selected replacement to the entity type if PERSON or CODE which have no automatic replacements
                    if entity_to_guess["entity_type"] in ["PERSON", "CODE"]:
                        selected_repl = replacements[0]
                    else:
                        # Try to guess the original span based on each replacement and the rest of the text 
                        # (Replacements are assumed to be sorted by the LLM from least to most specific) 
                        for ix, replacement in enumerate(replacements):
                            
                            # Create a copy of the original text with all mentions replaced by their replacement 
                            # (most specific, or other, already selected one) 
                            bracketed_text_w_repl = get_text_with_replacements(char_ixs_to_mention, selected_replacements, 
                                entity_to_guess_ixs, replacement, orig_text)

                            # Get guesses
                            guesses = get_guesses(bracketed_text_w_repl, replacement, entity_to_guess)
                            orig_span = entity_to_guess["span_text"]
                        
                            # If cannot identify list of guesses in LLM response, do not try to guess, move to next candidate
                            if type(guesses) == str:
                                nr_unparsed_guesses += 1
                                print("Nr unparsed guesses:", nr_unparsed_guesses)
                                guessed = True
                            # Get guesses and match all of them against the original span
                            else:
                                # Filter guesses that are identical to the replacement suggestion
                                guesses = [guess for guess in guesses if guess != replacement]
                                if guesses:
                                    print(f"Guesses for repl: {replacement} (orig: {orig_span}):")
                                    print(guesses)
                                    guessed = is_guessed(guesses, orig_span, entity_to_guess["entity_type"], replacement)
                                # No guesses left after filtering, move to next candidate if any
                                else:
                                    guessed = True

                            # If not guessed, choose current replacement
                            if not guessed:
                                print(f"Replacement selected since not guessed '{replacement}' (orig: {orig_span})")
                                selected_repl = replacement
                                break
                            # If guessed, try next replacement option
                            elif ix < len(replacements)-1:
                                continue
                            # When all replacement options are guessed, use the entity type and a running id as replacement
                            else:
                                print("will mask with entity type", ix)
                                if entity_to_guess["entity_id"] not in ent_ids_for_running_ids:
                                    ent_ids_for_running_ids.append(entity_to_guess["entity_id"])
                                selected_repl = entity_to_guess["entity_type"] + "_" + str(ent_ids_for_running_ids.index(entity_to_guess["entity_id"]))
                                
                        # Add all LLM guesses to the JSON object
                        updated_doc_obj["annotations"][annotator]["entity_mentions"][mention_ix_to_update]["guesses_sel_repl"] = guesses

                    # Add selected replacement to the JSON object    
                    print(f"Adding '{selected_repl}' as selected replacement")
                    selected_replacements[entity_to_guess_ixs] = selected_repl
                    updated_doc_obj["annotations"][annotator]["entity_mentions"][mention_ix_to_update]["selected_replacement"] = selected_repl
                    
                    # Update list of replacements candidates (since some are post-processed) 
                    if "replacements_rulebased" in updated_doc_obj["annotations"][annotator]["entity_mentions"][mention_ix_to_update].keys():
                        updated_doc_obj["annotations"][annotator]["entity_mentions"][mention_ix_to_update]["replacements_rulebased"] = replacements
                    else:
                        updated_doc_obj["annotations"][annotator]["entity_mentions"][mention_ix_to_update]["replacements_mistral"] = replacements
                            
                # Save in the output JSON a copy of text with selected replacements (re-use the last bracketed text and plug in last selected replacement)
                san_text = bracketed_text_w_repl.replace("[[" + replacement + "]]", selected_repl)

                # Correct double articles when inserting replacements (the a -> a / The a -> A)
                updated_doc_obj["sanitized_text"] = san_text.replace(" the a ", " a ").replace(" The a ", " A ")
                updated_data.append(updated_doc_obj)
                
                # Save document object updated with selected replacements 
                print("Saving updated doc obj")
                with open(outpath, "w") as f:
                    json.dump(updated_data, f, ensure_ascii=False, indent=4)
                print()
       

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
    argparser.add_argument("-fw", "--freq_words", type=str, 
                            help="Path to list of most frequent words to exclude when matching guesses to original spans.", 
                            default="freq_words.json")
    argparser.add_argument("-out", "--out_path", type=str, 
                            help="Path to file where to write the output.", 
                            default="TAB_sel_repl.json")
    args = argparser.parse_args()

    # Load tokenizer and model
    set_seed(99)
    model_type = args.llm_name.split('/')[-1].split('-')[0].lower()
    tokenizer, model = llm_inference_hf.load_tokenizer_and_model(args.llm_name, q_config=llm_inference_hf.bnb_config)
    nlp_pipeline = spacy.load("en_core_web_sm")
    
    # Load list of most frequent words
    with open(args.freq_words) as f:
        ALLOWED_WORDS = json.load(f)
    
    # Run replacement selection
    select_replacements(args.data_path, args.max_doc, tokenizer, model, model_type, args.out_path)


