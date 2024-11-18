## General

This repository contains the implementation of a text sanitization method called INference-
guided Truthful sAnitization for Clear Text (INTACT), currently being submitted for review. 

## Files

- llm_inference.py: loading an LLM and using it for inference
- prompt_w_examples.py: prompts used for replacement generation and selection 
- replacements.py: replacement generation functions
- guess.py: replacement selection (based on inference attack and matching LLM guesses to original text spans)
- freq_words.json: most frequent words from the training set of TAB (matches between a guess and the original span are allowed for these words)
- requirements.txt: dependencies
- TAB_sanitized.zip: the sanitized version of the Text Anonymization Benchmark (TAB) dataset. The original TAB dataset can be downloaded from [here](https://github.com/NorskRegnesentral/text-anonymization-benchmark).
- manual_eval_data_spec_truth.xlsx: a subset of sanitized TAB with manual evaluation scores for replacement specificity nand truthfulness.

## Data format

The sanitized TAB has the following additional attributes compared to the original TAB:

| Variable name      | Description       |
|----------------|----------------|
| replacements_mistral (or replacements_rulebased)  | Replacement candidates generated with a Mistral language model (or based on heuristics).  |
| selected_replacement  | The selected replacement out of the candidates listed under 'replacements_*'  |
| guesses_sel_repl  | The results of the inference attack for the selected replacement.  |

## Example calls 

1) Replacement generation for the development set of TAB: 

```{python}
python replacements.py -d text-anonymization-benchmark/echr_dev.json -max 3 -out TAB_dev_repl.json 
```

2) Replacement selection on the output of the previous replacement generation process:

```{python}
python guess.py -d TAB_dev_repl.json -out TAB_dev_sel_repl.json
```
