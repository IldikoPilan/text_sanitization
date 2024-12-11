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
- TAB_all_INTACT.zip: the sanitized version of the Text Anonymization Benchmark (TAB) dataset using INTACT and two other state-of-the-art baselines: GPT 3.5-generated replacements with [Presidio](https://github.com/microsoft/presidio/blob/main/docs/samples/python/synth_data_with_openai.ipynb) and the self-disclosure abstraction model from [Dou el al. (2024)](https://huggingface.co/douy/Llama-2-7B-lora-instruction-ft-abstraction-three-span). The original TAB dataset can be downloaded from [here](https://github.com/NorskRegnesentral/text-anonymization-benchmark).
- manual_eval_data_spec_truth_INTACT.xlsx: a subset of TAB sanitized with INTACT and manually evaluated for specificity and truthfulness.
- manual_eval_data_spec_truth_Dou.xlsx: a subset of TAB sanitized with Dou et al.'s model and manually evaluated for specificity and truthfulness.
- manual_eval_data_spec_truth_Presidio.xlsx: a subset of TAB sanitized with Presidio synthetic replacements and manually evaluated for specificity and truthfulness.
- eval_TPS.py: replacements utility evaluation using the Text Preserved Similarity (TPS) metric proposed in the paper.
- eval_doc_clustering.py: replacements utility evaluation using the document clustering downstream task.

## Data format

The sanitized TAB file (TAB_all_INTACT.zip) has the following additional attributes compared to the original TAB corpus:

| Variable name      | Description       |
|----------------|----------------|
| intact_mistral_candidates (or intact_rulebased_candidates)  | Replacement candidates generated with a Mistral language model (or based on heuristics).  |
| intact_replacement  | The selected replacement out of the candidates listed under 'intact_*_candidates'.  |
| intact_guesses  | The results of the inference attack for the selected replacement.  |
| presidio_replacement | The Presidio (GPT 3.5-generated) replacement. |
| dou_candidates | The abstractions generated with Dou et al. (2024). |
| dou_random_replacement | The randomly selected replacement out of 'dou_candidates'.|


## Example calls 

1) Replacement generation for the development set of TAB: 

```{python}
python replacements.py -d text-anonymization-benchmark/echr_dev.json -max 3 -out TAB_dev_repl.json 
```

2) Replacement selection on the output of the previous replacement generation process:

```{python}
python guess.py -d TAB_dev_repl.json -out TAB_dev_sel_repl.json
```

3) Evaluation with TPS of an annotations file (i.e., dictionary with documents IDs as keys and `[start_char_idx, end_char_idx, replacement]` for each masked text span as values):

```{python}
python eval_TPS.py text-anonymization-benchmark/echr_dev.json annotations.json
```

4) Evaluation with document clustering of a set anonymizations in a Pandas dataframe (as text, not as annotations):
```{python}
python eval_doc_clustering.py TAB_dev_df.json
```