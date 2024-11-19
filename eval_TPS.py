import json, re, abc, argparse, math, ntpath
from datetime import datetime
import numpy as np
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from tqdm.autonotebook import tqdm
import spacy
import intervaltree
from sentence_transformers import SentenceTransformer


# POS tags, tokens or characters that can be ignored from the recall scores 
# (because they do not carry much semantic content, and there are discrepancies
# on whether to include them in the annotated spans or not)
POS_TO_IGNORE = {"ADP", "PART", "CCONJ", "DET"} 
TOKENS_TO_IGNORE = {"mr", "mrs", "ms", "no", "nr", "about"}
CHARACTERS_TO_IGNORE = " ,.-;:/&()[]–'\" ’“”"

# Path to the results file
RESULTS_FILENAME = "results.csv"

@dataclass
class MaskedDocument:
    """Represents a document in which some text spans are masked, each span
    being expressed by their (start, end) character boundaries"""

    doc_id: str
    masked_spans : List[Tuple[int, int]]
    replacements : List[str]

    def get_masked_offsets(self):
        """Returns the character offsets that are masked"""
        if not hasattr(self, "masked_offsets"):
            self.masked_offsets = {i for start, end in self.masked_spans
                                   for i in range(start, end)}
        return self.masked_offsets


class TokenWeighting:
    """Abstract class for token weighting schemes (used to compute the precision)"""

    @abc.abstractmethod
    def get_weights(self, text:str, text_spans:List[Tuple[int,int]]):
        """Given a text and a list of text spans, returns a list of numeric weights
        (of same length as the list of spans) representing the information content
        conveyed by each span.

        A weight close to 0 represents a span with low information content (i.e. which
        can be easily predicted from the remaining context), while a weight close to 1
        represents a high information content (which is difficult to predict from the
        context)"""

        return

@dataclass
class AnnotatedEntity:
    """Represents an entity annotated in a document, with a unique identifier,
    a list of mentions (character-level spans in the document), whether it
    needs to be masked, and whether it corresponds to a direct identifier"""

    entity_id: str
    mentions: List[Tuple[int, int]]
    need_masking: bool
    is_direct: bool
    entity_type: str
    mention_level_masking: List[bool]

    def __post_init__(self):
        if self.is_direct and not self.need_masking:
            raise RuntimeError("Direct identifiers must always be masked")

    @property
    def mentions_to_mask(self):
        return [mention for i, mention in enumerate(self.mentions)
                if self.mention_level_masking[i]]


class GoldCorpus:
    """Representation of a gold standard corpus for text anonymisation, extracted from a
    JSON file. See annotation guidelines for the TAB corpus for details. """
    
    def __init__(self, gold_standard_json_file:str, spacy_model = "en_core_web_md"):
        
        # Loading the spacy model
        self.nlp = spacy.load(spacy_model, disable=["lemmatizer"])
        
        # documents indexed by identifier
        self.documents = {}
        
        # Train/dev/test splits
        self.splits = {}

        fd = open(gold_standard_json_file, encoding="utf-8")
        annotated_docs = json.load(fd)
        fd.close()
        print("Reading annotated corpus with %i documents"%len(annotated_docs))
        
        if type(annotated_docs)!=list:
            raise RuntimeError("JSON file should be a list of annotated documents")

        for ann_doc in tqdm(annotated_docs):
            for key in ["doc_id", "text", "annotations", "dataset_type"]:
                if key not in ann_doc:
                    raise RuntimeError("Annotated document is not well formed: missing variable %s"%key)
            
            # Parsing the document with spacy
            spacy_doc = self.nlp(ann_doc["text"])
            
            # Creating the actual document (identifier, text and annotations)          
            new_doc = GoldDocument(ann_doc["doc_id"], ann_doc["text"], 
                                   ann_doc["annotations"], spacy_doc)
            self.documents[ann_doc["doc_id"]] = new_doc
            
            # Adding it to the list for the specified split (train, dev or test)
            data_split = ann_doc["dataset_type"]
            self.splits[data_split] = self.splits.get(data_split, []) + [ann_doc["doc_id"]]     
    
    
    def get_TPS(self, masked_docs:List[MaskedDocument], token_weighting: TokenWeighting,
                word_alterning=6, sim_model_name="paraphrase-albert-base-v2", use_chunking=True):
        tps_array = np.empty(len(masked_docs))
        
        # Load embedding model and function for similarity
        embedding_func = self._get_embedding_func(sim_model_name)
        
        # Process each masked document
        for i, masked_doc in enumerate(tqdm(masked_docs)):
            gold_doc = self.documents[masked_doc.doc_id]

            # Get text spans
            spans = self._get_terms_spans(gold_doc.spacy_doc, use_chunking=use_chunking)

            # Get IC for all spans
            spans_IC = self._get_ICs(spans, gold_doc, token_weighting, word_alterning)

            # Get replacements, corresponding masked texts and corresponding spans indexes            
            repl_out = self._get_replacements_info(masked_doc, gold_doc, spans)
            (replacements, masked_texts, spans_idxs_per_replacement) = repl_out

            # Measure similarities of replacements
            masked_spans = self._filter_masked_spans(gold_doc, masked_doc)
            spans_mask = self._get_spans_mask(spans, gold_doc, masked_spans) # Non-masked=True(1), Masked=False(0)
            spans_sims = np.array(spans_mask, dtype=float) # Similarities for terms: Non-masked=1, Supressed=0, Replaced=[0,1]
            if len(replacements) > 0:
                texts_to_embed = masked_texts + replacements
                embeddings = embedding_func(texts_to_embed)
      
                masked_embedds = embeddings[:len(masked_texts)]
                repl_embedds = embeddings[len(masked_texts):]
                for masked_embed, repl_embed, spans_idxs in zip(masked_embedds, repl_embedds, spans_idxs_per_replacement):
                    spans_sims[spans_idxs] = self._cos_sim(masked_embed, repl_embed)
                
                # Limit similarities to range [0,1]
                spans_sims[spans_sims < 0] = 0
                spans_sims[spans_sims > 1] = 1

            # Get TPS
            masked_TPI = (spans_IC * spans_sims).sum()
            original_TPI = spans_IC.sum()
            tps_array[i] = masked_TPI / original_TPI

        # Get mean TPS
        tps = tps_array.mean() 

        return tps, tps_array    


    def _get_terms_spans(self, spacy_doc: spacy.tokens.Doc, use_chunking: bool=True) -> list:
        text_spans = []
        added_tokens = np.zeros(len(spacy_doc), dtype=bool)

        if use_chunking:
            for chunk in spacy_doc.ents:
                start = spacy_doc[chunk.start].idx
                last_token = spacy_doc[chunk.end - 1]
                end = last_token.idx + len(last_token)
                text_spans.append((start, end))
                added_tokens[chunk.start:chunk.end] = True

            for chunk in spacy_doc.noun_chunks:
                # If is it not already added
                if not added_tokens[chunk.start:chunk.end].any():
                    start = spacy_doc[chunk.start].idx
                    last_token = spacy_doc[chunk.end - 1]
                    end = last_token.idx + len(last_token)
                    text_spans.append((start, end))
                    added_tokens[chunk.start:chunk.end] = True
                

        # Add text spans after chunks (or all spans, if chunks are ignored)
        for token_idx in range(len(spacy_doc)):
            if not added_tokens[token_idx]:
                token = spacy_doc[token_idx]            
                if token.text.strip() not in ["", "\n"]:  # Avoiding empty spans
                    start = token.idx
                    end = start + len(token)
                    text_spans.append((start, end))

        # Sort text spans by starting position
        text_spans = sorted(text_spans, key=lambda span: span[0], reverse=False)

        return text_spans


    def _filter_masked_spans(self, gold_doc, masked_doc: MaskedDocument) -> list:
        filtered_masked_spans = []

        masking_array = np.zeros(len(gold_doc.spacy_doc.text), dtype=bool)
        for (s, e) in masked_doc.masked_spans:
            masking_array[s:e] = True
        
        ini_current_mask = -1
        for idx, elem in enumerate(masking_array):
            # Start of mask
            if ini_current_mask == -1 and elem:
                ini_current_mask = idx
            # End of mask
            elif ini_current_mask >= 0 and not elem:
                filtered_masked_spans.append((ini_current_mask, idx))
                ini_current_mask = -1
        
        return filtered_masked_spans


    def _get_spans_mask(self, spans: List[Tuple[int, int]], gold_doc, masked_spans: list) -> np.array:
        spans_mask = np.empty(len(spans), dtype=bool)
        sorted_masked_spans = sorted(masked_spans, key=lambda span: span[0], reverse=False)

        for i, (span_start, span_end) in enumerate(spans):
            # True(1)=Non-masked, False(0)=Masked
            spans_mask[i] = True
            for (masked_span_start, masked_span_end) in sorted_masked_spans:
                if span_start <= masked_span_start < span_end or span_start < masked_span_end <= span_end:
                    spans_mask[i] = False
                elif masked_span_start > span_end: # Break if masked span starts too late
                    break

        return spans_mask


    def _get_ICs(self, spans: List[Tuple[int, int]], gold_doc, token_weighting: TokenWeighting, word_alterning) -> np.array:
        spans_IC = np.empty(len(spans))
        if isinstance(word_alterning, int) and word_alterning > 1: # N Word Alterning
            # Get ICs by masking each N words, with all the document as context
            for i in range(word_alterning):
                spans_for_IC = spans[i::word_alterning]
                spans_IC[i::word_alterning] = self._get_spans_ICs(spans_for_IC, gold_doc, token_weighting)
        
        elif isinstance(word_alterning, str) and word_alterning == "sentence": # Sentence Word Alterning
            # Get masks by masking 1 word of each sentence, with the sentence as context
            # Get sentences spans
            sentences_spans = [[sent.start_char, sent.end_char] for sent in gold_doc.spacy_doc.sents]
            # Iterate sentences
            ini_span_idx = 0
            for sentence_span in sentences_spans:
                sentence_start, sentence_end = sentence_span
                # Get spans in the sentence
                span_idx = ini_span_idx
                first_sentence_span_idx = -1
                is_sentence_complete = False
                while span_idx < len(spans) and not is_sentence_complete:
                    # If span belongs to sentence (first spans may not belong to any sentence)
                    if spans[span_idx][0] >= sentence_start and spans[span_idx][1] < sentence_end:
                        if first_sentence_span_idx == -1:  # If first sentence span
                            first_sentence_span_idx = span_idx  # Store first index
                        span_idx += 1  # Go to next span
                    # If not belongs and sentence is started, sentence completed
                    elif first_sentence_span_idx != -1:
                        is_sentence_complete = True
                    # Otherwise, go to next span
                    else:
                        span_idx += 1
                spans_for_IC = spans[first_sentence_span_idx:span_idx]
                # Update initial span index for sentece's spans searching
                ini_span_idx = span_idx
                # Get IC for each span of the sentence
                for span in spans_for_IC:
                    original_info, masked_info, n_masked_terms = self._get_spans_ICs(gold_doc, [span], gold_doc,
                                                                                     token_weighting, context_span=sentence_span)
                    original_doc_info += original_info
                    masked_doc_info += masked_info
                    total_n_masked_terms += n_masked_terms
        else:
            raise Exception(f"Word alterning setting [{word_alterning}] is invalid")

        return spans_IC

    
    def _get_spans_ICs(self, spans: List[Tuple[int, int]], gold_doc, token_weighting: TokenWeighting, context_span=None) -> np.array:
        # By default, context span is all the document
        if context_span is None:
            context_span = (0, len(gold_doc.text))

        # Get context
        context_start, context_end = context_span
        context = gold_doc.text[context_start:context_end]

        # Adjust spans to the context
        in_context_spans = []
        for (start, end) in spans:
            in_context_spans.append((start - context_start, end - context_start))

        # Obtain the weights (Information Content) of each word
        ICs = token_weighting.get_weights(context, in_context_spans)
        ICs = np.array(ICs) # Transform to numpy

        return ICs
    
    
    def _get_embedding_func(self, sim_model_name:str):
        if sim_model_name is None: # Default spaCy model
            embedding_func = lambda x: np.array([self.nlp(text).vector for text in x])
        else:   # Sentence Transformer
            sim_model = SentenceTransformer(sim_model_name, trust_remote_code=True)
            embedding_func = lambda x : sim_model.encode(x)
        
        return embedding_func

    
    def _get_replacements_info(self, masked_doc: MaskedDocument, gold_doc, spans: list):
        replacements = []
        masked_texts = []
        spans_idxs_per_replacement = []
        
        for replacement, (masked_span_start, masked_span_end) in zip(masked_doc.replacements, masked_doc.masked_spans):
            if replacement is not None: # If there is a replacement
                replacements.append(replacement)
                masked_texts.append(gold_doc.text[masked_span_start:masked_span_end])
                replacement_spans_idxs = []
                for span_idx, (span_start, span_end) in enumerate(spans):
                    if span_start <= masked_span_start < span_end or span_start < masked_span_end <= span_end:
                        replacement_spans_idxs.append(span_idx)
                    elif span_start > masked_span_end:  # Break if candidate span starts too late
                        break
                spans_idxs_per_replacement.append(replacement_spans_idxs)
        
        return replacements, masked_texts, spans_idxs_per_replacement

    
    def _cos_sim(self, a:np.array, b:np.array) -> float:
        dot_product = np.dot(a, b)
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)
        sim = dot_product / (magnitude_a * magnitude_b)
        if np.isnan(sim):
            sim = 0
        return sim


class GoldDocument:
    """Representation of an annotated document"""
    
    def __init__(self, doc_id:str, text:str, annotations:Dict[str,List],
                 spacy_doc: spacy.tokens.Doc):
        """Creates a new annotated document with an identifier, a text content, and 
        a set of annotations (see guidelines)"""
        
        # The (unique) document identifier, its text and the spacy document
        self.doc_id = doc_id
        self.text = text
        self.spacy_doc = spacy_doc
        
        # Annotated entities (indexed by id)
        self.entities = {}        
        
        for annotator, ann_by_person in annotations.items():
            
            if "entity_mentions" not in ann_by_person:
                raise RuntimeError("Annotations must include entity_mentions")
            
            for entity in self._get_entities_from_mentions(ann_by_person["entity_mentions"]):
                
                # We require each entity_id to be specific for each annotator
                if entity.entity_id in self.entities:
                    raise RuntimeError("Entity ID %s already used by another annotator"%entity.entity_id)
                    
                entity.annotator = annotator
                entity.doc_id = doc_id
                self.entities[entity.entity_id] = entity
            
                    
    def _get_entities_from_mentions(self, entity_mentions):
        """Returns a set of entities based on the annotated mentions"""
        
        entities = {}
        
        for mention in entity_mentions:
                
            for key in ["entity_id", "identifier_type", "start_offset", "end_offset"]:
                if key not in mention:
                    raise RuntimeError("Unspecified key in entity mention: " + key)
                                   
            entity_id = mention["entity_id"]
            start = mention["start_offset"]
            end = mention["end_offset"]
                
            if start < 0 or end > len(self.text) or start >= end:
                raise RuntimeError("Invalid character offsets: [%i-%i]"%(start, end))
                
            if mention["identifier_type"] not in ["DIRECT", "QUASI", "NO_MASK"]:
                raise RuntimeError("Unspecified or invalid identifier type: %s"%(mention["identifier_type"]))

            need_masking = mention["identifier_type"] in ["DIRECT", "QUASI"]
            is_direct = mention["identifier_type"]=="DIRECT"
            
                
            # We check whether the entity is already defined
            if entity_id in entities:
                    
                # If yes, we simply add a new mention
                current_entity = entities[entity_id]
                current_entity.mentions.append((start, end))
                current_entity.mention_level_masking.append(need_masking)
                    
            # Otherwise, we create a new entity with one single mention
            else:
                new_entity = AnnotatedEntity(entity_id, [(start, end)], need_masking, is_direct, 
                                             mention["entity_type"], [need_masking])
                entities[entity_id] = new_entity
                
        for entity in entities.values():
            if set(entity.mention_level_masking) != {entity.need_masking}:
                entity.need_masking = True
                #print("Warning: inconsistent masking of entity %s: %s"
                #%(entity.entity_id, str(entity.mention_level_masking)))
                
        return list(entities.values())
    
    
    def is_masked(self, masked_doc:MaskedDocument, entity: AnnotatedEntity):
        """Given a document with a set of masked text spans, determines whether entity
        is fully masked (which means that all its mentions are masked)"""
        
        for incr, (mention_start, mention_end) in enumerate(entity.mentions):
            
            if self.is_mention_masked(masked_doc, mention_start, mention_end):
                continue
            
            # The masking is sometimes inconsistent for the same entity, 
            # so we verify that the mention does need masking
            elif entity.mention_level_masking[incr]:
                return False
        return True
    

    def is_mention_masked(self, masked_doc:MaskedDocument, mention_start:int, mention_end:int):
        """Given a document with a set of masked text spans and a particular mention span,
        determine whether the mention is fully masked (taking into account special
        characters or tokens to skip)"""
        
        mention_to_mask = self.text[mention_start:mention_end].lower()
                
        # Computes the character offsets that must be masked
        offsets_to_mask = set(range(mention_start, mention_end))

        # We build the set of character offsets that are not covered
        non_covered_offsets = offsets_to_mask - masked_doc.get_masked_offsets()
            
        # If we have not covered everything, we also make sure punctuations
        # spaces, titles, etc. are ignored
        if len(non_covered_offsets) > 0:
            span = self.spacy_doc.char_span(mention_start, mention_end, alignment_mode = "expand")
            for token in span:
                if token.pos_ in POS_TO_IGNORE or token.lower_ in TOKENS_TO_IGNORE:
                    non_covered_offsets -= set(range(token.idx, token.idx+len(token)))
        for i in list(non_covered_offsets):
            if self.text[i] in set(CHARACTERS_TO_IGNORE):
                non_covered_offsets.remove(i)

        # If that set is empty, we consider the mention as properly masked
        return len(non_covered_offsets) == 0


    def get_entities_to_mask(self,  include_direct=True, include_quasi=True):
        """Return entities that should be masked, and satisfy the constraints 
        specified as arguments"""
        
        to_mask = []
        for entity in self.entities.values():     
             
            # We only consider entities that need masking and are the right type
            if not entity.need_masking:
                continue
            elif entity.is_direct and not include_direct:
                continue
            elif not entity.is_direct and not include_quasi:
                continue  
            to_mask.append(entity)
                
        return to_mask      
    
    
    def get_annotators_for_span(self, start_token: int, end_token: int):
        """Given a text span (typically for a token), determines which annotators 
        have also decided to mask it. Concretely, the method returns a (possibly
        empty) set of annotators names that have masked that span."""
        
        
        # We compute an interval tree for fast retrieval
        if not hasattr(self, "masked_spans"):
            self.masked_spans = intervaltree.IntervalTree()
            for entity in self.entities.values():
                if entity.need_masking:
                    for i, (start, end) in enumerate(entity.mentions):
                        if entity.mention_level_masking[i]:
                            self.masked_spans[start:end] = entity.annotator
        
        annotators = set()      
        for mention_start, mention_end, annotator in self.masked_spans[start_token:end_token]:
            
            # We require that the span is fully covered by the annotator
            if mention_start <=start_token and mention_end >= end_token:
                annotators.add(annotator)
                    
        return annotators
                                                
    
    def split_by_tokens(self, start: int, end: int):
        """Generates the (start, end) boundaries of each token included in this span"""
        
        for match in re.finditer("\w+", self.text[start:end]):
            start_token = start + match.start(0)
            end_token = start + match.end(0)
            yield start_token, end_token

    
class BertTokenWeighting(TokenWeighting):
    """Token weighting based on a BERT language model. The weighting mechanism
    runs the BERT model on a text in which the provided spans are masked. The
    weight of each token is then defined as 1-(probability of the actual token value).
    
    In other words, a token that is difficult to predict will have a high
    information content, and therefore a high weight, whereas a token which can
    be predicted from its content will received a low weight. """
    
    def __init__(self, max_segment_size = 100):
        """Initialises the BERT tokenizers and masked language model"""
        
        from transformers import BertTokenizerFast, BertForMaskedLM
        self.tokeniser = BertTokenizerFast.from_pretrained('bert-base-uncased')

        import torch
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.model = self.model.to(self.device)
        
        self.max_segment_size = max_segment_size
        
        
    def get_weights(self, text:str, text_spans:List[Tuple[int,int]]):
        """Returns a list of numeric weights between 0 and 1, where each value
        corresponds to 1 - (probability of predicting the value of the text span
        according to the BERT model). 
        
        If the span corresponds to several BERT tokens, the probability is the 
        product of the probabilities for each token."""
        
        import torch
        
        # STEP 1: we tokenise the text
        bert_tokens = self.tokeniser(text, return_offsets_mapping=True)
        input_ids = bert_tokens["input_ids"]
        input_ids_copy = np.array(input_ids)
        
        # STEP 2: we record the mapping between spans and BERT tokens
        bert_token_spans = bert_tokens["offset_mapping"]
        tokens_by_span = self._get_tokens_by_span(bert_token_spans, text_spans, text)

        # STEP 3: we mask the tokens that we wish to predict
        attention_mask = bert_tokens["attention_mask"]
        for token_indices in tokens_by_span.values():
            for token_idx in token_indices:
                attention_mask[token_idx] = 0
                input_ids[token_idx] = self.tokeniser.mask_token_id
          
        # STEP 4: we run the masked language model     
        logits = self._get_model_predictions(input_ids, attention_mask)
        unnorm_probs = torch.exp(logits)
        probs = unnorm_probs / torch.sum(unnorm_probs, axis=1)[:,None]
        
        # We are only interested in the probs for the actual token values
        probs_actual = probs[torch.arange(len(input_ids)), input_ids_copy]
        probs_actual = probs_actual.detach().cpu().numpy()
              
        # STEP 5: we compute the weights from those predictions
        weights = []
        for (span_start, span_end) in text_spans:
            
            # If the span does not include any actual token, skip
            if not tokens_by_span[(span_start, span_end)]:
                weights.append(0)
                continue
            
            # if the span has several tokens, we take the minimum prob
            prob = np.min([probs_actual[token_idx] for token_idx in 
                           tokens_by_span[(span_start, span_end)]])
            
            # We finally define the weight as -log(p)
            weights.append(-np.log(prob))
        
        return weights


    def _get_tokens_by_span(self, bert_token_spans, text_spans, text:str):
        """Given two lists of spans (one with the spans of the BERT tokens, and one with
        the text spans to weight), returns a dictionary where each text span is associated
        with the indices of the BERT tokens it corresponds to."""            
        
        # We create an interval tree to facilitate the mapping
        text_spans_tree = intervaltree.IntervalTree()
        for start, end in text_spans:
            text_spans_tree[start:end] = True
        
        # We create the actual mapping between spans and tokens
        tokens_by_span = {span:[] for span in text_spans}
        for token_idx, (start, end) in enumerate(bert_token_spans):
            for span_start, span_end, _ in text_spans_tree[start:end]:
                tokens_by_span[(span_start, span_end)].append(token_idx) 
        
        # And control that everything is correct
        for span_start, span_end in text_spans:
            if len(tokens_by_span[(span_start, span_end)]) == 0:
                print(f"Warning: span ({span_start},{span_end}) without any token [{repr(text[span_start:span_end])}]")
        
        return tokens_by_span
    
    
    def _get_model_predictions(self, input_ids, attention_mask):
        """Given tokenised input identifiers and an associated attention mask (where the
        tokens to predict have a mask value set to 0), runs the BERT language and returns
        the (unnormalised) prediction scores for each token.
        
        If the input length is longer than max_segment size, we split the document in
        small segments, and then concatenate the model predictions for each segment."""
        
        import torch
        nb_tokens = len(input_ids)
        
        input_ids = torch.tensor(input_ids)[None,:].to(self.device)
        attention_mask = torch.tensor(attention_mask)[None,:].to(self.device)
        
        # If the number of tokens is too large, we split in segments
        if nb_tokens > self.max_segment_size:
            nb_segments = math.ceil(nb_tokens/self.max_segment_size)
            
            # Split the input_ids (and add padding if necessary)
            split_pos = [self.max_segment_size * (i + 1) for i in range(nb_segments - 1)]
            input_ids_splits = torch.tensor_split(input_ids[0], split_pos)

            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_splits, batch_first=True)
            
            # Split the attention masks
            attention_mask_splits = torch.tensor_split(attention_mask[0], split_pos)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_splits, batch_first=True)
                   
        # Run the model on the tokenised inputs + attention mask
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # And get the resulting prediction scores
        scores = outputs.logits
        
        # If the batch contains several segments, concatenate the result
        if len(scores) > 1:
            scores = torch.vstack([scores[i] for i in range(len(scores))])
            scores = scores[:nb_tokens]
        else:
            scores = scores[0]
        
        return scores     


def get_masked_docs_from_file(masked_output_file:str):
    """Given a file path for a JSON file containing the spans to be masked for
    each document, returns a list of MaskedDocument objects"""
    
    fd = open(masked_output_file, "r", encoding="utf-8")
    masked_output_docs = json.load(fd)
    fd.close()
    
    if type(masked_output_docs)!= dict:
        raise RuntimeError("%s must contain a mapping between document identifiers"%masked_output_file
                           + " and lists of masked spans in this document")
    
    masked_docs = []
    for doc_id, masked_spans in masked_output_docs.items():
        doc = MaskedDocument(doc_id, [], [])
        if type(masked_spans)!=list:
            raise RuntimeError("Masked spans for the document must be a list of (start, end) tuples")
        
        for elems in masked_spans:
            # Store span
            start = elems[0]
            end = elems[1]
            doc.masked_spans.append((start, end))

            # Store replacement (None if non-existent or it's an empty string)
            replacement = None if len(elems) < 3 or elems[2] == "" else elems[2]
            doc.replacements.append(replacement)
            
        masked_docs.append(doc)
        
    return masked_docs
    

if __name__ == "__main__":
    # Arguments parsing
    parser = argparse.ArgumentParser(description='Computes evaluation metrics for text sanitization')
    parser.add_argument('gold_standard_file', type=str, 
                        help='the path to the JSON file containing the gold standard annotations')
    parser.add_argument('masked_output_file', type=str, nargs="+",
                        help='the path to the JSON file containing the actual spans masked by the system')
    args = parser.parse_args()

    gold_corpus = GoldCorpus(args.gold_standard_file)
    
    for masked_output_file in args.masked_output_file:
        print("=========")
        masked_docs = get_masked_docs_from_file(masked_output_file)
        
        for masked_doc in masked_docs:
            if masked_doc.doc_id not in gold_corpus.documents:
                raise RuntimeError("Document %s not present in gold corpus"%masked_doc.doc_id)
        
        # Weighting scheme
        weighting_scheme = BertTokenWeighting()
        
        # Metrics settings        
        sim_model_name = "paraphrase-albert-base-v2"
        word_alterning = 6
        title = f"TPS-({sim_model_name}|{ntpath.basename(masked_output_file)})"

        # Compute TPS
        print("Computing evaluation metrics for", masked_output_file, "(%i documents)"%len(masked_docs))
        tps, tps_array = gold_corpus.get_TPS(masked_docs, weighting_scheme, word_alterning=word_alterning, sim_model_name=sim_model_name)           
        tps_std = tps_array.std()

        # Print results
        print(f"==> {title}: {tps:3f}±{tps_std:3f}")

        # Results to file
        datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(RESULTS_FILENAME, "a+") as f:
            f.write(f"{datetime_str},{title},{tps},{tps_std}\n")