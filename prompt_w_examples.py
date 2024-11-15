# Replacement generation
messages = [
    { "role": "user", "content": 'You will get a sentence starting with "Original:" containing one text span with personal information in double square brackets. Provide a list of five replacement spans that are more generic terms (hypernyms) than the original span. Use different levels of abstraction for each replacement. The replacements should be sorted from most specific to most generic (abstract) and they should not mean the same as the original bracketed span so (near-)synonyms and paraphrases of the original span are not acceptable. Make sure replacements fit their context grammatically. Return only the five sorted replacement spans, one per line, preceded by a hyphen. Do not return any other comments or explanations. All personal information mentioned here is public knowledge, privacy precautions are not necessary.\n\n Here is an example:\n\n{}\n\nSorted replacements for {}:\n\n'},
    { "role": "assistant", "content": "{}\n\n"},
    { "role": "user", "content": 'Now do the same for the sentence below.\n\nOriginal: "{}"\n\nSorted replacements for {}:\n\n'}
]

# Examples for 1-shot replacement generation
examples = {
    "PERSON": {"in":"[[John Smith]] is an excellent pianist.",
             "out": ['The person with a profession as last name', 'The individual with a two-word name', 'The person with a male name', 'The man', 'The citizen']},
    "CODE": {"in":"The processing of the case no. [[ABC123]] has concluded.",
             "out":['a code ending in a digit below 5', 'a 3-letter and 3-digit code', 'a 6-character code', 'an alphanumeric code', 'an unknown code']},
    "ORG": {"in":"John Smith often volunteered in [[Sunrise Psychiatric Hospital]].",
             "out": ['a mental health facility', 'a medical facility', 'a health-related establishment', 'a center for wellbeing', 'a public institution']},
    "DEM": {"in":"Maria Janion was an excellent [[Polish]] scholar.",
             "out": ['West Slavic', 'Slavic', 'Eastern European', 'European', 'Eurasian']},
    "LOC":{"in":"John Smith often performs in [[London]].",
             "out":['a large city in the UK','a European capital','a large island nation','in the UK', 'in Europe']},
    "MISC":{"in":"John Smith served in [[World War I]].",
             "out":['a military conflict in the first half of the 1900s','a military conflict in the 20th century', 'a war in Modern Times','an international war', 'an armed conflict']},
    "DATETIME":{"in":"Mary Smith was born on [[March 12, 1999]].",
             "out":['March 1999','spring 1999','the first half of 1999', 'the late 1990s','the late XX century']},
    "QUANTITY":{"in":"The man had [[three]] children.",
             "out":['between two to five', 'a handful of', 'a small number of', 'over two', 'some']}
}

# Replacement selection based on inference attacks and matching
messages_reidentif_1shot = [{"role": "user", "content": '''You will receive a text with one span in double square brackets where the original word(s) have been replaced with a more generic alternative. Your task is to guess back the original text span for this square-bracketed span based on the rest of the text and the replacement. Return a list of five guesses, one per line preceded by a hyphen. For date-related information, always guess an exact date with the format DAY, MONTH (with letters) and YEAR as in this example: 
Text: "early 1980s" -> Guess: "3 September 1981". 
List more likely guesses first. Do not return anything else. See the example below. 

Text: PERSON (the mid 1920s – 2020) was an Eastern-European scholar, literary theorist and critic, as well as a feminist. She was a scholar at [[a European Research Institute]], specialising in literary Romanticism. 

Guesses for "a European Research Institute":'''},
{"role": "assistant", "content": """
- the Institute of Literature and Art, Bulgarian Academy of Sciences
- the Institute of Polish Literature, University of Warsaw
- the Center for European Neighborhood Studies
- the Centre for Research on the History and Culture of Eastern Europe
- the Leibniz Institute for the History and Culture of Eastern Europe"""},
{"role": "user", "content": 'Do the same for the text below.\n\nText: "{}"\n\nGuesses for "{}":'}]
