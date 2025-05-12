# This code is not complete and required some manual work after extracting. It can be improved a lot, but adding it as a reference for extracting future documents.
from docx import Document
import re
import csv
import pandas as pd

# Load the .docx file
doc_path = '../Kleist, Chronology and Canon 449a.docx'

doc = Document(doc_path)

def extract_all_corpus_ids(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Remove .txt from each line and strip whitespace
    all_corpus_ids = {line.strip().replace('.txt', '') for line in lines if line.strip()}
    return all_corpus_ids

# Storage for extracted entries
entries = []
corpus_ids_in_chronology = set()

# Regex pattern to match Chronology entries
entry_pattern = re.compile(r'†?(\d+(?:\.\d+)*)(?:\.?)\s*(?:\[(.*?)\])?\s*(.*?)(\*)?(?:\(|$)')

# Go through paragraphs in the document
for para in doc.paragraphs:
    text = para.text.strip()
    if not text:
        continue
    
    match = entry_pattern.match(text)
    if match:
        chronology_id = match.group(1)
        corpus_id = match.group(2) or ''
        title = match.group(3)
        has_asterisk = match.group(4)

        authorship = ''
        if has_asterisk:
            authorship = 2
        else:
            authorship = 1

        if corpus_id:
            new_corpus_id = corpus_id.replace(".", "", 1)
            corpus_ids_in_chronology.add(new_corpus_id.strip())

        entries.append({
            'ChronologyID': chronology_id,
            'CorpusID': corpus_id.strip(),
            'Authorship': authorship
        })

# All corpus IDs from the Corpus Document
file_path = './corpusID.txt'
all_corpus_ids = extract_all_corpus_ids(file_path)

# Find corpus IDs that were never referenced
unreferenced_corpus = all_corpus_ids - corpus_ids_in_chronology

for corpus_id in unreferenced_corpus:
    entries.append({
        'ChronologyID': '',
        'CorpusID': corpus_id,
        'Authorship': 3
    })

# Write to Excel file
output_path = './chronology_extracted_entries.xlsx'
df = pd.DataFrame(entries)
df.to_excel(output_path, index=False)

print(f"Extraction complete! File saved to {output_path}")
