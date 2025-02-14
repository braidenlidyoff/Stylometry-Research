
'''This program is meant to analyze everything from p.39 and on, of the Old English Corpus
It stores values like WorkID, CollectionID, Title, and Text in a DataFrame
The dataframe on output is truncated, and can be fixed to show the entire body of text'''

import pandas as pd
import re
import pdfplumber

# This will allow the entire body of text to be displayed
# pd.set_option('display.max_colwidth', None)

def data_extract(pdf_path, start_page, end_page):
    data = []
    current_work = None
    current_collection = None
    current_title = None
    current_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in range(start_page - 1, end_page):
            page = pdf.pages[page_num]
            text = page.extract_text() or ""  # Ensures we always get a string
            lines = text.split("\n")  # Splits text into lines

            for line in lines:
                line = line.strip()  # Clean up whitespace

                # Skip unwanted lines
                if re.match(r'^\d{4}\. Bounds, Sawyer \d+ \(DOE Electronic edition\)', line): 
                    continue  # Skip lines like "1256. Bounds, Sawyer 635 (DOE Electronic edition)"

                if line in ["THE OLD ENGLISH CORPUS", "A: Texts"]:  # Skip section headers
                    continue

                # Detect WorkID (e.g., "A1.1: GenA,B")
                if re.match(r'^[A-Z]\d+(\.\d+)?:\s*\S+', line):
                    if current_work and current_text:  # Save previous work before starting new one
                        data.append([current_work, current_collection, current_title, " ".join(current_text)])
                    
                    current_work = line.strip()
                    current_collection = re.match(r'^([A-Z]\d+)', current_work).group(1)
                    current_text = []  # Reset text collection
                    continue

                # Extract Title
                if line.startswith("Title:"):
                    current_title = line.replace("Title:", "").strip().split(":")[0]
                    continue

                # Skip citations
                if line.startswith("Citation:"):
                    continue

                # Skip the numbers, but keep the content
                line = re.sub(r'^\d+:\s*', '', line)  # Remove line numbers
                current_text.append(line)

        # Save last collected work
        if current_work and current_text:
            data.append([current_work, current_collection, current_title, " ".join(current_text)])

    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=["WorkID", "CollectionID", "Title", "Text"])
    print(df)
    return df

# Path & Function Call
pdf_path = "/Users/braidenlidyoff/Desktop/Stylometry Research/The Old English Corpus.pdf"
df = data_extract(pdf_path, start_page=39, end_page=200)
