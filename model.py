"""
RAG Pipeline

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_text_file
def load_text_file(path):
    # TODO: read a UTF-8 text file at `path` and return its contents as one string.
    text = ""
    with open(path, "r") as f:
        text += f.read()

    return text

# Step 2 - load_text_directory
def load_text_directory(directory):
    # TODO: read every .txt file in `directory` and return their contents as a list of strings
    contents = []
    for fn in sorted(os.listdir(directory)):
        if fn.endswith('.txt'):
            contents.append(load_text_file(os.path.join(directory, fn)))

    return contents

