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

# Step 3 - extract_text_from_html
from html.parser import HTMLParser


class Parser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_blocks = []
        self.skip_tags = {"script", "head", "style", "title", "meta", "[document]"}
        self.current_tag = None


    def handle_starttag(self, tag, attrs):
        self.current_tag = tag.lower()


    def handle_endtag(self, tag):
        self.current_tag = None


    def handle_data(self, data):
        if self.current_tag not in self.skip_tags:
            self.text_blocks.append(data)


    def get_text(self):
        return "".join(self.text_blocks)


def extract_text_from_html(html):
    # TODO: strip HTML tags and return only the visible text content
    parser = Parser()
    parser.feed(html)
    return parser.get_text()

# Step 4 - normalize_text
import unicodedata


def normalize_text(text):
    # TODO: NFKC-normalize the text and collapse runs of whitespace into single spaces.
    return " ".join(unicodedata.normalize("NFKC", text).split())

# Step 5 - make_document
def make_document(text, source, title):
    # TODO: wrap text with source and title metadata into a document dict.
    return dict(
        text=text,
        source=source,
        title=title
    )

# Step 6 - chunk_fixed_size
def chunk_fixed_size(text, chunk_size):
    # TODO: split text into consecutive non-overlapping chunks of length chunk_size
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks

# Step 7 - chunk_by_tokens
def chunk_by_tokens(text, tokenizer, max_tokens):
    # TODO: split text into chunks of at most max_tokens token ids using the tokenizer
    token_ids = tokenizer.encode(text)
    token_chunks = []
    for i in range(0, len(token_ids), max_tokens):
        token_chunks.append(token_ids[i:i+max_tokens])

    return [tokenizer.decode(chunk) for chunk in token_chunks]

# Step 8 - chunk_by_sentences
import re


def chunk_by_sentences(text, max_chars):
    # TODO: split text on .!? boundaries and greedily pack whole sentences under max_chars.
    sents = re.split(r"(?<=[.!?])\s+", text)
    sent_chunks = []
    if not sents:
        return sent_chunks

    i = 1
    chunk = sents[0].strip()
    while i < len(sents):
        if len(chunk) + len(sents[i]) + 1 <= max_chars:
            chunk += " " + sents[i]
            chunk = chunk.strip()
        else:
            sent_chunks.append(chunk)
            chunk = sents[i]

        i += 1

    if chunk:
        sent_chunks.append(chunk)
    return sent_chunks

