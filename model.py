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

