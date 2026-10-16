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

# Step 9 - chunk_with_overlap
def chunk_with_overlap(text, chunk_size, overlap):
    # TODO: return sliding-window chunks of length chunk_size sharing `overlap` chars
    chunks = []
    for i in range(0, len(text), max(chunk_size - overlap, 1)):
        chunks.append(text[i:i+chunk_size])

    return chunks

# Step 10 - attach_chunk_metadata
def attach_chunk_metadata(chunks, source):
    # TODO: wrap each chunk string with source, position, and chunk_id metadata.
    return [dict(text=text,source=source,position=i,chunk_id=f"{source}::{i}") for i, text in enumerate(chunks)]

# Step 11 - load_embedding_model
from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name):
    # TODO: return a sentence-transformers model instance for the given model_name.
    return SentenceTransformer(model_name)

# Step 12 - embed_text
def embed_text(model, text):
    # TODO: Return a 1D float32 numpy embedding vector for the given text string.
    return model.encode(text)

# Step 13 - embed_chunks
def embed_chunks(model, chunks, batch_size=32):
    """Batch-embed a list of chunk strings or chunk dicts into a 2D float32 matrix."""
    # TODO: normalize chunk inputs to strings, encode in batches, return (n, d) float32 array
    texts = chunks
    if not chunks:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)
    if isinstance(chunks[0], dict):
        texts = [chunk['text'] for chunk in chunks]

    return model.encode(texts, batch_size=batch_size)

# Step 14 - l2_normalize
import numpy as np

def l2_normalize(matrix):
    # TODO: rescale each row of `matrix` to unit L2 norm, leaving all-zero rows unchanged.
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms = np.where(norms==0.0, 1.0, norms)

    return matrix/norms

# Step 15 - save_corpus
import os, json


def save_corpus(embeddings, chunks, directory):
    # TODO: persist embeddings (.npy) and chunks (.json) into directory, then reload and return both.
    os.makedirs(directory, exist_ok=True)
    emb_file = os.path.join(directory, "embeddings.npy")
    np.save(emb_file, embeddings)

    js_file = os.path.join(directory, "chunks.json")
    with open(js_file, "w+") as fp:
        json.dump(chunks, fp, indent=4)

    reload_emb = np.load(emb_file)
    with open(js_file, "r+") as fp:
        reload_chunks = json.load(fp)

    return dict(
        embeddings=reload_emb,
        chunks=reload_chunks
    )

# Step 16 - cosine_similarity_search
import numpy as np

def cosine_similarity_search(query_vector, chunk_matrix):
    """Cosine similarity between query_vector (d,) and each row of chunk_matrix (n,d)."""
    # TODO: compute cosine similarity between the query vector and every chunk row
    norm_vec = l2_normalize(query_vector)
    norm_mat = l2_normalize(chunk_matrix)

    return (norm_mat * norm_vec[None, :]).sum(axis=-1)

# Step 17 - top_k_indices
import numpy as np

def top_k_indices(scores, k):
    """Return indices of the k highest scores in descending order."""
    # TODO: rank the score array and return the top-k positions as a numpy array
    return np.argsort(-scores)[:k]

# Step 18 - top_k_chunks
import numpy as np

def top_k_chunks(scores, chunks, k):
    # TODO: return list of (chunk, score) tuples for the top-k scores, sorted descending
    inds = top_k_indices(scores, k)
    return [(chunks[i], float(scores[i])) for i in inds]

# Step 19 - retrieve
def retrieve(query, model, chunk_matrix, chunks, k):
    # TODO: embed the query, score it against chunk_matrix, return top-k (chunk, score) pairs.
    qvec = embed_text(model, query)
    scores = cosine_similarity_search(qvec, chunk_matrix)

    return top_k_chunks(scores, chunks, k)

# Step 20 - build_faiss_index
def build_faiss_index(chunk_matrix):
    # TODO: build a FAISS inner-product index and add all rows of chunk_matrix to it
    ind = faiss.IndexFlatIP(chunk_matrix.shape[-1])
    ind.add(chunk_matrix)

    return ind

# Step 21 - faiss_search
import numpy as np
import faiss


def faiss_search(index, query_vector, k):
    """Return top-k (scores, indices) as 1D arrays for a single query vector."""
    # TODO: query the FAISS index with the single query vector and return flat top-k arrays
    qvec = l2_normalize(query_vector).astype(np.float32)
    if isinstance(index, faiss.IndexFlatIP):
        scores, inds = index.search(qvec[None, :], index.ntotal)
        scores, inds = scores[0], inds[0]
        indices = sorted(range(index.ntotal), key=lambda x: (-scores[x], inds[x]))[:k]
        scores, inds = scores[indices], inds[indices]

    else:
        scores, inds = index.search(qvec[None, :], k)
        scores, inds = scores[0], inds[0]

    return scores, inds

# Step 22 - compare_faiss_to_numpy
def compare_faiss_to_numpy(query_vector, chunk_matrix, index, k):
    # TODO: return True iff FAISS and numpy cosine search agree on the top-k indices
    scores = cosine_similarity_search(query_vector, chunk_matrix)
    np_inds = top_k_indices(scores, k)

    _, fa_inds = faiss_search(index, query_vector, k)

    return all(np_inds == fa_inds)

# Step 23 - save_faiss_index
def save_faiss_index(index, path):
    """Write `index` to `path` and return the index loaded back from disk."""
    # TODO: persist the index to `path` and reload it; return the reloaded index
    faiss.write_index(index, path)
    reload_ind = faiss.read_index(path)
    return reload_ind

# Step 24 - build_prompt_template
def build_prompt_template():
    # TODO: return a RAG prompt template string with {context} and {question} placeholders.
    return "{context}\n{question}"

# Step 25 - format_context
def format_context(retrieved):
    # TODO: render each (chunk, score) as '[i] {text} (source={source})' and join with newlines
    return "\n".join([f"""[{i}] {chunk["text"]} (source={chunk["source"]})""" for i, (chunk, _) in enumerate(retrieved, 1)])

# Step 26 - truncate_context
def truncate_context(context, max_chars):
    # TODO: trim context so len(result) <= max_chars, preferring a whitespace boundary
    words = context.split()
    cur_len = 0
    nwords = len(words)
    if nwords == 1:
        return context[:max_chars]

    i = 0
    while i < nwords:
        cur_len += len(words[i])
        if cur_len + i > max_chars:
            break
        i += 1

    return " ".join(words[:i])

# Step 27 - add_system_instruction
def add_system_instruction(prompt):
    """Prepend a fixed system instruction to the prompt."""
    # TODO: return a string that starts with a system instruction telling the model to use only the context
    return "You are a helpful assistant. Answer the question using ONLY the provided context. If the answer is not in the context, say 'I do not know'.\n\n" + prompt

# Step 28 - load_generator
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_generator(model_name='sshleifer/tiny-gpt2'):
    # TODO: load a small local causal LM and its tokenizer, ensuring tokenizer.pad_token is set.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer

# Step 29 - generate_answer
import torch
from transformers import AutoModel


def generate_answer(model, tokenizer, prompt, max_new_tokens=32):
    # TODO: greedy-decode a continuation from `prompt` and return only the new tokens as a string.
    # new_tokens = []
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs['input_ids']

    if isinstance(model, AutoModel):
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_length=len(inputs['input_ids']) + max_new_tokens
        )
    else:
        outputs = model.generate(
            **inputs,
            do_sample=False,
            # max_length=len(inputs['input_ids']) + max_new_tokens
            max_new_tokens=max_new_tokens
        )

    # with torch.no_grad():
    #     for _ in range(max_new_tokens):
    #         logits = model(input_ids).logits[:, -1]
    #         next_id = torch.argmax(logits, dim=-1, keepdim=True)
    #         new_tokens.append(next_id[0])
    #         input_ids = torch.cat([input_ids, next_id], dim=-1)

    #         if next_id == tokenizer.eos_token_id:
    #             break

    if isinstance(tokenizer, AutoTokenizer):
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    else:
        decoded = "".join(tokenizer.decode(outputs, skip_special_tokens=True))
    decoded = decoded.replace(prompt, "").strip()

    return decoded

# Step 30 - rag_answer
def rag_answer(query, chunks, embeddings, embed_model, generator, tokenizer, k=3):
    # TODO: embed query, retrieve top-k chunks, build prompt, generate answer, return dict.
    retrieved = retrieve(query, embed_model, embeddings, chunks, k)
    context = format_context(retrieved)

    sources = [chunk for chunk, _ in retrieved]
    prompt = build_prompt_template().format(context=context, question=query)
    answer = generate_answer(generator, tokenizer, prompt)

    return dict(
        answer=answer,
        sources=sources,
        query=query
    )

# Step 31 - track_source_chunk_ids
def track_source_chunk_ids(source_chunks):
    # TODO: return the list of chunk ids from the retrieved source chunks, preserving order
    return [chunk['id'] for chunk in source_chunks if 'id' in chunk]

