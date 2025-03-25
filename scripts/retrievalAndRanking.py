import json
import math
import re
from collections import defaultdict
from nltk.corpus import wordnet
from preprocess import preprocess_text, load_stopwords  # Import from preprocess.py

# BM25 parameters
k1 = 1.2
b = 0.75

def load_queries(query_file):
    """Loads test queries from the JSONL file safely."""
    queries = []
    with open(query_file, 'r') as f:
        for line in f:
            try:
                queries.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping malformed query: {line.strip()}")
    return queries

def expand_query(query_tokens, inverted_index, doc_lengths, idf_values, top_n=5):
    """Expand query using WordNet synonyms + Pseudo-Relevance Feedback."""
    expanded_query = set(query_tokens)

    # Add WordNet synonyms (limited to frequent words)
    for token in query_tokens:
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                if lemma.count() > 2:  # Only add frequently occurring synonyms
                    expanded_query.add(lemma.name().replace("_", " "))

    # Pseudo-relevance feedback (PRF)
    scores = compute_bm25(query_tokens, inverted_index, doc_lengths, sum(doc_lengths.values()) / len(doc_lengths), idf_values)
    top_docs = [doc_id for doc_id, _ in scores[:top_n]]

    # Extract frequent words from top-ranked docs
    word_freq = defaultdict(int)
    for doc_id in top_docs:
        for word in inverted_index.get(doc_id, {}):
            word_freq[word] += inverted_index[word].get(doc_id, 0)

    # Add most relevant terms from top documents
    sorted_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    expanded_query.update(term for term, _ in sorted_terms)

    return list(expanded_query)

def compute_bm25(query_tokens, inverted_index, doc_lengths, avg_doc_length, idf_values):
    """Compute BM25 scores for ranking documents."""
    scores = defaultdict(float)

    for term in query_tokens:
        if term in inverted_index:
            idf = idf_values.get(term, 0)
            for doc_id, tf in inverted_index[term].items():
                doc_length = doc_lengths[doc_id]
                bm25_score = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length))))
                scores[doc_id] += bm25_score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def retrieve_and_rank(query, inverted_index, doc_lengths, idf_values, stopwords_set):
    """Retrieve ranked documents using BM25."""
    query_tokens = preprocess_text(query, stopwords_set)
    expanded_query = expand_query(query_tokens, inverted_index, doc_lengths, idf_values)
    avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths)

    return compute_bm25(expanded_query, inverted_index, doc_lengths, avg_doc_length, idf_values)

def load_inverted_index(index_file):
    """Loads the inverted index, document lengths, and IDF values."""
    with open(index_file, 'r') as f:
        data = json.load(f)
    return data["index"], data["doc_lengths"], data["idf"]

if __name__ == "__main__":
    index_file = "../output/invertedIndex.json"
    query_file = "../scifact/queries.jsonl"
    stopwords_file = "../stopwords.txt"
    output_file = "../output/Results.txt"

    # Load resources
    inverted_index, doc_lengths, idf_values = load_inverted_index(index_file)
    queries = load_queries(query_file)
    stopwords_set = load_stopwords(stopwords_file)

    # Process queries
    with open(output_file, "w") as f:
        f.write("Query ID | Q0 | doc ID | ranking | score | Tag\n")
        for query in queries:
            ranked_results = retrieve_and_rank(query["text"], inverted_index, doc_lengths, idf_values, stopwords_set)
            for rank, (doc_id, score) in enumerate(ranked_results[:100], start=1):
                f.write(f"{query['_id']} Q0 {doc_id} {rank} {score:.4f} BM25+QueryExpansion\n")

    print("Retrieval and ranking completed. Results saved to Results.txt.")
