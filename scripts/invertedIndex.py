import json
import math
from preprocess import preprocess_text, load_stopwords  # Import from preprocess.py

def generate_inverted_index(input_file, output_file, stopwords_file):
    """Generates an inverted index from the preprocessed corpus."""
    inverted_index = {}
    doc_lengths = {}
    stopwords_set = load_stopwords(stopwords_file)

    with open(input_file, "r") as file:
        documents = json.load(file)  # Assuming preprocessed_corpus.json is a JSON array

        doc_count = len(documents)
        for doc in documents:
            doc_id = doc["doc_id"]
            tokens = preprocess_text(" ".join(doc["tokens"]), stopwords_set)  # Ensure consistent preprocessing
            doc_lengths[doc_id] = len(tokens)

            token_frequency = {}
            for token in tokens:
                token_frequency[token] = token_frequency.get(token, 0) + 1

            for token, tf in token_frequency.items():
                inverted_index.setdefault(token, {})[doc_id] = tf

    # Compute IDF values
    idf_values = {term: math.log(doc_count / len(postings)) for term, postings in inverted_index.items()}

    with open(output_file, 'w') as f:
        json.dump({"index": inverted_index, "doc_lengths": doc_lengths, "idf": idf_values}, f, indent=4)

if __name__ == "__main__":
    generate_inverted_index("../output/preprocessed_corpus.json", "../output/invertedIndex.json", "../stopwords.txt")
    print("Inverted index created successfully.")
