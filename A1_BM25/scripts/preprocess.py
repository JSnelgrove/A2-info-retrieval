import json
import re
import os
import os
from nltk.stem import PorterStemmer

def load_stopwords(stopwords_file):
    """Loads stopwords into a set."""
    with open(stopwords_file, 'r') as f:
        return {word.lower() for word in f.read().splitlines()}  # Normalize case

def tokenize(text):
    """Tokenizes and removes numbers/punctuation."""
    return re.findall(r'\b[a-zA-Z]+\b', text)  # Only alphabetic words

def preprocess_text(text, stopwords_set):
    """Preprocesses text: tokenization, case normalization, stopword removal, stemming."""
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token.lower()) for token in tokenize(text) if token.lower() not in stopwords_set]
    return tokens

def preprocess_corpus(corpus_file, stopwords_file, output_file):
    """Reads corpus, preprocesses text, and saves tokenized output."""
    stopwords_set = load_stopwords(stopwords_file)
    
    preprocessed_data = []
    with open(corpus_file, 'r') as f:
        for line in f:
            try:
                doc = json.loads(line.strip())
                combined_text = f"{doc.get('title', '')} {doc.get('text', '')}"  # Merge title & text
                tokens = preprocess_text(combined_text, stopwords_set)
                if tokens:  # Skip empty documents
                    preprocessed_data.append({"doc_id": doc["_id"], "tokens": tokens})
            except json.JSONDecodeError:
                print(f"Skipping malformed line: {line.strip()}")

    with open(output_file, 'w') as f:
        json.dump(preprocessed_data, f)

if __name__ == "__main__":
    # Setup paths
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a1_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    preprocess_corpus(
        os.path.join(root_dir, "data/scifact/corpus.jsonl"),
        os.path.join(root_dir, "data/stopwords.txt"),
        os.path.join(a1_dir, "output/preprocessed_corpus.json")
    )
    print(f"Preprocessing complete. Output saved to {os.path.join(a1_dir, 'output/preprocessed_corpus.json')}")
