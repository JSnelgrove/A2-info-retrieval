import json
import re
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
    preprocess_corpus("../scifact/corpus.jsonl", "../stopwords.txt", "../output/preprocessed_corpus.json")
    print("Preprocessing complete. Output saved to ../output/preprocessed_corpus.json")
