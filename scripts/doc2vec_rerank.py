import json
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess

# Path to your corpus file (each line is a JSON document)
corpus_file = "../scifact/corpus.jsonl"

# List to hold TaggedDocument objects
tagged_documents = []

# Read and process the corpus file
with open(corpus_file, "r", encoding="utf8") as f:
    for line in f:
        # Parse each JSON line
        doc = json.loads(line.strip())
        doc_id = doc["_id"]
        # Combine title and text (optional, depending on your design)
        content = f"{doc['title']} {doc['text']}"
        tokens = simple_preprocess(content)
        tagged_documents.append(TaggedDocument(words=tokens, tags=[doc_id]))

print(f"Processed {len(tagged_documents)} documents.")