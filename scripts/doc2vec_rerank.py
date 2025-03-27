import json
import os
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from gensim.models import Doc2Vec

# File paths
corpus_file = "../scifact/corpus.jsonl"
queries_file = "../scifact/queries.jsonl"
model_file = "../output/doc2vec_model.model"
results_file = "../output/Results_doc2vec.txt"

# List to hold TaggedDocument objects
tagged_documents = []

# Read and process the corpus file
with open(corpus_file, "r", encoding="utf8") as f:
    for line in f:
        doc = json.loads(line.strip())
        doc_id = doc["_id"]
        # Combine title and text (you can change this if needed)
        content = f"{doc['title']} {doc['text']}"
        tokens = simple_preprocess(content)
        tagged_documents.append(TaggedDocument(words=tokens, tags=[doc_id]))

# Load or train the Doc2Vec model
if os.path.exists(model_file):
    model = Doc2Vec.load(model_file)
    print("Loaded Doc2Vec model from file.")
else:
    model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=20)
    model.build_vocab(tagged_documents)
    model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(model_file)
    print("Trained new Doc2Vec model and saved to file.")

# Function to infer query vector
def infer_query_vector(query_text):
    tokens = simple_preprocess(query_text)
    return model.infer_vector(tokens)

# Load queries
queries = []
with open(queries_file, "r", encoding="utf8") as f:
    for line in f:
        queries.append(json.loads(line.strip()))

# Open the results file for writing
with open(results_file, "w", encoding="utf8") as f_out:
    # Write header (if needed by your evaluation script)
    f_out.write("QueryID Q0 DocID Rank Score Doc2Vec\n")
    
    # Process each query and retrieve similar documents
    for query in queries:
        query_id = query["_id"]
        query_text = query["text"]
        query_vec = infer_query_vector(query_text)
        
        # Retrieve top 10 similar documents using Doc2Vec's built-in similarity method
        similar_docs = model.docvecs.most_similar(positive=[query_vec], topn=10)
        
        # Write each result in TREC format: QueryID Q0 DocID Rank Score Doc2Vec
        for rank, (doc_id, score) in enumerate(similar_docs, start=1):
            f_out.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} Doc2Vec\n")

print("Results saved to", results_file)
