import json
import os
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess

# File paths (adjust these paths as needed)
CORPUS_FILE = "../scifact/corpus.jsonl"           
QUERIES_FILE = "../scifact/queries.jsonl"         
BM25_RESULTS_FILE = "../output/Results.txt"         
DOC2VEC_MODEL_FILE = "../output/doc2vec_model.model" 
HYBRID_RESULTS_FILE = "../output/Results_hybrid.txt" 


ALPHA = 0.5

def load_corpus(file_path):
    """Load corpus from a JSONL file and return a list of TaggedDocument objects."""
    tagged_docs = []
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            doc = json.loads(line.strip())
            doc_id = doc["_id"]
            content = f"{doc.get('title', '')} {doc.get('text', '')}"
            tokens = simple_preprocess(content)
            tagged_docs.append(TaggedDocument(words=tokens, tags=[doc_id]))
    return tagged_docs

def load_bm25_results(file_path):
    """
    Load BM25 retrieval results in TREC format.
    Expected format per line: QueryID Q0 DocID Rank Score Tag
    Returns a dictionary: {query_id: {doc_id: bm25_score, ...}, ...}
    """
    results = {}
    with open(file_path, "r", encoding="utf8") as f:
        header = next(f)  # Skip header line
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            query_id, _, doc_id, rank, score, tag = parts
            if query_id not in results:
                results[query_id] = {}
            results[query_id][doc_id] = float(score)
    return results

def load_queries(file_path):
    """Load queries from a JSONL file."""
    queries = []
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            queries.append(json.loads(line.strip()))
    return queries

def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def normalize_scores(score_dict):
    """Normalize a dictionary of scores to the [0,1] range."""
    scores = list(score_dict.values())
    min_score = min(scores)
    max_score = max(scores)
    norm_scores = {}
    for key, score in score_dict.items():
        if max_score - min_score == 0:
            norm_scores[key] = 0
        else:
            norm_scores[key] = (score - min_score) / (max_score - min_score)
    return norm_scores

def infer_query_vector(query_text, model):
    """Infer a vector for the query using the Doc2Vec model."""
    tokens = simple_preprocess(query_text)
    return model.infer_vector(tokens)

if __name__ == "__main__":
    print("Loading corpus...")
    tagged_documents = load_corpus(CORPUS_FILE)
    
    if os.path.exists(DOC2VEC_MODEL_FILE):
        model = Doc2Vec.load(DOC2VEC_MODEL_FILE)
        print("Loaded Doc2Vec model from file.")
    else:
        print("Training new Doc2Vec model...")
        model = Doc2Vec(vector_size=100, window=10, min_count=5, workers=4, epochs=20)
        model.build_vocab(tagged_documents)
        model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(DOC2VEC_MODEL_FILE)
        print("Trained and saved new Doc2Vec model.")
    
    print("Loading BM25 results and queries...")
    bm25_results = load_bm25_results(BM25_RESULTS_FILE)
    queries = load_queries(QUERIES_FILE)

    hybrid_results = {}
    for query in queries:
        query_id = query["_id"]
        query_text = query["text"]
        if query_id not in bm25_results:
            print(f"No BM25 results for query {query_id}, skipping.")
            continue

        bm25_candidates = bm25_results[query_id]
        
        query_vec = infer_query_vector(query_text, model)

        doc2vec_scores = {}
        for doc_id in bm25_candidates:
            try:
                doc_vec = model.dv[doc_id]
                sim = cosine_similarity(query_vec, doc_vec)
                doc2vec_scores[doc_id] = sim
            except KeyError:
                continue  
        

        norm_bm25 = normalize_scores(bm25_candidates)
        norm_doc2vec = normalize_scores(doc2vec_scores)
        
        # final_score = ALPHA * (normalized BM25 score) + (1 - ALPHA) * (normalized Doc2Vec similarity)
        combined_scores = {}
        for doc_id in norm_bm25:
            combined_scores[doc_id] = ALPHA * norm_bm25.get(doc_id, 0) + (1 - ALPHA) * norm_doc2vec.get(doc_id, 0)
        
        ranked_candidates = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        hybrid_results[query_id] = ranked_candidates

    print("Writing hybrid results to file...")
    with open(HYBRID_RESULTS_FILE, "w", encoding="utf8") as f_out:
        f_out.write("QueryID Q0 DocID Rank Score Hybrid\n")
        for query_id, candidates in hybrid_results.items():
            for rank, (doc_id, score) in enumerate(candidates, start=1):
                f_out.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} Hybrid\n")
    
    print("Hybrid retrieval complete.")
    print("Hybrid results saved to", HYBRID_RESULTS_FILE)

