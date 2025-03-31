import json
import os
import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
import pytrec_eval

# File paths
CORPUS_FILE = "../scifact/corpus.jsonl"
QUERIES_FILE = "../scifact/queries.jsonl"
BM25_RESULTS_FILE = "../output/Results.txt"
DOC2VEC_MODEL_FILE = "../output/doc2vec_model.model"
DOC2VEC_RESULTS_FILE = "../output/Results_doc2vec.txt"
EVAL_OUTPUT_FILE = "../output/evaluation_results_doc2vec.txt"
GROUND_TRUTH_FILE = "../scifact/qrels/test.tsv"

# Blending ratio between BM25 and Doc2Vec scores
ALPHA = 0.5

def load_corpus(file_path):
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
    results = {}
    with open(file_path, "r", encoding="utf8") as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            query_id, _, doc_id, _, score, _ = parts
            if query_id not in results:
                results[query_id] = {}
            results[query_id][doc_id] = float(score)
    return results

def load_queries(file_path):
    queries = []
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            queries.append(json.loads(line.strip()))
    return queries

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def normalize_scores(score_dict):
    scores = list(score_dict.values())
    min_score, max_score = min(scores), max(scores)
    if max_score - min_score == 0:
        return {k: 0 for k in score_dict}
    return {k: (v - min_score) / (max_score - min_score) for k, v in score_dict.items()}

def infer_query_vector(query_text, model):
    tokens = simple_preprocess(query_text)
    return model.infer_vector(tokens)

def evaluate_results(ground_truth_file, result_file, output_eval_file):
    def load_relevance(file_path):
        rel = {}
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                qid, _, docid, rel_score = parts
                rel.setdefault(qid, {})[docid] = int(rel_score)
        return rel

    def load_run(file_path):
        run = {}
        with open(file_path, "r") as f:
            next(f)
            for line in f:
                parts = line.strip().split()
                qid, _, docid, _, score, _ = parts
                run.setdefault(qid, {})[docid] = float(score)
        return run

    relevance = load_relevance(ground_truth_file)
    results = load_run(result_file)

    metrics = {'map', 'P_10', 'recall_100', 'recall_20', 'ndcg'}
    evaluator = pytrec_eval.RelevanceEvaluator(relevance, metrics)
    evaluation = evaluator.evaluate(results)

    averages = {m: 0 for m in metrics}
    for qid in evaluation:
        for m in metrics:
            averages[m] += evaluation[qid].get(m, 0)
    num_q = len(evaluation)
    for m in averages:
        averages[m] /= num_q

    with open(output_eval_file, "w") as f:
        for m, s in averages.items():
            f.write(f"{m}: {s:.4f}\n")

    print("\n📊 Doc2Vec Evaluation Results:")
    for m, s in averages.items():
        print(f"{m}: {s:.4f}")
    print(f"\n📁 Saved to: {output_eval_file}")

if __name__ == "__main__":
    print("📥 Loading corpus...")
    tagged_documents = load_corpus(CORPUS_FILE)

    if os.path.exists(DOC2VEC_MODEL_FILE):
        print("✅ Loading existing Doc2Vec model...")
        model = Doc2Vec.load(DOC2VEC_MODEL_FILE)
    else:
        print("🧠 Training new Doc2Vec model...")
        model = Doc2Vec(vector_size=100, window=10, min_count=5, workers=4, epochs=20)
        model.build_vocab(tagged_documents)
        model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(DOC2VEC_MODEL_FILE)
        print("✅ Model trained and saved.")

    print("📥 Loading BM25 results and queries...")
    bm25_results = load_bm25_results(BM25_RESULTS_FILE)
    queries = load_queries(QUERIES_FILE)

    hybrid_results = {}
    for query in queries:
        query_id = query["_id"]
        query_text = query["text"]
        if query_id not in bm25_results:
            continue

        bm25_scores = bm25_results[query_id]
        query_vec = infer_query_vector(query_text, model)

        doc2vec_scores = {}
        for doc_id in bm25_scores:
            try:
                doc_vec = model.dv[doc_id]
                sim = cosine_similarity(query_vec, doc_vec)
                doc2vec_scores[doc_id] = sim
            except KeyError:
                continue

        norm_bm25 = normalize_scores(bm25_scores)
        norm_doc2vec = normalize_scores(doc2vec_scores)

        combined_scores = {}
        for doc_id in norm_bm25:
            combined_scores[doc_id] = ALPHA * norm_bm25.get(doc_id, 0) + (1 - ALPHA) * norm_doc2vec.get(doc_id, 0)

        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        hybrid_results[query_id] = ranked

    print("💾 Writing combined results to file...")
    with open(DOC2VEC_RESULTS_FILE, "w") as f_out:
        f_out.write("QueryID Q0 DocID Rank Score Doc2Vec\n")
        for query_id, docs in hybrid_results.items():
            for rank, (doc_id, score) in enumerate(docs, start=1):
                f_out.write(f"{query_id} Q0 {doc_id} {rank} {score:.4f} Doc2Vec\n")

    print("✅ Results written to:", DOC2VEC_RESULTS_FILE)

    print("📈 Running evaluation...")
    evaluate_results(GROUND_TRUTH_FILE, DOC2VEC_RESULTS_FILE, EVAL_OUTPUT_FILE)
