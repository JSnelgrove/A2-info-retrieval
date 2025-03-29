import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import pytrec_eval

# Load MiniLM model (efficient BERT-based)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load BM25 results (top docs per query)
def load_bm25_results(results_file):
    top_docs = defaultdict(list)
    with open(results_file, 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            query_id, _, doc_id, _, _, _ = parts
            top_docs[query_id].append(doc_id)
    return top_docs

# Load corpus
def load_corpus(corpus_file):
    doc_texts = {}
    with open(corpus_file, 'r') as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc["_id"]
            text = doc.get("title", "") + " " + doc.get("text", "")
            doc_texts[doc_id] = text
    return doc_texts

# Load queries
def load_queries(query_file):
    queries = {}
    with open(query_file, 'r') as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    return queries

# Rerank top 25 docs using MiniLM
def rerank_with_minilm(bm25_results, queries, corpus, output_file, top_k=25):
    with open(output_file, 'w') as f_out:
        f_out.write("Query ID | Q0 | doc ID | ranking | score | Tag\n")
        for qid, doc_ids in bm25_results.items():
            query = queries.get(qid)
            if not query:
                continue

            doc_ids = doc_ids[:top_k]  # Speed boost: only re-rank top 25 docs

            doc_texts = [corpus[doc_id] for doc_id in doc_ids if doc_id in corpus]
            valid_doc_ids = [doc_id for doc_id in doc_ids if doc_id in corpus]

            if not doc_texts:
                continue

            query_embedding = model.encode(query, convert_to_tensor=True)
            doc_embeddings = model.encode(
                doc_texts,
                convert_to_tensor=True,
                batch_size=16,
                show_progress_bar=False
            )

            similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
            doc_scores = list(zip(valid_doc_ids, similarities.tolist()))
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            for rank, (doc_id, score) in enumerate(doc_scores[:top_k], start=1):
                f_out.write(f"{qid} Q0 {doc_id} {rank} {score:.4f} MiniLM\n")

    print(f"\n✅ Reranked results written to {output_file}")

# Evaluation with pytrec_eval
def evaluate_results(relevance_file, results_file):
    def load_relevance(file_path):
        relevance = {}
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                query_id, _, doc_id, relevance_score = parts
                if query_id not in relevance:
                    relevance[query_id] = {}
                relevance[query_id][doc_id] = int(relevance_score)
        return relevance

    def load_results(file_path):
        results = {}
        with open(file_path, "r") as f:
            next(f)
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    query_id, _, doc_id, _, score, _ = parts
                    if query_id not in results:
                        results[query_id] = {}
                    results[query_id][doc_id] = float(score)
        return results

    relevance = load_relevance(relevance_file)
    results = load_results(results_file)

    metrics = {'map', 'P_10', 'recall_100', 'recall_20', 'ndcg'}
    evaluator = pytrec_eval.RelevanceEvaluator(relevance, metrics)
    evaluation = evaluator.evaluate(results)

    average_scores = {metric: 0 for metric in metrics}
    num_queries = len(evaluation)
    if num_queries == 0:
        print("⚠️  No queries found in evaluation.")
        return

    for query_id in evaluation:
        for metric in metrics:
            average_scores[metric] += evaluation[query_id].get(metric, 0)

    for metric in average_scores:
        average_scores[metric] /= num_queries

    print("\n📊 Evaluation Results (MiniLM Re-ranking):")
    for metric, score in average_scores.items():
        print(f"{metric}: {score:.4f}")

    with open("../output/evaluation_results_minilm.txt", "w") as f:
        for metric, score in average_scores.items():
            f.write(f"{metric}: {score:.4f}\n")

    print("📁 Evaluation saved to output/evaluation_results_minilm.txt")

# Run pipeline
if __name__ == "__main__":
    # Load data
    print("🚀 Loading data...")
    bm25_results = load_bm25_results("../output/Results.txt")
    queries = load_queries("../scifact/queries.jsonl")
    corpus = load_corpus("../scifact/corpus.jsonl")
    
    # Re-rank with MiniLM
    print("⚡ Re-ranking top 25 docs per query with MiniLM...")
    output_file = "../output/Results_neural_minilm.txt"
    rerank_with_minilm(bm25_results, queries, corpus, output_file)
    
    # Evaluate results
    print("📈 Evaluating MiniLM results...")
    evaluate_results("../scifact/qrels/test.tsv", output_file)

