import json
import pytrec_eval
import os

def count_unique_terms(preprocessed_corpus_file):
    """Counts the total number of unique terms in the corpus."""
    unique_terms = set()
    with open(preprocessed_corpus_file, 'r') as f:
        corpus = json.load(f)
        for doc in corpus:
            unique_terms.update(doc["tokens"])  # Add tokens to set
    return len(unique_terms), list(unique_terms)[:100]  # Return total count and a sample of 100 tokens

def extract_top_results(results_file, query_ids, top_n=10):
    """Extracts top N results for given queries, ensuring at least 10 results from each query."""
    extracted_results = {query_id: [] for query_id in query_ids}
    with open(results_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            query_id = parts[0]
            if query_id in query_ids and len(extracted_results[query_id]) < top_n:
                extracted_results[query_id].append(line.strip())
    combined_results = []
    for query_id in query_ids:
        combined_results.extend(extracted_results[query_id])
    return combined_results

def load_relevance(file_path):
    """Load ground truth relevance judgments (test.tsv)."""
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
    """Load system retrieval results from a results file in TREC format."""
    results = {}
    with open(file_path, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            query_id, _, doc_id, _, score, _ = parts
            if query_id not in results:
                results[query_id] = {}
            results[query_id][doc_id] = float(score)
    return results

def evaluate_results(relevance_file, results_file):
    """Evaluates the retrieval performance using pytrec_eval."""
    relevance = load_relevance(relevance_file)
    results = load_results(results_file)
    metrics = {'map', 'P_10', 'recall_100', 'recall_20', 'ndcg'}
    evaluator = pytrec_eval.RelevanceEvaluator(relevance, metrics)
    evaluation_results = evaluator.evaluate(results)
    average_scores = {metric: 0 for metric in metrics}
    num_queries = len(evaluation_results)
    for query_id in evaluation_results:
        for metric in metrics:
            average_scores[metric] += evaluation_results[query_id].get(metric, 0)
    if num_queries > 0:
        for metric in average_scores:
            average_scores[metric] /= num_queries
    return average_scores

if __name__ == "__main__":
    # Count unique terms in corpus
    preprocessed_corpus_file = "../output/preprocessed_corpus.json"
    num_terms, sample_tokens = count_unique_terms(preprocessed_corpus_file)
    print(f"Total unique terms in the corpus: {num_terms}")
    print(f"Sample 100 Tokens: {sample_tokens}")

    # File paths
    baseline_results_file = "../output/Results_hybrid.txt"
    doc2vec_results_file = "../output/Results_doc2vec.txt"
    relevance_file = "../scifact/qrels/test.tsv"

    # Extract top results for specific queries
    top_results_baseline = extract_top_results(baseline_results_file, {"1", "3"}, top_n=10)
    top_results_doc2vec = extract_top_results(doc2vec_results_file, {"1", "3"}, top_n=10)

    print("\nFirst 10 Baseline Results for Queries 1 & 3:")
    for result in top_results_baseline:
        print(result)

    print("\nFirst 10 Doc2Vec Results for Queries 1 & 3:")
    for result in top_results_doc2vec:
        print(result)

    # Evaluate both methods
    evaluation_baseline = evaluate_results(relevance_file, baseline_results_file)
    evaluation_doc2vec = evaluate_results(relevance_file, doc2vec_results_file)

    # Save combined evaluation summary to a file
    with open("../output/evaluation_summary.txt", "w") as f:
        f.write(f"Total unique terms: {num_terms}\n")
        f.write(f"Sample 100 Tokens: {sample_tokens}\n\n")
        f.write("Baseline Results (First 10 for Queries 1 & 3):\n")
        for result in top_results_baseline:
            f.write(result + "\n")
        f.write("\nDoc2Vec Results (First 10 for Queries 1 & 3):\n")
        for result in top_results_doc2vec:
            f.write(result + "\n")
        f.write("\nBaseline Evaluation Results:\n")
        for metric, score in evaluation_baseline.items():
            f.write(f"{metric}: {score:.4f}\n")
        f.write("\nDoc2Vec Evaluation Results:\n")
        for metric, score in evaluation_doc2vec.items():
            f.write(f"{metric}: {score:.4f}\n")

    print("\nEvaluation summary saved to evaluation_summary.txt.")
