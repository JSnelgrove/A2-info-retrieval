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
    """Load system retrieval results from Results.txt."""
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
    preprocessed_corpus_file = "../output/preprocessed_corpus.json"
    results_file = "../output/Results.txt"
    relevance_file = "../scifact/qrels/test.tsv"

    # Compute vocabulary size and sample 100 tokens
    vocab_size, sample_tokens = count_unique_terms(preprocessed_corpus_file)
    print(f"Total unique terms in the corpus: {vocab_size}")
    print(f"Sample 100 Tokens: {sample_tokens}")

    # Extract top 10 results for queries 1 & 3
    top_results = extract_top_results(results_file, {"1", "3"}, top_n=10)
    print("\nFirst 10 Results for Queries 1 & 3:")
    for result in top_results:
        print(result)

    # Compute and display MAP and other evaluation metrics
    evaluation_scores = evaluate_results(relevance_file, results_file)
    print("\nEvaluation Results:")
    for metric, score in evaluation_scores.items():
        print(f"{metric}: {score:.4f}")

    # Save results to a file
    with open("../output/evaluation_summary.txt", "w") as f:
        f.write(f"Total unique terms: {vocab_size}\n")
        f.write(f"Sample 100 Tokens: {sample_tokens}\n\n")
        f.write("First 10 Results for Queries 1 & 3:\n")
        for result in top_results:
            f.write(result + "\n")
        f.write("\nEvaluation Results:\n")
        for metric, score in evaluation_scores.items():
            f.write(f"{metric}: {score:.4f}\n")

    print("\nEvaluation summary saved to evaluation_summary.txt.")
