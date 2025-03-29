import subprocess
import os
from tabulate import tabulate

# Paths
scripts = {
    "A1": ["preprocess.py", "invertedIndex.py", "retrievalAndRanking.py", "evaluate.py"],
    "doc2vec": ["doc2vec_reranker.py"],
    "minilm_short": ["neural_rerank_minilm.py", "--top_k=25"],
    "minilm_long": ["neural_rerank_minilm.py", "--top_k=100"]
}

# Result files and tags
results = {
    "A1": "../output/evaluation_summary.txt",
    "doc2vec": "../output/evaluation_results_doc2vec.txt",
    "minilm": "../output/evaluation_results_minilm.txt"
}

# Prompt
def menu(prompt, options):
    print(f"\n{prompt}")
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")
    choice = input("Enter your choice: ").strip()
    return int(choice) - 1

# Run a script and print output live
def run_script(script, extra_args=[]):
    if isinstance(script, list):
        cmd = ["python3"] + script + extra_args
    else:
        cmd = ["python3", script]
    print(f"\n🚀 Running: {' '.join(cmd)}")
    subprocess.run(cmd)

# Load results
def load_scores(filepath):
    scores = {}
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        for line in f:
            if ":" in line:
                metric, score = line.strip().split(":")
                metric = metric.strip().lower()
                if metric in ["map", "p_10", "recall_20", "recall_100", "ndcg"]:
                    scores[metric] = float(score.strip())
    return scores

def calculate_improvement(a1_scores, a2_scores):
    if not a1_scores or not a2_scores:
        return None
    improvements = {}
    for metric in ["map", "P_10", "recall_20", "recall_100", "ndcg"]:
        if metric in a1_scores and metric in a2_scores:
            improvement = ((a2_scores[metric] - a1_scores[metric]) / a1_scores[metric]) * 100
            improvements[metric] = improvement
        else:
            improvements[metric] = 0
    return improvements

# Main logic
def main():
    print("📚 A2 Master Script: Compare A1 vs A2 Retrieval Systems")

    # Ask user: Run A1 then A2, or A2 only?
    run_type = menu("What do you want to run?", [
        "Run A1 (BM25) + A2 (doc2vec & LLM)",
        "Run only A2 (doc2vec & LLM)"
    ])

    run_a1 = run_type == 0

    if run_a1:
        print("\n▶️ Running Assignment 1 (BM25)...")
        for script in scripts["A1"]:
            run_script(script)

    print("\n▶️ Running A2: doc2vec reranker...")
    run_script(scripts["doc2vec"][0])

    # Ask for MiniLM version
    llm_type = menu("Choose which LLM reranker to run:", [
        "MiniLM (FAST – top 25 docs)",
        "MiniLM (FULL – top 100 docs)"
    ])

    if llm_type == 0:
        print("\n⚠️  WARNING: You have chosen the FAST version (top 25 docs)")
        print("   This version may have lower Recall@100 and Precision compared to the FULL version")
        print("   because it reranks fewer documents. Consider using the FULL version")
        print("   for better recall and precision metrics.\n")
        print("▶️ Running MiniLM SHORT (fast)...")
        run_script(scripts["minilm_short"])
    else:
        print("\n▶️ Running MiniLM LONG (extended)...")
        run_script(scripts["minilm_long"])

    # Load all results
    table = []
    a1_scores = None
    for label, path in results.items():
        scores = load_scores(path)
        if scores:
            if label == "A1":
                a1_scores = scores
            row = [label.upper()] + [f"{scores.get(metric, 0):.4f}" for metric in ["map", "P_10", "recall_20", "recall_100", "ndcg"]]
            table.append(row)
        else:
            table.append([label.upper()] + ["N/A"] * 5)

    # Calculate improvements for both DOC2VEC and MINILM
    if a1_scores:
        # Calculate DOC2VEC improvements
        doc2vec_scores = load_scores(results["doc2vec"])
        if doc2vec_scores:
            doc2vec_improvements = calculate_improvement(a1_scores, doc2vec_scores)
            if doc2vec_improvements:
                doc2vec_row = ["DOC2VEC IMPROVEMENT"] + [f"{doc2vec_improvements.get(metric, 0):+.1f}%" for metric in ["map", "P_10", "recall_20", "recall_100", "ndcg"]]
                table.append(doc2vec_row)

        # Calculate MINILM improvements
        minilm_scores = load_scores(results["minilm"])
        if minilm_scores:
            minilm_improvements = calculate_improvement(a1_scores, minilm_scores)
            if minilm_improvements:
                minilm_row = ["MINILM IMPROVEMENT"] + [f"{minilm_improvements.get(metric, 0):+.1f}%" for metric in ["map", "P_10", "recall_20", "recall_100", "ndcg"]]
                table.append(minilm_row)

    # Display table
    print("\n📊 Final Evaluation Comparison:")
    print(tabulate(table, headers=["System", "MAP", "P@10", "Recall@20", "Recall@100", "NDCG"], tablefmt="fancy_grid"))

if __name__ == "__main__":
    main()
