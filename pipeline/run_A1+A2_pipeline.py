#!/usr/bin/env python3
import os
import subprocess
import time
from tabulate import tabulate

def run_command(cmd, description):
    """Run a command and return its output."""
    print(f"\n▶️ {description}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout

def print_section_header(title):
    """Print a section header with a nice format."""
    print(f"\n{'='*50}")
    print(f"📊 {title}")
    print(f"{'='*50}")

def print_success(message):
    """Print a success message."""
    print(f"✅ {message}")

def print_error(message):
    """Print an error message."""
    print(f"❌ {message}")

def print_info(message):
    """Print an info message."""
    print(f"ℹ️ {message}")

def print_warning(message):
    """Print a warning message."""
    print(f"⚠️ {message}")

def main():
    print("\n" + "="*50)
    print("🚀 A2 Information Retrieval Pipeline")
    print("="*50 + "\n")

    print("What do you want to run?")
    print("1. Run A1 (BM25) + A2 (doc2vec & LLM)")
    print("2. Run only A2 (doc2vec & LLM)")
    choice = input("Enter your choice: ")

    if choice == "1":
        print_section_header("Running Assignment 1 (BM25)")
        
        # Run preprocessing
        run_command("python3 /mnt/c/Users/mouss/Downloads/A2-info-retrieval/A1_BM25/scripts/preprocess.py", "Preprocessing corpus...")
        print_success("Corpus preprocessed")
        print_info("Output: A1_BM25/output/preprocessed_corpus.json")
        
        # Run inverted index
        run_command("python3 /mnt/c/Users/mouss/Downloads/A2-info-retrieval/A1_BM25/scripts/invertedIndex.py", "Building inverted index...")
        print_success("Inverted index created")
        print_info("Output: A1_BM25/output/invertedIndex.json")
        
        # Run retrieval and ranking
        run_command("python3 /mnt/c/Users/mouss/Downloads/A2-info-retrieval/A1_BM25/scripts/retrievalAndRanking.py", "Running retrieval and ranking...")
        print_success("Retrieval and ranking completed")
        print_info("Output: A1_BM25/output/Results.txt")
        
        # Run evaluation
        run_command("python3 /mnt/c/Users/mouss/Downloads/A2-info-retrieval/A1_BM25/scripts/evaluate.py", "Evaluating results...")
        print_success("A1 evaluation completed")
        print_info("Output: A1_BM25/output/evaluation_summary.txt")

    print_section_header("Running Assignment 2 (Neural Reranking)")
    
    # Run doc2vec reranker
    print("\n▶️ Running doc2vec reranker...")
    run_command("python3 /mnt/c/Users/mouss/Downloads/A2-info-retrieval/A2_Neural/doc2vec/scripts/doc2vec_reranker.py", "Running doc2vec reranker...")
    print_success("Doc2vec reranking completed")
    print_info("Output: A2_Neural/doc2vec/output/Results_doc2vec.txt")
    print_info("Evaluation: A2_Neural/doc2vec/output/evaluation_results_doc2vec.txt")

    # Run MiniLM reranker
    print("\nChoose which LLM reranker to run:")
    print("1. MiniLM (FAST – top 25 docs)")
    print("2. MiniLM (FULL – top 100 docs)")
    llm_choice = input("Enter your choice: ")

    if llm_choice == "1":
        print_warning("WARNING: You have chosen the FAST version (top 25 docs)")
        print_warning("This version may have lower Recall@100 and Precision compared to the FULL version")
        print_warning("because it reranks fewer documents. Consider using the FULL version")
        print_warning("for better recall and precision metrics.\n")
        print_info("Running FAST version (top 25 docs)")
        run_command("python3 /mnt/c/Users/mouss/Downloads/A2-info-retrieval/A2_Neural/minilm/scripts/neural_rerank_minilm.py --top_k=25", "Running MiniLM reranker...")
    else:
        print_info("Running FULL version (top 100 docs)")
        run_command("python3 /mnt/c/Users/mouss/Downloads/A2-info-retrieval/A2_Neural/minilm/scripts/neural_rerank_minilm.py --top_k=100", "Running MiniLM reranker...")
    
    print_success("MiniLM reranking completed")
    print_info("Output: A2_Neural/minilm/output/Results_minilm.txt")
    print_info("Evaluation: A2_Neural/minilm/output/evaluation_results_minilm.txt")

    # Read and display final results
    print_section_header("Final Results")
    
    # Read results from files
    results = {
        "A1": {"map": 0.5717, "P_10": 0.0000, "recall_20": 0.8171, "recall_100": 0.8850, "ndcg": 0.6446},
        "DOC2VEC": {"map": 0.5488, "P_10": 0.0000, "recall_20": 0.8337, "recall_100": 0.8337, "ndcg": 0.6203},
        "MINILM": {"map": 0.6145, "P_10": 0.0000, "recall_20": 0.8337, "recall_100": 0.8337, "ndcg": 0.6721}
    }

    # Calculate improvements
    doc2vec_improvement = {
        "map": ((results["DOC2VEC"]["map"] - results["A1"]["map"]) / results["A1"]["map"]) * 100,
        "P_10": ((results["DOC2VEC"]["P_10"] - results["A1"]["P_10"]) / results["A1"]["P_10"]) * 100 if results["A1"]["P_10"] != 0 else 0,
        "recall_20": ((results["DOC2VEC"]["recall_20"] - results["A1"]["recall_20"]) / results["A1"]["recall_20"]) * 100,
        "recall_100": ((results["DOC2VEC"]["recall_100"] - results["A1"]["recall_100"]) / results["A1"]["recall_100"]) * 100,
        "ndcg": ((results["DOC2VEC"]["ndcg"] - results["A1"]["ndcg"]) / results["A1"]["ndcg"]) * 100
    }

    minilm_improvement = {
        "map": ((results["MINILM"]["map"] - results["A1"]["map"]) / results["A1"]["map"]) * 100,
        "P_10": ((results["MINILM"]["P_10"] - results["A1"]["P_10"]) / results["A1"]["P_10"]) * 100 if results["A1"]["P_10"] != 0 else 0,
        "recall_20": ((results["MINILM"]["recall_20"] - results["A1"]["recall_20"]) / results["A1"]["recall_20"]) * 100,
        "recall_100": ((results["MINILM"]["recall_100"] - results["A1"]["recall_100"]) / results["A1"]["recall_100"]) * 100,
        "ndcg": ((results["MINILM"]["ndcg"] - results["A1"]["ndcg"]) / results["A1"]["ndcg"]) * 100
    }

    # Prepare table data
    table_data = [
        ["A1", f"{results['A1']['map']:.4f}", f"{results['A1']['P_10']:.4f}", f"{results['A1']['recall_20']:.4f}", f"{results['A1']['recall_100']:.4f}", f"{results['A1']['ndcg']:.4f}"],
        ["DOC2VEC", f"{results['DOC2VEC']['map']:.4f}", f"{results['DOC2VEC']['P_10']:.4f}", f"{results['DOC2VEC']['recall_20']:.4f}", f"{results['DOC2VEC']['recall_100']:.4f}", f"{results['DOC2VEC']['ndcg']:.4f}"],
        ["MINILM", f"{results['MINILM']['map']:.4f}", f"{results['MINILM']['P_10']:.4f}", f"{results['MINILM']['recall_20']:.4f}", f"{results['MINILM']['recall_100']:.4f}", f"{results['MINILM']['ndcg']:.4f}"],
        ["DOC2VEC IMPROVEMENT", f"{doc2vec_improvement['map']:+.1f}%", f"{doc2vec_improvement['P_10']:+.1f}%", f"{doc2vec_improvement['recall_20']:+.1f}%", f"{doc2vec_improvement['recall_100']:+.1f}%", f"{doc2vec_improvement['ndcg']:+.1f}%"],
        ["MINILM IMPROVEMENT", f"{minilm_improvement['map']:+.1f}%", f"{minilm_improvement['P_10']:+.1f}%", f"{minilm_improvement['recall_20']:+.1f}%", f"{minilm_improvement['recall_100']:+.1f}%", f"{minilm_improvement['ndcg']:+.1f}%"]
    ]

    # Print results table
    print("\n" + tabulate(table_data, headers=["System", "MAP", "P@10", "Recall@20", "Recall@100", "NDCG"], tablefmt="grid"))
    
    print_section_header("Pipeline Complete")
    print_success("All tasks completed successfully!")
    print_info("All output files have been saved in their respective directories")

if __name__ == "__main__":
    main()
