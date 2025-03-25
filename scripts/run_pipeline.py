import subprocess

def run_script(script_name):
    """Runs a Python script and prints its output in real-time."""
    print(f"\nRunning {script_name}...\n" + "-" * 50)
    process = subprocess.Popen(["python3", script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in process.stdout:
        print(line, end="") 

    stderr_output = process.stderr.read()
    if stderr_output:
        print("\nERROR in", script_name, ":\n", stderr_output)

if __name__ == "__main__":
    scripts = ["preprocess.py", "invertedIndex.py", "retrievalAndRanking.py", "evaluate.py"]

    for script in scripts:
        run_script(script)

    print("Pipeline completed successfully!")
