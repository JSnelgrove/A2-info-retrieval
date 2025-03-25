# CSI4107 - Winter 2025
## Assignment 1: Information Retrieval System
**Due: Feb 9, 10 PM**

## **Group Members**
- **Jack Snelgrove** - 300247435
- **Lina Moussadek** - 300259985
- **Eli Wynn** - 300248135

### **Task Division**
- **Preprocessing & Stopword Removal:** Lina Moussadek
- **Indexing & Inverted Index Construction:** Jack Snelgrove
- **Retrieval & Ranking using BM25:** Eli Wynn
- **Evaluation & Report Writing:** Lina, Jack and Eli
- **README writeup:** Lina

---

## **Project Overview**
This project implements an **Information Retrieval (IR) system** using the **BM25 ranking algorithm** and **pseudo-relevance feedback** for improved query expansion. The system is designed to work with the **Scifact dataset** from the **BEIR collection**.

### **Functional Overview**
1. **Preprocessing:** Tokenization, stopword removal, and stemming.
2. **Indexing:** Construction of an inverted index with TF-IDF weighting.
3. **Retrieval & Ranking:** BM25-based retrieval and query expansion using **WordNet synonyms** and **pseudo-relevance feedback**.

The system produces a **ranked list of documents** for each query and outputs results in a **trec_eval-compatible format**.

---

## **Installation & Running the Code**
### **Dependencies**
- Python 3.8+
- Required Python Libraries:
  ```bash
  pip install nltk jsonlines
  ```
- Download and set up NLTK resources:
  ```python
  import nltk
  nltk.download('stopwords')
  nltk.download('wordnet')
  ```

### **Running the System**

**I - Running the Pipeline altogether:**
   ```bash
   wsl
   python3 run_pipeline.py
   ```
**II - Running the Retrieval tasks Separately:**

1. **Preprocessing Step:**
   ```bash
   python preprocess.py
   ```
   - **Input:** `corpus.jsonl`, `stopwords.txt`
   - **Output:** `preprocessed_corpus.json`

2. **Indexing Step:**
   ```bash
   python invertedIndex.py
   ```
   - **Input:** `preprocessed_corpus.json`
   - **Output:** `invertedIndex.json`

3. **Retrieval & Ranking Step:**
   ```bash
   python retrievalAndRanking.py
   ```
   - **Input:** `queries.jsonl`, `invertedIndex.json`
   - **Output:** `Results.txt`

4. **Evaluation:**
   ```bash
   wsl
   python3 evaluate.py
   ```

---

## **Algorithmic Implementation**

### **Step 1: Preprocessing**
#### **Algorithm**
Preprocessing is an important step in Information Retrieval (IR), ensuring that raw text is structured and standardized before indexing. This phase transforms unstructured data into a clean format by applying **tokenization, stopword removal, and stemming** to improve efficiency and accuracy in retrieval.

1. **`load_stopwords(stopwords_file)`**:
   - **Purpose:** Loads stopwords from an external file and stores them in a Python **set** for fast lookup.
   - **Why a set** Lookups in sets are O(1) on average, making stopword filtering extremely efficient.
   - **Case Normalization:** Converts all stopwords to lowercase to maintain uniformity in filtering.
   - **Example:** If "the", "and", and "is" are in `stopwords.txt`, they will be stored as `{"the", "and", "is"}` and ignored during tokenization.

2. **`tokenize(text)`**:
   - **Purpose:** Extracts words from raw text while discarding numbers and punctuation using a **regular expression**.
   - **Why Regular Expressions** They provide an efficient and controlled way to extract only alphabetic words.
   - **Example:**
     ```python
     tokenize("COVID-19 pandemic affects 2021 economy!")
     # Output: ['COVID', 'pandemic', 'affects', 'economy']
     ```
   - **Why exclude numbers?** In scientific retrieval, numbers might not be meaningful search terms unless explicitly required.

3. **`preprocess_text(text, stopwords_set)`**:
   - **Purpose:** Applies multiple text normalization steps:
     1. **Lowercasing**: Ensures consistency (e.g., "Science" and "science" are treated as the same term).
     2. **Tokenization**: Calls `tokenize(text)` to extract words.
     3. **Stopword Removal**: Uses `stopwords_set` to filter out common words that do not add retrieval value.
     4. **Stemming**: Uses the **Porter Stemmer** from NLTK to reduce words to their root form.
   - **Why Stemming** It helps improve recall by mapping related words to a common base (e.g., "running" → "run").
   - **Example:**
     ```python
     preprocess_text("The scientists are researching effective treatments for COVID-19.", stopwords_set)
     # Output: ['scientist', 'research', 'effect', 'treatment', 'covid']
     ```

4. **`preprocess_corpus(corpus_file, stopwords_file, output_file)`**:
   - **Purpose:** Reads a **JSONL-formatted corpus**, processes each document’s title and text, and outputs tokenized data to a file.
   - **Steps:**
     1. **Loads stopwords** using `load_stopwords()`.
     2. **Iterates through each document** in the corpus.
     3. **Combines the title and body text** into a single string.
     4. **Preprocesses the text** using `preprocess_text()`.
     5. **Skips empty documents** to avoid indexing unnecessary data.
     6. **Handles malformed JSON lines gracefully**, printing warnings for invalid entries.
   - **Example Input (`corpus.jsonl`)**:
     ```json
     {"_id": "123", "title": "COVID-19 Vaccine Success", "text": "The vaccine is 95% effective against severe cases."}
     ```
   - **Example Output (`preprocessed_corpus.json`)**:
     ```json
     {"doc_id": "123", "tokens": ["covid", "vaccin", "success", "effect"]}
     ```

---

### **Step 2: Indexing**
#### **Algorithm**
The indexing process is responsible for transforming the preprocessed corpus into a data structure that enables efficient document retrieval. This implementation constructs an **inverted index**, which maps each unique term to a list of documents in which it appears, along with its frequency. 

1. **Term Frequency (TF) Calculation:**
   - This step involves counting occurrences of each token within a document. **Term frequency (TF)** represents the importance of a term within a document, and higher frequencies indicate stronger relevance.
   - The function iterates over tokenized words and updates a **dictionary-based frequency counter**.
   - Example: In a document with text "science science experiment," the term "science" would have a TF of 2, and "experiment" would have a TF of 1.

2. **Inverse Document Frequency (IDF) Calculation:**
   - **Inverse Document Frequency (IDF)** is computed as `log(N / df)`, where:
     - `N` is the total number of documents in the corpus.
     - `df` is the number of documents containing the term.
   - The purpose of IDF is to **assign higher importance to rare terms** and penalize frequently occurring words (such as "the" or "is").
   - This technique improves retrieval precision by ensuring that common words do not overshadow more informative terms.

3. **Index Construction:**
   - The inverted index is stored in a **dictionary-based data structure**, allowing efficient retrieval of documents containing specific terms.
   - This is implemented by iterating through all tokens in a document, updating their associated **document frequency list** in the inverted index.
   - Example format:
     ```json
     {
       "science": {"doc1": 2, "doc3": 1},
       "experiment": {"doc2": 1, "doc3": 3}
     }
     ```
   - The dictionary-based approach enables **fast lookups and optimized query performance**.

#### **Data Structure**
- **Inverted Index Format:**
  ```json
  {
    "term": {"doc_id": tf},
    "doc_lengths": {"doc_id": length},
    "idf": {"term": idf_value}
  }
  ```
- **Document Lengths:**
  - A separate dictionary stores the length of each document.
  - This is important for ranking models like **BM25**, which normalize term frequencies based on document length.
- **Storage Efficiency:**
  - **Dictionary-based storage** allows efficient retrieval of posting lists during query execution.
  - **JSON serialization** ensures that the inverted index can be easily saved and loaded for use in retrieval.

By constructing this **inverted index**, the system allows fast lookup of relevant documents for a given query term, forming the backbone of the retrieval and ranking phase. This indexing strategy ensures that document searches remain **scalable and efficient**, even for large text collections.

---

### **Step 3: Retrieval & Ranking**
#### **Algorithm**
The retrieval and ranking step is responsible for returning the most relevant documents for a given query using the **BM25 ranking function**. This step involves **query preprocessing**, **query expansion**, **BM25 scoring**, and **ranking of documents**.

1. **Query Preprocessing:**
   - The user’s input query is first preprocessed using `preprocess_text()` from `preprocess.py`. This ensures consistency between document indexing and query representation.
   - Stopword removal, stemming, and tokenization are applied to refine the query terms and improve matching efficiency.

2. **Query Expansion:**
   - This step enhances recall by adding relevant terms to the query using two techniques:
     - **WordNet Synonyms:** Retrieves synonyms of each query term from WordNet. This allows related words to be matched even if they are not in the original query.
     - **Pseudo-Relevance Feedback (PRF):** Uses the top retrieved documents from an initial search to extract additional important terms and append them to the query.
   - Example: If the original query is "climate change," WordNet expansion might add "global warming," and PRF might add "carbon emissions" if those terms are frequent in top-ranked documents.

3. **BM25 Scoring:**
   - The BM25 ranking formula is applied to compute a relevance score between the expanded query and each document in the inverted index:
     ```math
     BM25 = IDF * ((TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (doc_length / avg_doc_length))))
     ```
     - `k1 = 1.2`, `b = 0.75` are empirically tuned parameters.
     - TF represents the term frequency of the query term in the document.
     - IDF ensures that rare words receive higher weight.
     - Document length normalization prevents longer documents from having an unfair advantage.
   - Each document receives a final BM25 score, indicating how relevant it is to the query.

4. **Ranking Documents:**
   - The computed BM25 scores are stored in a dictionary and sorted in **descending order**.
   - The top `N` documents (default = 100) are returned as ranked results.
   - Example output format:
     ```
     1 Q0 doc123 1 0.8723 BM25+QE
     1 Q0 doc456 2 0.7654 BM25+QE
     ```
   - This format is **trec_eval-compatible**, ensuring seamless evaluation using standard IR metrics.

#### **Data Structure**
- **Query Representation:**
  - Tokenized and expanded query stored as a **list of words**.
- **Document Ranking:**
  - **Dictionary of document scores**, where keys are document IDs and values are BM25 scores.
  - Sorted list used to return top-ranked results efficiently.

---

## **Evaluation & Results**
### **Vocabulary Size**
- **Total unique terms in the corpus:** **19,767**

#### **Sample 100 Tokens**
```
['chromoendoscopi', 'meca', 'mcherri', 'subscal', 'varna', 'hind', 'glomeruli', 'gpib', 'cystogenesi', 'subtrop', 'ethiopia', 'theobroma', 'bmax', 'obes', 'aldh', 'fructo', 'orx', 'vasculogenesi', 'ucv', 'aga', 'gliotoxin', 'halothan', 'candid', 'snapshot', 'vasculitid', 'periplasm', 'costal', 'further', 'vulva', 'vanderw', 'wield', 'nonbenefici', 'falsifi', 'sensit', 'thermosens', 'gliotransmiss', 'weibel', 'kelvin', 'eupathdb', 'ascari', 'ws', 'ziprasidon', 'casp', 'androgenet', 'frontotempor', 'docetaxel', 'ehrlichia', 'underestim', 'neutral', 'transtheoret', 'interspers', 'corticotrop', 'lymphangioleiomyomatosi', 'inkt', 'recur', 'smac', 'methanogen', 'xenopu', 'ephrin', 'cushion', 'taylor', 'greenhous', 'hla', 'inspir', 'ni', 'jak', 'turner', 'misexpress', 'proposit', 'raybio', 'decam', 'mercuri', 'mellitu', 'postdisast', 'rorbeta', 'khoula', 'procedur', 'raven', 'desert', 'glutaryl', 'rhabdoid', 'ribulos', 'scholarli', 'glutaminas', 'dna', 'paradigm', 'cardin', 'tangl', 'zomba', 'reassembl', 'hemophiliac', 'perimet', 'strictli', 'compartment', 'iodo', 'nih', 'insidi', 'ac', 'ahf', 'curricula']
```

#### **First 10 Results for Queries 1 & 3**
| Query ID | Q0 | Document ID | Rank | Score  | Run Name               |
|----------|----|------------|------|--------|------------------------|
| 1        | Q0 | 21257564   | 1    | 9.7562 | BM25+QueryExpansion    |
| 1        | Q0 | 18953920   | 2    | 8.3847 | BM25+QueryExpansion    |
| 1        | Q0 | 13231899   | 3    | 7.9082 | BM25+QueryExpansion    |
| 1        | Q0 | 7581911    | 4    | 7.7017 | BM25+QueryExpansion    |
| 1        | Q0 | 20155713   | 5    | 7.5632 | BM25+QueryExpansion    |
| 1        | Q0 | 36480032   | 6    | 7.4186 | BM25+QueryExpansion    |
| 1        | Q0 | 26071782   | 7    | 7.2713 | BM25+QueryExpansion    |
| 1        | Q0 | 3566945    | 8    | 6.3707 | BM25+QueryExpansion    |
| 1        | Q0 | 1203035    | 9    | 6.2630 | BM25+QueryExpansion    |
| 1        | Q0 | 21456232   | 10   | 6.2630 | BM25+QueryExpansion    |
| 3        | Q0 | 4414547    | 1    | 31.9735 | BM25+QueryExpansion   |
| 3        | Q0 | 4378885    | 2    | 28.0432 | BM25+QueryExpansion   |
| 3        | Q0 | 2739854    | 3    | 26.8367 | BM25+QueryExpansion   |
| 3        | Q0 | 23389795   | 4    | 25.5799 | BM25+QueryExpansion   |
| 3        | Q0 | 14717500   | 5    | 25.4328 | BM25+QueryExpansion   |
| 3        | Q0 | 4632921    | 6    | 24.8410 | BM25+QueryExpansion   |
| 3        | Q0 | 13519661   | 7    | 23.6196 | BM25+QueryExpansion   |
| 3        | Q0 | 2107238    | 8    | 23.3042 | BM25+QueryExpansion   |
| 3        | Q0 | 19058822   | 9    | 22.2002 | BM25+QueryExpansion   |
| 3        | Q0 | 43334921   | 10   | 21.3640 | BM25+QueryExpansion   |

Query expansion using pseudo-relevance feedback (PRF) and WordNet improved recall but may introduce noise. The top-ranked documents for Query 1 had closely clustered scores, indicating strong relevance, while Query 3 showed high term matching being almost triple the highest match to query 1. While the system effectively retrieves relevant results, PRF and synonym weighting could be refined to reduce irrelevant matches.

### **Mean Average Precision (MAP)**
- **MAP Score:** **0.5717**
- **Precision @ 10:** **0.0833**
- **Recall @ 20:** **0.8171**
- **Recall @ 100:** **0.8850**
- **NDCG:** **0.6446**

## **References**
- BEIR Collection: [https://beir.ai/](https://beir.ai/)
- Scifact Dataset Paper: [https://arxiv.org/abs/2004.14974](https://arxiv.org/abs/2004.14974)
- TREC Eval: [https://github.com/usnistgov/trec_eval](https://github.com/usnistgov/trec_eval)
