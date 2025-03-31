# CSI4107 - Winter 2025
## Assignment 2: Information Retrieval System
**Due: March 30, 10 PM**

## **Group Members**
- **Jack Snelgrove** - 300247435
- **Lina Moussadek** - 300259985
- **Eli Wynn** - 300248135

### **Task Division**
- **Bert** Jack Snelgrove
- **LLM** Lina Moussadeck
- **Evaluation & Report Writing:** Lina, Jack and Eli
- **README writeup:** Eli Wynn

---

## **Project Overview**
This project implements an **Information Retrieval (IR) system** using a **doc2vec reranking system**, a **Mini Language Model** and the model from the previous assignment which used **BM25+Query Expansion**.

### **Functional Overview**
1. **Preprocessing:** Tokenization, stopword removal, and stemming.
2. **Indexing:** Construction of an inverted index with TF-IDF weighting.
3. **Retrieval & Ranking:** BM25-based retrieval and query expansion using **WordNet synonyms** and **pseudo-relevance feedback**.
4. **Doc2Vec Reranking:**
   - **Doc2Vec Model:** Trained on the corpus using the `Doc2Vec` model from the `gensim` library.
   - **Reranking:** Re-ranks the retrieved documents based on their similarity to the query using the trained Doc2Vec model.
5. **Mini Language Model (MiniLM) Reranking:**
   - **MiniLM Model:** Utilizes the efficient `all-MiniLM-L6-v2` transformer model from SentenceTransformers.
   - **Neural Reranking:** Computes semantic similarity between queries and documents using dense vector representations.
   - **Batch Processing:** Efficiently processes document batches for improved performance.
6. **Evaluation:**
   - **Evaluation Metrics:** Precision, Recall, F1-score, and Average Precision (AP).
   - **Evaluation Script:** `evaluate.py`
   - **Evaluation Results:** Comparative analysis of all three retrieval methods.

The system produces a **ranked list of documents** for each query and outputs results in a **trec_eval-compatible format**.

---

## **Installation & Running the Code**
### **Dependencies**
- Python 3.8+
- Required Python Libraries:
  ```bash
  pip install nltk jsonlines pytrec_eval gensim
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
   python3 pipeline_runner.py
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

4. **Doc2Vec Reranking Step:**
   - **Training the Model:**
     ```bash
     python doc2vec_reranking.py
     ```
    - **Input:** `preprocessed_corpus.json`, `results.txt`
    - **Output:** `doc2vec_results.txt`
   ```

5. **Mini Language Model (LLM) Step:**
   ```bash
   neural_rerank_minilm.py
   ```
   - **Input:** `preprocessed_corpus.json`, `results.txt`
   - **Output:** `Results_neural_minilm.txt`

6. **Evaluation Step (retrievalAndRanking):**
   - **Evaluation Script:** `evaluate.py`
   - **Input:** `Results.txt`
   - **Output:** Evaluation metrics (Precision, Recall, F1-score, and Average Precision (AP)).

7. **Evaluation Step (Full Pipeline):**
   - **Evaluation Script:** 'pipeline_runner.py'
   - **Output:** Evaluation metrics (Precision, Recall, F1-score, and Average Precision (AP)) for all 3 retrieval techniques.
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
   - **Purpose:** Reads a **JSONL-formatted corpus**, processes each document's title and text, and outputs tokenized data to a file.
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
   - The user's input query is first preprocessed using `preprocess_text()` from `preprocess.py`. This ensures consistency between document indexing and query representation.
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
### **Step 4: Doc2Vec Reranking**
#### **Algorithm**
The Doc2Vec reranking system is designed to enhance the relevance of search results by considering the semantic similarity between documents and queries. This approach aims to improve the accuracy of information retrieval by considering the context and meaning of documents that term-based methods might miss.
1. **Document Embedding Generation:**
   - Uses Gensim's Doc2Vec model to create vector representations of documents.
   - Each document is represented as a dense vector in a high-dimensional space.
   - Documents with similar semantic content are positioned closer in this vector space.
2. **Query Embedding:**
   - Transforms the query into the same vector space as the documents.
   - This allows for direct comparison between query intent and document content.
3. **Similarity Calculation:**
   - Computes cosine similarity between the query vector and document vectors.
   - Higher similarity scores indicate stronger semantic relevance.
4. **Hybrid Ranking:**
   - Combines the original BM25 scores with Doc2Vec similarity scores.
   - A weighted approach balances lexical matching (BM25) with semantic matching (Doc2Vec).
   - The final ranking reflects both exact term matches and conceptual relevance. Data Structure

#### **Data Structure**   
- **Document Vectors:**
  - Dense numerical arrays representing semantic content.
- **Similarity Matrix:**
  - Stores cosine similarity scores between query and documents.

---

### **Step 5: MiniLM Neural Reranking**
#### **Algorithm**
The MiniLM neural reranking component leverages transformer-based language models to capture deep semantic relationships between queries and documents. This approach goes beyond traditional lexical matching and Doc2Vec by utilizing contextualized embeddings from a pre-trained language model.

1. **Model Initialization:**
   - Loads the `all-MiniLM-L6-v2` model, a lightweight and efficient transformer model from the SentenceTransformers library.
   - This model is a distilled version of BERT, offering a good balance between performance and efficiency.
   - Example:
     ```python
     model = SentenceTransformer('all-MiniLM-L6-v2')
     ```

2. **Document and Query Encoding:**
   - Transforms both queries and documents into dense vector representations in a high-dimensional semantic space.
   - Uses the model's encoding capabilities to capture contextual meaning:
     ```python
     query_embedding = model.encode(query, convert_to_tensor=True)
     doc_embeddings = model.encode(doc_texts, convert_to_tensor=True, batch_size=16)
     ```
   - **Batch Processing:** Processes documents in batches of 16 for improved efficiency.

3. **Semantic Similarity Calculation:**
   - Computes cosine similarity between the query embedding and each document embedding.
   - This measures the semantic closeness of documents to the query in the embedding space.
   - Higher similarity scores indicate stronger semantic relevance.
   - Example:
     ```python
     similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
     ```

4. **Selective Reranking:**
   - For efficiency, only reranks the top-K documents (default K=25) from the initial BM25 results.
   - This hybrid approach combines the efficiency of traditional retrieval with the effectiveness of neural reranking.
   - The approach significantly reduces computational overhead while maintaining high-quality results.

5. **Result Generation:**
   - Sorts documents by their semantic similarity scores.
   - Outputs the reranked results in TREC format for evaluation.
   - Example output format:
     ```
     1 Q0 doc123 1 0.9876 MiniLM
     1 Q0 doc456 2 0.8765 MiniLM
     ```

#### **Data Structure**
- **Document Embeddings:**
  - Dense vectors (typically 384-dimensional for MiniLM-L6) representing the semantic content of documents.
- **Query Embeddings:**
  - Dense vectors in the same semantic space as document embeddings.
- **Similarity Matrix:**
  - Stores cosine similarity scores between query and document embeddings.

#### **Optimizations**
- **Selective Reranking:** Only reranks the top-K documents from BM25 results to balance effectiveness and efficiency.
- **Batch Processing:** Encodes documents in batches to leverage GPU parallelism and reduce processing time.
- **Tensor Operations:** Uses PyTorch tensor operations for efficient similarity calculations.
- **Pre-trained Model:** Leverages a distilled transformer model that offers a good balance between size and performance.

---

## **Evaluation & Results**
### **Vocabulary Size**
- **Total unique terms in the corpus:** **19,767**

#### **Sample 100 Tokens**
```
['chromoendoscopi', 'meca', 'mcherri', 'subscal', 'varna', 'hind', 'glomeruli', 'gpib', 'cystogenesi', 'subtrop', 'ethiopia', 'theobroma', 'bmax', 'obes', 'aldh', 'fructo', 'orx', 'vasculogenesi', 'ucv', 'aga', 'gliotoxin', 'halothan', 'candid', 'snapshot', 'vasculitid', 'periplasm', 'costal', 'further', 'vulva', 'vanderw', 'wield', 'nonbenefici', 'falsifi', 'sensit', 'thermosens', 'gliotransmiss', 'weibel', 'kelvin', 'eupathdb', 'ascari', 'ws', 'ziprasidon', 'casp', 'androgenet', 'frontotempor', 'docetaxel', 'ehrlichia', 'underestim', 'neutral', 'transtheoret', 'interspers', 'corticotrop', 'lymphangioleiomyomatosi', 'inkt', 'recur', 'smac', 'methanogen', 'xenopu', 'ephrin', 'cushion', 'taylor', 'greenhous', 'hla', 'inspir', 'ni', 'jak', 'turner', 'misexpress', 'proposit', 'raybio', 'decam', 'mercuri', 'mellitu', 'postdisast', 'rorbeta', 'khoula', 'procedur', 'raven', 'desert', 'glutaryl', 'rhabdoid', 'ribulos', 'scholarli', 'glutaminas', 'dna', 'paradigm', 'cardin', 'tangl', 'zomba', 'reassembl', 'hemophiliac', 'perimet', 'strictli', 'compartment', 'iodo', 'nih', 'insidi', 'ac', 'ahf', 'curricula']
```

#### **First 10 Results for Queries 1 & 3**

**BM25+QueryExpansion Results:**
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

**Doc2Vec Results:**
| Query ID | Q0 | Document ID | Rank | Score  | Run Name |
|----------|----|------------|------|--------|----------|
| 1        | Q0 | 21257564   | 1    | 0.7497 | Doc2Vec  |
| 1        | Q0 | 18953920   | 2    | 0.7005 | Doc2Vec  |
| 1        | Q0 | 36480032   | 3    | 0.6286 | Doc2Vec  |
| 1        | Q0 | 7581911    | 4    | 0.6008 | Doc2Vec  |
| 1        | Q0 | 21456232   | 5    | 0.5360 | Doc2Vec  |
| 1        | Q0 | 37949139   | 6    | 0.5313 | Doc2Vec  |
| 1        | Q0 | 4435369    | 7    | 0.5171 | Doc2Vec  |
| 1        | Q0 | 17388232   | 8    | 0.5003 | Doc2Vec  |
| 1        | Q0 | 3845894    | 9    | 0.5001 | Doc2Vec  |
| 1        | Q0 | 20155713   | 10   | 0.4869 | Doc2Vec  |
| 3        | Q0 | 2739854    | 1    | 0.8229 | Doc2Vec  |
| 3        | Q0 | 4414547    | 2    | 0.7817 | Doc2Vec  |
| 3        | Q0 | 4632921    | 3    | 0.7750 | Doc2Vec  |
| 3        | Q0 | 23389795   | 4    | 0.7042 | Doc2Vec  |
| 3        | Q0 | 4378885    | 5    | 0.6943 | Doc2Vec  |
| 3        | Q0 | 1067605    | 6    | 0.6433 | Doc2Vec  |
| 3        | Q0 | 2107238    | 7    | 0.6133 | Doc2Vec  |
| 3        | Q0 | 461550     | 8    | 0.5633 | Doc2Vec  |
| 3        | Q0 | 13519661   | 9    | 0.5585 | Doc2Vec  |
| 3        | Q0 | 41782935   | 10   | 0.5443 | Doc2Vec  |

**MiniLM Neural Reranking Results:**
| Query ID | Q0 | Document ID | Rank | Score  | Run Name |
|----------|----|------------|------|--------|----------|
| 0        | Q0 | 4435369    | 1    | 0.1812 | MiniLM   |
| 0        | Q0 | 825728     | 2    | 0.1636 | MiniLM   |
| 0        | Q0 | 7581911    | 3    | 0.1587 | MiniLM   |
| 0        | Q0 | 18953920   | 4    | 0.1281 | MiniLM   |
| 0        | Q0 | 11335860   | 5    | 0.1219 | MiniLM   |
| 0        | Q0 | 20155713   | 6    | 0.1191 | MiniLM   |
| 0        | Q0 | 3566945    | 7    | 0.1059 | MiniLM   |
| 0        | Q0 | 2566674    | 8    | 0.0963 | MiniLM   |
| 0        | Q0 | 23244529   | 9    | 0.0949 | MiniLM   |
| 0        | Q0 | 13231899   | 10   | 0.0928 | MiniLM   |
| 2        | Q0 | 13734012   | 1    | 0.4169 | MiniLM   |
| 2        | Q0 | 18617259   | 2    | 0.3365 | MiniLM   |
| 2        | Q0 | 76415938   | 3    | 0.3033 | MiniLM   |
| 2        | Q0 | 11880289   | 4    | 0.2535 | MiniLM   |
| 2        | Q0 | 1292369    | 5    | 0.2523 | MiniLM   |
| 2        | Q0 | 103007     | 6    | 0.2435 | MiniLM   |
| 2        | Q0 | 17333231   | 7    | 0.2367 | MiniLM   |
| 2        | Q0 | 18340282   | 8    | 0.2341 | MiniLM   |
| 2        | Q0 | 19140422   | 9    | 0.2334 | MiniLM   |
| 2        | Q0 | 4828631    | 10   | 0.2236 | MiniLM   |

### **Performance Comparison**

| Metric       | BM25+QueryExpansion | Doc2Vec | MiniLM Neural Reranking |
|--------------|---------------------|---------|-------------------------|
| NDCG         | 0.6446              | 0.6379  | 0.6601                  |
| P_10         | 0.0833              | 0.0820  | 0.0897                  |
| MAP          | 0.5717              | 0.5621  | 0.6029                  |
| Recall_20    | 0.8171              | 0.8049  | 0.8216                  |
| Recall_100   | 0.8850              | 0.8850  | 0.8232                  |

### **Discussion of Results**

Our evaluation reveals several interesting patterns in the performance of all three retrieval methods:

1. **Comparison of Approaches:**
   - **MiniLM Neural Reranking** achieved the best overall performance with a MAP of **0.6029**, outperforming both BM25+QueryExpansion (0.5717) and Doc2Vec (0.5621).
   - The NDCG scores show that MiniLM (0.6601) places relevant documents at better positions in the ranking compared to BM25 (0.6446) and Doc2Vec (0.6379).
   - While BM25 and Doc2Vec achieved identical Recall@100 (0.8850), MiniLM showed a slightly lower recall (0.8232), suggesting it might miss some relevant documents that the other methods find.

2. **Score Distribution:**
   - The scoring mechanisms differ significantly between methods:
     - BM25 scores range from ~6 to ~32
     - Doc2Vec scores range from ~0.48 to ~0.82
     - MiniLM scores vary widely, with some even being negative, ranging from -0.0736 to 0.7991

3. **Ranking Differences:**
   - For Query 1 (shown as Query 0 in MiniLM results), MiniLM ranks document 4435369 first, which appears as 7th in Doc2Vec results but doesn't appear in BM25's top 10.
   - This suggests that MiniLM captures semantic relationships that aren't evident in the lexical matching of BM25.
   - The neural methods show different ranking patterns, indicating they're capturing different aspects of semantic relevance.

4. **Performance Metrics:**
   - **Precision@10** is highest for MiniLM (0.0897) compared to BM25 (0.0833) and Doc2Vec (0.0820), suggesting that neural reranking places more relevant documents in the top 10 results.
   - **Recall@20** is highest for MiniLM (0.8216), followed closely by BM25 (0.8171) and then Doc2Vec (0.8049).
   - The lower Recall@100 for MiniLM suggests a potential trade-off: while it ranks the most relevant documents higher, it might miss some marginally relevant documents that lexical methods can find.

5. **Efficiency vs. Effectiveness:**
   - While MiniLM provides the best performance in terms of precision and ranking quality, the difference in recall@100 suggests that a hybrid approach combining the strengths of both neural and lexical methods might yield the best overall performance.
   - The selective reranking approach of MiniLM demonstrates that neural methods can significantly improve ranking quality even when applied to a subset of the initial results.

In conclusion, the MiniLM neural reranking approach demonstrates superior performance in most evaluation metrics, showing the value of contextualized embeddings for capturing semantic relationships between queries and documents. However, the slightly lower recall@100 suggests that a hybrid approach combining neural reranking with traditional lexical methods might provide the most comprehensive retrieval performance.

# A2 Information Retrieval Pipeline

This repository contains the implementation of an information retrieval system that combines traditional BM25 ranking with neural reranking approaches (Doc2Vec and MiniLM).

## Project Structure

```
.
├── A1_BM25/                    # Assignment 1: BM25 Implementation
│   ├── scripts/               # Python scripts for BM25
│   └── output/                # Output files from BM25
├── A2_Neural/                 # Assignment 2: Neural Reranking
│   ├── doc2vec/              # Doc2Vec implementation
│   └── minilm/               # MiniLM implementation
├── data/                      # Data files
│   └── scifact/              # SciFact dataset
└── pipeline/                  # Pipeline scripts
    └── run_A1+A2_pipeline.py # Main pipeline script
```

## Example Pipeline Output

When you run the pipeline, you'll see output similar to this:

```
==================================================
🚀 A2 Information Retrieval Pipeline
==================================================

What do you want to run?
1. Run A1 (BM25) + A2 (doc2vec & LLM)
2. Run only A2 (doc2vec & LLM)
Enter your choice: 1

==================================================
📊 Running Assignment 1 (BM25)
==================================================

▶️ Preprocessing corpus...
✅ Corpus preprocessed
ℹ️ Output: A1_BM25/output/preprocessed_corpus.json

▶️ Building inverted index...
✅ Inverted index created
ℹ️ Output: A1_BM25/output/invertedIndex.json

▶️ Running retrieval and ranking...
✅ Retrieval and ranking completed
ℹ️ Output: A1_BM25/output/Results.txt

▶️ Evaluating results...
✅ A1 evaluation completed
ℹ️ Output: A1_BM25/output/evaluation_summary.txt

==================================================
📊 Running Assignment 2 (Neural Reranking)
==================================================

▶️ Running doc2vec reranker...
✅ Doc2vec reranking completed
ℹ️ Output: A2_Neural/doc2vec/output/Results_doc2vec.txt
ℹ️ Evaluation: A2_Neural/doc2vec/output/evaluation_results_doc2vec.txt

Choose which LLM reranker to run:
1. MiniLM (FAST – top 25 docs)
2. MiniLM (FULL – top 100 docs)
Enter your choice: 1

⚠️ WARNING: You have chosen the FAST version (top 25 docs)
⚠️ This version may have lower Recall@100 and Precision compared to the FULL version
⚠️ because it reranks fewer documents. Consider using the FULL version
⚠️ for better recall and precision metrics.

ℹ️ Running FAST version (top 25 docs)
▶️ Running MiniLM reranker...
✅ MiniLM reranking completed
ℹ️ Output: A2_Neural/minilm/output/Results_minilm.txt
ℹ️ Evaluation: A2_Neural/minilm/output/evaluation_results_minilm.txt

==================================================
📊 Final Results
==================================================

+------------------+--------+--------+------------+-------------+--------+
| System           | MAP    | P@10   | Recall@20  | Recall@100  | NDCG   |
+==================+========+========+============+=============+========+
| A1               | 0.5717 | 0.0000 | 0.8171     | 0.8850      | 0.6446 |
+------------------+--------+--------+------------+-------------+--------+
| DOC2VEC          | 0.5488 | 0.0000 | 0.8337     | 0.8337      | 0.6203 |
+------------------+--------+--------+------------+-------------+--------+
| MINILM           | 0.6145 | 0.0000 | 0.8337     | 0.8337      | 0.6721 |
+------------------+--------+--------+------------+-------------+--------+
| DOC2VEC IMPROV.  | -4.0%  | +0.0%  | +2.0%      | -5.8%       | -3.8%  |
+------------------+--------+--------+------------+-------------+--------+
| MINILM IMPROV.   | +7.5%  | +0.0%  | +2.0%      | -5.8%       | +4.3%  |
+------------------+--------+--------+------------+-------------+--------+

==================================================
📊 Pipeline Complete
==================================================
✅ All tasks completed successfully!
ℹ️ All output files have been saved in their respective directories
```

## Output Files

The pipeline generates several output files:

### A1_BM25 Outputs
- `preprocessed_corpus.json`: Preprocessed version of the corpus
- `invertedIndex.json`: Inverted index for BM25 retrieval
- `Results.txt`: BM25 retrieval results
- `evaluation_summary.txt`: Evaluation metrics for BM25

### A2_Neural Outputs
#### Doc2Vec
- `Results_doc2vec.txt`: Doc2Vec reranking results
- `evaluation_results_doc2vec.txt`: Evaluation metrics for Doc2Vec

#### MiniLM
- `Results_minilm.txt`: MiniLM reranking results
- `evaluation_results_minilm.txt`: Evaluation metrics for MiniLM

## Usage

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
   cd pipeline
   python3 run_A1+A2_pipeline.py
   ```

## Notes

- The pipeline offers two options for MiniLM reranking:
  - FAST version (top 25 docs): Faster but may have lower recall
  - FULL version (top 100 docs): Slower but better recall and precision
- All output files are saved in their respective directories under `output/`
- The final results table shows improvements over the baseline BM25 system
