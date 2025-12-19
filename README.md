# Russian Taxonomy Enrichment using Retrieval-Augmented Generation

## Introduction

In this project, I tackled the problem of Taxonomy Enrichment for the Russian language (ruWordNet). The challenge was to take new "orphan" words—often complex verbs or specific nouns—and correctly attach them to their parent categories (hypernyms) in an existing hierarchical graph.

When I initially analyzed the provided baselines, I noticed that static embeddings like FastText struggled significantly with polysemy. For example, the verb "run" has a similar vector to "jog" (a sibling), but without context, a vector model cannot determine if I mean "run a company" (Manage) or "run a race" (Move).

To solve this, I moved away from simple vector similarity and built a Retrieval-Augmented Generation (RAG) pipeline. My goal was to mimic human reasoning: first define the word based on its context, then search for that definition in the database, and finally use a logic-based reranker to select the best fit.

## System Architecture

I designed the pipeline to operate in four distinct stages. This modular approach allowed me to debug specific failures (like hallucination or retrieval misses) without retraining the entire system.

### 1. Definition Generation (The Semantic Anchor)
The biggest hurdle was ambiguity. To fix this, I used a Large Language Model (specifically **Vikhr-Nemo-12B**, a state-of-the-art Russian model) to act as a lexicographer. Instead of searching for the orphan word directly, I prompt the model to generate a dictionary-style definition based on the provided context sentence.

*   **Why this matters:** This collapses the context and the word into a single, unambiguous semantic description.
*   **The Fallback:** I realized that even 12B models hallucinate on extremely rare linguistic terms (e.g., specific dialect verbs). To counter this, I implemented an asynchronous scraper that checks Wiktionary for a real definition before asking the LLM.

### 2. Dense Retrieval (Recall)
Once I had a clean definition, I needed to find potential parents in the `ruWordNet` database. I encoded the generated definitions using **`intfloat/multilingual-e5-large`**.

I chose an asymmetric search strategy. I encoded the orphan's *definition* as the query and the taxonomy's *synset names* as the passages. This allowed me to retrieve the top 50 candidates, ensuring that the correct parent was almost always included in the initial pool, even if the phrasing didn't match exactly.

### 3. Cross-Encoder Reranking (Precision)
Bi-encoders are fast but "blurry"—they often rank synonyms or siblings higher than parents because they are semantically close. To filter the list, I integrated a Cross-Encoder, **`BAAI/bge-reranker-v2-m3`**.

Unlike the bi-encoder, this model reads the orphan definition and the candidate parent side-by-side. It acts as a strict judge, assigning a relevance score to each pair. This step was the single largest contributor to my improved metrics, as it effectively filtered out related but incorrect terms.

### 4. Graph Expansion (Logic)
Finally, I leveraged the structure of the taxonomy itself. Since the evaluation metric (MAP) rewards predicting ancestors (grandparents), I wrote a graph traversal algorithm. Once the model identifies the best immediate parent, I automatically append its ancestors to the submission list. This maximizes the score even if the model picks a node slightly too deep in the tree.
## Implementation Details

I organized the project into modular components rather than a single monolithic notebook. This separation of concerns made it easier to swap out models (e.g., changing the LLM from Qwen to Vikhr) without breaking the retrieval logic.

### File Structure

*   **`runner.py`**: The main entry point. It handles argument parsing, data loading, and orchestrates the flow of data between the models. I added logic here to automatically detect whether to process nouns or verbs based on the input flags.
*   **`llm_judge.py`**: This class manages the Large Language Model. It handles 4-bit loading, prompt engineering for definition generation, and the logic for the final selection step.
*   **`retriever.py`**: Wraps the `E5-Large` model. It handles the vectorization of definitions and performs the initial dense retrieval against the taxonomy.
*   **`reranker.py`**: Wraps the `BGE-M3` model. It contains the logic for pair-wise scoring of candidates.
*   **`dataloader.py`**: A utility script I wrote to parse the raw XML files from `ruWordNet` directly from the zip archive, creating a clean Pandas DataFrame for the search index.
*   **`submission_converter.py`**: A helper script that validates the output JSON and packages it into the strict TSV/ZIP format required by the evaluation platform.

### Hardware Requirements

Running this pipeline requires careful memory management. I am loading three distinct neural networks into VRAM simultaneously:
1.  **Vikhr-Nemo-12B** (LLM)
2.  **E5-Large** (Embedding Model)
3.  **BGE-M3** (Reranker)

To make this fit on a standard **15GB NVIDIA T4x2 GPU** (available on Kaggle/Colab), I utilized `bitsandbytes` to load the LLM in **4-bit NF4 precision**. Without this optimization, the 12B model alone would require over 24GB of VRAM.

## Usage

I designed `runner.py` with a command-line interface (CLI) to make it easy to switch between development (Practice phase) and final testing (Post-Evaluation phase).

### 1. Installation

First, I install the necessary libraries, specifically `bitsandbytes` and `peft` for the quantization support.

```bash
pip install -r requirements.txt
```

### 2. Running Inference

To generate predictions, I run the runner script. I included a `--batch_size` argument to control memory usage; for the 12B model, a batch size of 2 is the safe limit on a T4 GPU.

**Example: Processing Verbs for the Practice Phase**

```bash
python runner.py \
    --mode verbs \
    --phase public \
    --batch_size 12 \
    --use_cache
```

*   `--mode`: Switches between Nouns and Verbs (automatically filters the XML graph).
*   `--phase`: Switches between the Public (Practice) and Private (Evaluation) test sets.
*   `--use_cache`: If I've already generated definitions, this flag skips the slow LLM generation step and jumps straight to retrieval.

### 3. Formatting for Submission

The evaluation platform is extremely strict about file formats. I wrote a dedicated converter that takes the raw JSON output from the runner, formats it into a TSV file, and zips it with the specific filename required by the leaderboard (e.g., `verbs.tsv`).

```bash
python submission_converter.py \
    --input submission_verbs_public.json \
    --output verbs.tsv
```

This generates `submission.zip`, which is ready for upload.

## Experiments and Results

I evaluated the system using Mean Average Precision (MAP), which is the official metric for the competition. This metric is particularly unforgiving because it requires not just finding the correct parent, but ranking it highly in the list.

### Deep Dive: Solving the "Hallucination" Problem

One of the most interesting challenges I encountered was with the verb *«акать»* (akat').

In my early experiments using **Qwen-7B** without definitions, the model consistently predicted parents related to **"Transport"** or **"Vehicles"**. This was baffling until I realized the model was tokenizing *«акать»* and hallucinating a connection to "A Car" or "A Cat" due to a lack of Russian cultural context.

I solved this by switching to **Vikhr-Nemo-12B** (a native Russian model) and forcing it to generate a dictionary definition first.
*   **Before:** Prediction: *Vehicle*.
*   **After Definition:** The model generated *"A dialect feature where 'O' is pronounced as 'A'."*
*   **Result:** The retriever immediately found the correct parent: *«Speaking / Pronouncing»*.

This proved that for rare vocabulary, **contextual definition** is a far more powerful search query than the word embedding alone.

## Alternative Approach: The Fine-Tuning Experiment

Before settling on the RAG architecture, I attempted to solve this task by **Supervised Fine-Tuning (SFT)**. I wanted to see if I could "bake" the taxonomy directly into the model's weights.

I set up a training pipeline using **QLoRA** to fine-tune **Qwen-2.5-7B-Instruct**. I converted the training data into a ChatML format, masking the user prompts so the loss was calculated only on the predicted Synset IDs. I trained this on a T4 GPU for several epochs.

## Conclusion

This project demonstrated that complex semantic tasks like Taxonomy Enrichment cannot be solved by simple vector similarity alone. By decomposing the problem into **Definition**, **Retrieval**, and **Reranking**, I was able to build a system that doesn't just guess relationships based on word proximity, but actually "understands" the semantic link between a specific instance and its abstract category.

Using **4-bit quantization** was the key engineering enabler, allowing me to deploy a pipeline that would normally require enterprise-grade hardware on a standard research GPU.
