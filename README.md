# Fine-Tuning LLaMA 3.2 Instruct for Medical QA with RAG and Evaluation

## This notebook demonstrates 4 flows:

### 1. Base Model
### 2. QLoRA Fine-tuned Model
### 3. RAG + Base Model
### 4. RAG + Fine-tuned Model
### 5. Evaluation is done using BERTScore on unseen data.


## Compute & Environment Constraints (Important Note)

1. All experiments were conducted using Google Colab, which imposes strict compute limitations:
2. GPU availability is limited to ~4 hours per session
3. Once the GPU quota is exhausted, access may be restricted for up to 48 hours
4. Due to these constraints, the fine-tuning strategy was intentionally conservative and small dataset (11000 rows)


## Flow 1 - Base Model

1. Used LLaMA-3.2-1B-Instruct as the baseline model.
2. No additional training was performed.
3. Serves as a strong reference point to evaluate the impact of fine-tuning and RAG.


## Flow 2 - Fine-Tuning with QLoRA
Applied QLoRA on ~10,000 medical Q&A samples.
Used conservative hyperparameters:

1. Low LoRA rank
2. Single training epoch
3. Small learning rate

This ensures:

1. Minimal memory usage
2. Safe behavioral adaptation rather than aggressive retraining
3. Fine-tuning focuses on response style and domain alignment, not memorization of facts.



## Flow 3 - Rag with base model
Built a semantic retriever using: Sentence-BERT embeddings (all-MiniLM-L6-v2)
FAISS vector index
Stored clean question–answer documents as the knowledge base.
At inference time:

1. Relevant documents are retrieved
2. Injected into the prompt
3. The model is instructed to answer using retrieved context only




## Flow 4 - RAG + Fine-Tuning
Combined both techniques:
1. Fine-tuned model for better instruction following
2. External retrieval for factual grounding
3. Evaluated carefully to observe interaction effects and failure modes.



## Evaluation Strategy (Bert Score)

1. Automated Evaluation -Used BERTScore (F1) to measure semantic similarity between generated answers and reference answers.
Suitable for generative tasks where multiple valid phrasings exist.

2. Refusal-Aware Scoring

Explicitly detected refusal responses such as:
1. “Answer not found in context”
2. Refusal cases were excluded from semantic scoring to avoid misleading results.

This ensures honest evaluation in RAG scenarios.
🔹 Aggregate Evaluation (100 Queries)

# Model Avg BERTScore (F1)

## Base Model - 0.82
## Fine-tuned Model - 0.83
## RAG Base Model - 0.81
## RAG + Fine-tuned - 0.81


## Interpretation:

1. Fine-tuning provides a small but consistent improvement over the base model.
2. RAG does not always improve BERTScore because:
3. RAG prioritizes factual grounding
4. BERTScore measures semantic similarity, not correctness
