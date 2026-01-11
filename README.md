Fine-Tuning LLaMA 3.2 Instruct for Medical QA with RAG and Evaluation

This notebook demonstrates 4 flows:

Base Model
QLoRA Fine-tuned Model
RAG + Base Model
RAG + Fine-tuned Model
Evaluation is done using BERTScore on unseen data.


Compute & Environment Constraints (Important Note)
All experiments were conducted using Google Colab, which imposes strict compute limitations:

GPU availability is limited to ~4 hours per session

Once the GPU quota is exhausted, access may be restricted for up to 48 hours

Due to these constraints, the fine-tuning strategy was intentionally conservative and small dataset (11000 rows)


Flow 1 - Base Model
Used LLaMA-3.2-1B-Instruct as the baseline model.

No additional training was performed.

Serves as a strong reference point to evaluate the impact of fine-tuning and RAG.



Flow 2 - Fine-Tuning with QLoRA
Applied QLoRA on ~10,000 medical Q&A samples.

Used conservative hyperparameters:

Low LoRA rank
Single training epoch
Small learning rate
This ensures:

Minimal memory usage
Safe behavioral adaptation rather than aggressive retraining
Fine-tuning focuses on response style and domain alignment, not memorization of facts.



Flow 3 - Rag with base model
Built a semantic retriever using: Sentence-BERT embeddings (all-MiniLM-L6-v2)
FAISS vector index
Stored clean question–answer documents as the knowledge base.
At inference time:

Relevant documents are retrieved
Injected into the prompt
The model is instructed to answer using retrieved context only




Flow 4 - RAG + Fine-Tuning
Combined both techniques:

Fine-tuned model for better instruction following
External retrieval for factual grounding
Evaluated carefully to observe interaction effects and failure modes.



Evaluation Strategy (Bert Score)
🔹 Automated Evaluation

Used BERTScore (F1) to measure semantic similarity between generated answers and reference answers.
Suitable for generative tasks where multiple valid phrasings exist.
🔹 Refusal-Aware Scoring

Explicitly detected refusal responses such as:

“Answer not found in context”
Refusal cases were excluded from semantic scoring to avoid misleading results.
This ensures honest evaluation in RAG scenarios.
🔹 Aggregate Evaluation (100 Queries)

Model Avg BERTScore (F1)

Base Model - 0.82

Fine-tuned Model - 0.83

RAG Base Model - 0.81

RAG + Fine-tuned - 0.81

Interpretation:

Fine-tuning provides a small but consistent improvement over the base model.
RAG does not always improve BERTScore because:
RAG prioritizes factual grounding
BERTScore measures semantic similarity, not correctness
