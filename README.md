# IMDB Sentiment Analysis — Part 1 (Classical NLP)

Short intro
This repository contains a comparative sentiment-analysis project on the IMDB Large Movie Review dataset. Part 1 implements a classical NLP pipeline (data cleaning, tokenization, lemmatization/stemming, vectorization) and trains a logistic regression classifier. Part 2 will use HuggingFace transformers for comparison.

Notebook overview
The Colab notebook `nlp-basics-imdb-movie-review-sentiment.ipynb` performs the following sequence:

1. Data loading and initial inspection
2. Data cleaning
   - Drop duplicates
   - Remove punctuation and HTML-like tokens
   - Lowercasing and stopword removal
3. Tokenization and basic statistics
4. Padding tokens to a fixed MAX_LEN
5. Normalization
   - Lemmatization (WordNetLemmatizer)
   - Stemming (PorterStemmer)
6. POS tagging for exploratory analysis
7. Word embeddings
   - Build vocabulary from lemmatized tokens
   - Load GloVe (fallback to random init for missing tokens)
8. Vectorization
   - Bag-of-words via CountVectorizer
9. Label encoding
   - Binary mapping: positive -> 1, negative -> 0
10. Model training and evaluation
    - Train/test split
    - Logistic Regression model (liblinear)
    - Report accuracy and F1 score
    Mean Accuracy: 0.8672941176470588
    F1 Score: 0.8689938943456332
11. Inference examples and wordcloud visualization

Details
- Cleaning: removes duplicates, punctuation, and common stopwords to reduce noise.
- Tokenization & padding: tokenizes with NLTK and pads/truncates to a dataset-derived MAX_LEN.
- Normalization: both lemmatized and stemmed token sets are generated for comparison.
- Embeddings: GloVe 200d is read and aligned to the dataset vocabulary; missing vectors use random initialization.
- Vectorization: CountVectorizer builds sparse feature matrix used for classical classification.
- Model & metrics: logistic regression is used as a baseline; evaluation includes accuracy and F1.

Outputs & artifacts
- Trained LogisticRegression model and evaluation metrics (accuracy, F1).
Mean Accuracy: 0.8672941176470588
F1 Score: 0.8689938943456332
- Example inference predictions for sample sentences.
- Wordcloud of the corpus and printed sample embeddings.

Reproducibility / requirements
- Dataset: expects `../datas/IMDB Dataset.csv` relative to the notebook.
- GloVe: expects `../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt` (or update path).
- Key Python packages: numpy, pandas, scikit-learn, nltk, matplotlib, wordcloud, regex.
- NLTK downloads used in notebook: `omw-1.4` (and standard tokenizers/wordnet as needed).

Next steps (Part 2)
Part 2 will implement a transformer-based approach (HuggingFace) for fine-tuning a pretrained model on the same dataset and compare performance and inference behavior against this classical baseline.

## Part 2 — Transformer-based approach (imdb-sentiment-transformer.ipynb)

Notebook overview
The transformer notebook fine-tunes a pretrained HuggingFace model on the IMDB dataset and compares results to the classical baseline.

Key steps
1. Environment and checks
   - Install evaluate, import datasets, transformers, torch.
   - Detect and report GPU availability.
2. Data loading & sampling
   - Load the IMDB dataset via datasets.load_dataset.
   - Optionally sample/shuffle subsets for faster experiments.
3. Tokenization & preprocessing
   - Use AutoTokenizer (e.g., distilbert-base-uncased).
   - Tokenize with truncation and padding to a fixed max_length.
4. Data collator
   - Use DataCollatorWithPadding to batch variable-length inputs.
5. Model
   - Load AutoModelForSequenceClassification with num_labels=2.
6. Training setup
   - Define TrainingArguments (output_dir, lr, batch sizes, epochs, eval/save strategy, load_best_model_at_end).
   - Use Trainer with compute_metrics that loads evaluate.metrics (accuracy, f1).
7. Training & evaluation
   - Train the model and evaluate per epoch; report accuracy and F1.
   Final Performance:
   1st version:
      Accuracy: 86.4%
      F1 Score: 0.863
      Training Loss: 0.227
      Validation loss increased slightly in the final epoch, indicating early signs of overfitting.

   2nd Version:
      Accuracy: 89.9%
      F1 Score: 0.9
      Training Loss: 0.228
8. Inference
   - Use TextClassificationPipeline for quick predictions on sample sentences.

Notes & reproducibility
- Checkpoint used in the notebook: distilbert-base-uncased (changeable).
- Typical hyperparameters: lr ~1e-4, batch_size 32, epochs 2–3 (adjust for GPU/CPU).
- Requirements: transformers, datasets, evaluate, torch, numpy.
- Running on CPU is possible but slower; GPU recommended for full runs.

Outputs & comparison
- Trained transformer model and per-epoch metrics (accuracy, F1).
- Inference examples from pipeline.

Note: In the initial transformer experiment, the model underperformed relative to the classical Logistic Regression baseline. The transformer achieved an accuracy of 86.4% and an F1 score of 0.863, which was slightly lower than the classical NLP model (Accuracy: 0.8673, F1: 0.8690). Upon analysis, this weaker performance was attributed to suboptimal hyperparameters—specifically, a relatively high learning rate of 0.0001 and a shorter maximum sequence length of 132 tokens.
To address this, the learning rate was reduced to 2e-5, which aligns with standard fine-tuning practices for BERT-based models, and the maximum sequence length was increased to 256 tokens to better capture the longer contextual structure of IMDb reviews.
Following these adjustments, the transformer’s performance improved significantly. The model achieved a final accuracy of 89.9% and an F1 score of 0.9000, clearly surpassing the classical baseline. Additionally, validation loss remained stable across epochs, indicating improved generalization and reduced overfitting compared to the initial run.

Final thoughts:
This experiment demonstrates the sensitivity of transformer models to hyperparameter configuration and highlights the importance of proper fine-tuning. When appropriately configured, transformer-based models substantially outperform classical NLP methods due to their contextual representation capabilities.


Further Improvements & Experimental Directions
Although performance improved considerably, there are several avenues to further enhance the model:
   -Weight Decay (Regularization):
      Adding weight decay (e.g., weight_decay=0.01) can improve generalization by preventing the model from overfitting large weights during training.
   -Learning Rate Scheduling:
      Using a linear scheduler with warmup steps can stabilize early training and improve convergence.
   -Early Stopping:
      Monitoring validation loss and stopping training when performance plateaus can prevent overfitting.
   -Larger Model Variants:
      Experimenting with stronger architectures such as bert-base-uncased or roberta-base may yield further improvements.
   -Full Dataset Training:
      Training on the entire 25,000-sample training split (instead of subsets) can provide richer learning signals and improve generalization.
   -Gradient Accumulation / Larger Effective Batch Sizes:
      Increasing the effective batch size can stabilize optimization dynamics.
   -Hyperparameter Search:
      Systematic tuning of learning rate (e.g., 1e-5 to 5e-5), batch size, and epochs can help identify optimal configurations.