# IMDB Sentiment Analysis

A comparative sentiment analysis project on the IMDB Large Movie Review dataset, implementing both classical NLP and transformer-based approaches.

## Part 1: Classical NLP Pipeline

### Overview
Implementation of a traditional NLP pipeline using logistic regression for sentiment classification.

**Notebook:** `nlp-basics-imdb-movie-review-sentiment.ipynb`

### Pipeline Steps

1. **Data Loading and Inspection**
   - Load IMDB dataset from CSV

2. **Data Cleaning**
   - Remove duplicates
   - Strip punctuation and HTML tokens
   - Lowercase and remove stopwords

3. **Tokenization and Statistics**
   - Tokenize text using NLTK
   - Pad/truncate to fixed MAX_LEN

4. **Normalization**
   - Lemmatization using WordNetLemmatizer
   - Stemming using PorterStemmer

5. **Feature Engineering**
   - POS tagging for analysis
   - GloVe embeddings (200d) with random initialization for missing tokens
   - Bag-of-words via CountVectorizer

6. **Model Training**
   - Binary label encoding (positive=1, negative=0)
   - Logistic regression with liblinear solver
   - Train/test split evaluation

### Results

```
Mean Accuracy: 0.8673
F1 Score: 0.8690
```

### Requirements

- Dataset: `../datas/IMDB Dataset.csv`
- GloVe: `../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt`
- Python packages: numpy, pandas, scikit-learn, nltk, matplotlib, wordcloud, regex
- NLTK data: omw-1.4, tokenizers, wordnet

## Part 2: Transformer-Based Approach

### Overview
Fine-tuning a pretrained transformer model using HuggingFace for comparison with the classical baseline.

**Notebook:** `imdb-sentiment-transformer.ipynb`

### Pipeline Steps

1. **Environment Setup**
   - Install transformers, datasets, evaluate
   - GPU detection and configuration

2. **Data Preparation**
   - Load via `datasets.load_dataset`
   - Optional sampling for experimentation

3. **Tokenization**
   - AutoTokenizer (distilbert-base-uncased)
   - Truncation and padding to max_length

4. **Model Configuration**
   - AutoModelForSequenceClassification (num_labels=2)
   - DataCollatorWithPadding for batching

5. **Training**
   - TrainingArguments with evaluation strategy
   - Trainer with accuracy and F1 metrics

### Results

**Initial Run (Suboptimal Hyperparameters):**
```
Learning Rate: 1e-4
Max Length: 132 tokens
Accuracy: 0.864
F1 Score: 0.863
Training Loss: 0.227
Note: Validation loss increased in final epoch (early overfitting)
```

**Optimized Run:**
```
Learning Rate: 2e-5
Max Length: 256 tokens
Accuracy: 0.899
F1 Score: 0.900
Training Loss: 0.228
```

### Key Findings

The transformer model initially underperformed the classical baseline due to high learning rate and short sequence length. After optimization:
- Reduced learning rate to 2e-5 (standard for BERT fine-tuning)
- Increased max length to 256 tokens (better context capture)
- Achieved 3.2% accuracy improvement over classical approach
- Stable validation loss indicating better generalization

### Requirements

- Model: distilbert-base-uncased (configurable)
- Python packages: transformers, datasets, evaluate, torch, numpy
- Hardware: GPU recommended (CPU compatible but slower)

## Comparison Summary

| Approach | Accuracy | F1 Score |
|----------|----------|----------|
| Classical (Logistic Regression) | 0.8673 | 0.8690 |
| Transformer (Initial) | 0.864 | 0.863 |
| Transformer (Optimized) | 0.899 | 0.900 |

## Future Improvements

- **Regularization:** Add weight decay (0.01) to prevent overfitting
- **Learning Rate Scheduling:** Linear warmup for training stability
- **Early Stopping:** Monitor validation loss to prevent overfitting
- **Larger Models:** Experiment with bert-base-uncased or roberta-base
- **Full Dataset:** Train on complete 25K training samples
- **Gradient Accumulation:** Increase effective batch size
- **Hyperparameter Search:** Systematic tuning of learning rate (1e-5 to 5e-5), batch size, and epochs

## Conclusion

This project demonstrates the importance of proper hyperparameter tuning for transformer models. When appropriately configured, transformers significantly outperform classical NLP methods due to their superior contextual understanding. However, classical methods remain valuable baselines and can be effective with proper feature engineering.
