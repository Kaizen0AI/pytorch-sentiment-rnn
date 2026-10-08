# PyTorch Sentiment RNN

A sentiment classification project built with **PyTorch** using the **IMDb movie review dataset**.

The goal of this project was to learn the fundamentals of NLP sequence classification in PyTorch, starting with a vanilla RNN and then improving the model by replacing it with a GRU.

The project also includes a simple **Streamlit web app** for running predictions on custom text.

---

## Project Overview

The model takes a movie review as input and predicts whether the sentiment is:

- **Positive**
- **Negative**

The pipeline is:

```text
IMDb Reviews
    ↓
Text preprocessing
    ↓
Tokenization
    ↓
Vocabulary
    ↓
Token → Integer IDs
    ↓
Padding / Truncation
    ↓
Embedding
    ↓
RNN / GRU
    ↓
Linear layer
    ↓
Sentiment prediction
```

---

## Dataset

The project uses the IMDb dataset containing **50,000 labeled movie reviews**.

The data was split into:

- **80% training**
- **10% validation**
- **10% test**

The splits were stratified and generated with `random_state=42`.

---

## Preprocessing

The preprocessing pipeline was intentionally kept simple:

1. Remove HTML tags from reviews.
2. Convert text to lowercase.
3. Split text using whitespace tokenization.
4. Build the vocabulary using **training data only**.
5. Limit the vocabulary to **30,000 tokens**.
6. Reserve:
   - `<PAD>` → `0`
   - `<UNK>` → `1`
7. Convert tokens to integer IDs.
8. Truncate or pad reviews to a maximum sequence length of **200** tokens.

No pretrained embeddings or advanced tokenizers were used.

---

## Models

### Experiment 1 — Vanilla RNN

The first model was a simple PyTorch RNN:

```text
Embedding
    ↓
RNN
    ↓
Linear
    ↓
Output logit
```

Configuration:

| Parameter | Value |
|---|---:|
| Vocabulary size | 30,000 |
| Embedding dimension | 128 |
| Hidden dimension | 128 |
| Output dimension | 1 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | BCEWithLogitsLoss |
| Maximum sequence length | 200 |

Test results:

| Metric | Result |
|---|---:|
| Loss | 0.6874 |
| Accuracy | 54.14% |
| Precision | 59.66% |
| Recall | 25.56% |
| F1 | 35.79% |

The vanilla RNN performed poorly, especially on positive-class recall.

---

### Experiment 2 — GRU

The second experiment kept the main pipeline and hyperparameters the same and replaced the vanilla RNN with a **GRU**.

```text
Embedding
    ↓
GRU
    ↓
Linear
    ↓
Output logit
```

Training used:

- ReduceLROnPlateau
- Early stopping
- Best-model checkpointing based on validation loss

The best checkpoint was from **epoch 3**.

Test results:

| Metric | Result |
|---|---:|
| Loss | 0.3112 |
| Accuracy | **86.58%** |
| Precision | **89.37%** |
| Recall | **83.04%** |
| F1 | **86.09%** |

Confusion matrix counts:

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 2253 | 247 |
| Actual Positive | 424 | 2076 |

### RNN vs GRU

| Metric | Vanilla RNN | GRU |
|---|---:|---:|
| Test Loss | 0.6874 | **0.3112** |
| Accuracy | 54.14% | **86.58%** |
| Precision | 59.66% | **89.37%** |
| Recall | 25.56% | **83.04%** |
| F1 | 35.79% | **86.09%** |

The GRU produced a large improvement, particularly in recall and F1 score.

---

## Training

Training includes:

- Validation after every epoch
- Learning-rate reduction when validation loss plateaus
- Early stopping
- Saving the best model checkpoint

The best checkpoint is saved as:

```text
checkpoints/best_model.pt
```

---

## Project Structure

```text
pytorch-sentiment-rnn/
│
├── src/
│   ├── constants.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluation.py
│   └── helper_funcs.py
│
├── checkpoints/
│   └── best_model.pt
│
├── artifacts/
│   └── vocab.json
│
├── train.py
├── evaluate.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/pytorch-sentiment-rnn.git
cd pytorch-sentiment-rnn
```

### 2. Create the environment

For example, using conda:

```bash
conda create -n rnn_env python=3.11
conda activate rnn_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

Make sure the IMDb CSV is available and the dataset path is configured.

Then run:

```bash
python train.py
```

### 5. Evaluate the model

```bash
python evaluate.py
```

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

The web app allows you to enter a custom movie review and get the model's predicted sentiment and probability.

---

## Streamlit App

The Streamlit application uses the saved:

```text
best_model.pt
vocab.json
```

The vocabulary is saved separately because the model depends on the exact token-to-index mapping used during training.

The app performs the same preprocessing and inference steps used during evaluation.

---

## What I Learned

This project was mainly a learning exercise rather than an attempt to build a state-of-the-art sentiment classifier.

Some of the main things I learned:

- Building a text preprocessing pipeline for NLP.
- Creating a vocabulary from training data.
- Handling `<PAD>` and `<UNK>` tokens.
- Converting text into sequences of integer IDs.
- Using `nn.Embedding`.
- Building sequence classifiers with PyTorch.
- Understanding the difference between a vanilla RNN and a GRU.
- Using `BCEWithLogitsLoss` for binary classification.
- Implementing validation and test evaluation.
- Tracking precision, recall, F1, TP, TN, FP, and FN.
- Using learning-rate scheduling.
- Implementing early stopping and model checkpointing.
- Loading a trained model for inference.
- Deploying a trained model with Streamlit.

---

## Notes

This project intentionally uses a relatively simple NLP pipeline.

It does not include:

- Pretrained language models
- Transformer architectures
- Pretrained word embeddings
- Advanced tokenization
- Attention mechanisms

The goal was to understand the fundamentals of recurrent neural networks and sequence classification before moving on to more advanced concepts.

---

## Project Status

**Complete.**

The purpose of this project was to learn the fundamentals of RNN-based sentiment classification in PyTorch and to take the model all the way from data preprocessing to a working web demo.

The next step is to move on to other recurrent-network concepts and projects rather than continuing to optimize this model.
