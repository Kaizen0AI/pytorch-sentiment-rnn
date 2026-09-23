# Experiment 1 — Baseline Vanilla RNN

Task: IMDb binary sentiment classification
Dataset: IMDb Dataset — 50,000 reviews
Split: 80% train / 10% validation / 10% test
Random state: 42
Stratification: Yes

### Preprocessing

HTML tags removed with regex
Text converted to lowercase
Whitespace-based tokenization
Vocabulary built from training set only
Vocabulary size: 30,000
<PAD> token: ID 0
<UNK> token: ID 1
Maximum sequence length: 200
Shorter sequences padded
Longer sequences truncated

### Model

Embedding dimension: 128
Vanilla nn.RNN
Hidden dimension: 128
Output dimension: 1
Single-layer, unidirectional
BCEWithLogitsLoss

### Training

Optimizer: Adam
Initial learning rate: 0.001
Batch size: 64
Maximum epochs: 10
Validation loss used for monitoring
ReduceLROnPlateau: factor 0.5, patience 3
Early stopping: patience 5
Best validation-loss checkpoint saved

### Initial sanity check

3 epochs completed successfully
Training loss decreased
Validation loss/accuracy showed overfitting behavior

### Final test-set results

Metric	Result
Test Loss	0.6874
Accuracy	54.14%
Precision	59.66%
Recall	25.56%
F1	35.79%
TP	639
TN	2,068
FP	432
FN	1,861

### Observation

The vanilla RNN learned only weakly. It exhibited a strong negative-prediction bias, resulting in particularly poor positive recall (25.56%) and F1 score (35.79%). Validation performance remained close to chance while training loss decreased, indicating poor generalization. The model has substantial parameter capacity, so model size alone was not established as the cause.

Decision: Replace the vanilla RNN with a GRU while keeping the rest of the pipeline unchanged initially.

# Experiment 2 — GRU

### Architecture

Embedding: 128
GRU hidden dimension: 128
Single-layer, unidirectional
Output: 1 logit
BCEWithLogitsLoss

### Training

Adam, LR = 0.001
Batch size = 64
Maximum epochs = 50
ReduceLROnPlateau: factor = 0.5, patience = 3
Early stopping: patience = 5
Best checkpoint selected by validation loss

### Best checkpoint

Epoch: 3
Validation loss: 0.3087
Validation accuracy: 86.68%

### Test results

Test loss: 0.3112
Accuracy: 86.58%
Precision: 89.37%
Recall: 83.04%
F1: 86.09%
TP: 2,076
TN: 2,253
FP: 247
FN: 424

### Observation

Replacing the vanilla RNN with a GRU produced a substantial improvement across all evaluation metrics. Test accuracy increased from 54.14% to 86.58%, while F1 increased from 35.79% to 86.09%. The most significant improvement was recall, which increased from 25.56% to 83.04%, indicating that the GRU was substantially better at identifying positive reviews. The model still overfit rapidly, with validation loss reaching its minimum at epoch 3 while training loss continued to decrease.