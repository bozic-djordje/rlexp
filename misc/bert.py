import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT-base model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
model.eval()

# Example sentence
sentence = "The quick brown fox jumps over the lazy dog."

# Tokenize and encode the sentence
inputs = tokenizer(sentence, return_tensors='pt')

# Perform inference (no gradient required)
with torch.no_grad():
    outputs = model(**inputs)

# `last_hidden_state` is the embeddings for each token (autoregressive state)
# Shape: [batch_size, sequence_length, hidden_size]
embeddings = outputs.last_hidden_state

# Optionally, hidden states from all layers
# hidden_states = outputs.hidden_states

print("Embeddings shape:", embeddings.shape)
# Example: embedding vector for the first token '[CLS]'
print("Embedding for [CLS] token:", embeddings[0, 0, :])