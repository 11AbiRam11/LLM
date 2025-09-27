# Large Language Model

## File Structure
- `base`
- `gpt-download3`
- `gpt2`
- `Preprocessing dolly_data`
- `TrainingData` (folder)

---

### `base.ipynb`
This notebook contains the initial implementation of the GPT model, built incrementally block by block. Below are the foundational components of the LLM.

---

## Byte Pair Encoding (BPE) — Tokenization Algorithm

### Why BPE?
Before an LLM can learn from text, raw text must be converted into tokens. **Byte Pair Encoding (BPE)** is a subword tokenization algorithm that balances:
- **Too small vocabularies**: Only characters, inefficient for long sequences.  
- **Too large vocabularies**: Full words, unable to handle unseen tokens.  

Modern LLMs such as **GPT** and **LLaMA** rely on BPE (or byte-level BPE).

### Training Algorithm
BPE is **iterative and greedy**, merging the most frequent adjacent token pairs until the vocabulary reaches a target size.

1. **Initial State**  
   - Preprocess corpus into UTF-8 bytes (for byte-level BPE).  
   - Initialize vocabulary with 256 unique byte values.  
   - Represent the corpus as sequences of these base units.  

2. **Iterative Merging**  
   - Count frequencies of adjacent token pairs.  
   - Merge the most frequent pair.  
   - Update the corpus with the merged token.  
   - Add the new token to the vocabulary.  

This process naturally identifies frequent subword units such as `"ing"`, `"un"`, or entire words like `"the"`.

### Encoding (Tokenization)
- Convert raw text into base units (bytes).  
- Apply learned merge rules sequentially, replacing frequent pairs with merged tokens until no further merges apply.  

### Key Benefits
- Produces compact, frequent tokens.  
- Handles unseen words by breaking them into subword units.  
- Enables efficient training and inference.  

### Implementation Challenges
- Efficient management of a large corpus.  
- Tracking pair frequencies at scale.  
- Applying merges quickly during both training and encoding.  

---

## Word Embedding: Converting Tokens to Vectors

Word embeddings map tokens into dense, real-valued vectors that encode semantic meaning and context.

### Key Concepts
- **Dense Representation**: Compact real-valued vectors instead of sparse one-hot encodings.  
- **Semantic Meaning**: Similar words appear close together in vector space.  
- **Trainable Parameters**: Learned during training as part of the network.  

### Integration in Transformers
Embeddings are the **first step** in the forward pass, converting input token IDs into a matrix of vectors.  

---

## Positional Encoding (PE)

Transformers lack recurrence, so positional information must be injected.

### Sinusoidal Encoding
The original Transformer used fixed sinusoidal vectors. This allows the model to generalize to relative positions.

### Combining with Embeddings
Final input representation:  
**Word Embedding + Positional Encoding**  

This injects sequential information into the semantic vector space.

### Modern Approach
Many LLMs use **learned positional embeddings**, but GPT-2 uses learnable embeddings.

---

## Multi-Head Attention (MHA)

MHA allows the model to attend to information from multiple subspaces simultaneously.

1. **Linear Projections**: Input queries, keys, and values are projected into multiple heads.  
2. **Parallel Attention**: Each head performs scaled dot-product attention independently.  
3. **Concatenation**: Outputs from all heads are concatenated.  
4. **Projection**: A final linear layer projects back to the model’s hidden dimension.  

---

## Layer Normalization

Layer Normalization (LayerNorm) stabilizes training by normalizing across features for each sample.

- Normalizes token representations across the embedding dimension.  
- Unlike BatchNorm, it is independent of batch size.  
- Ensures consistent behavior during training and inference.  

---

## GELU Activation

The Gaussian Error Linear Unit (GELU) is used in Transformers as the non-linear activation function.

- Smooth and non-monotonic, unlike ReLU.  
- Improves gradient flow and model performance.  
- Standard choice for feed-forward layers in modern LLMs.  

---

## Feed-Forward Network (FFN)

Each Transformer block contains a position-wise feed-forward network.

- Two linear layers with a non-linearity (typically GELU).  
- Expands the embedding dimension (e.g., D → 4D) and projects back to D.  
- Applied independently to each token.  

---

## Transformer Block: Residual + Add & Norm

Each block is structured as:

1. **LayerNorm → Multi-Head Attention → Residual Connection**  
   - Normalize input, compute self-attention, add back to input.  

2. **LayerNorm → Feed-Forward Network → Residual Connection**  
   - Normalize, apply FFN, add back to input.  

This **residual + normalization flow** stabilizes gradients and enables deep stacking of layers.

---

## Transformer Architecture

GPT models use only the decoder stack of the original Transformer.

### Decoder
- Generates output sequences sequentially.  
- Components per block:  
  - Masked Multi-Head Attention (prevents future token access)  
  - Feed-Forward Network  
  - Residual connections + LayerNorm  

### Output
- Final linear layer projects hidden states to vocabulary logits.  
- Softmax converts logits into token probabilities.  

---

## GPTModel Class Overview

The `GPTModel` class implements a GPT-style language model:

1. **Embeddings**: Token indices → dense vectors + positional embeddings + dropout.  
2. **Transformer Layers**: Contextualize embeddings via stacked Transformer blocks.  
3. **Normalization**: Apply final LayerNorm.  
4. **Output Projection**: Hidden states → vocabulary logits (weight tying applied).  
5. **Forward Pass**: Produces logits for each token position in the input.  

**Summary:** Maps input token sequences through embeddings and Transformer layers to predict the next token distribution.

---

## Pre-Training Phase

Pre-training a GPT model from scratch requires vast data (e.g., web pages, books) and significant compute. For example, GPT-2 was trained on 40GB of data, taking several days on large hardware.  

Given limited resources (Google Colab T4 GPU, 16GB VRAM), full pre-training would take weeks.  

Instead, pre-trained GPT-2 weights can be loaded into your model.

### Downloading Pretrained Weights
```python
from gpt_download3 import download_and_load_gpt2
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```

Now run the ``` load_weights_into_gpt()``` To load the weight into our model.
Congrats now we've can test this out using ```generate()``` to get output from the model.

Note: Most of the time the response may be non-sense because its just pre-trianed model, to get perfect response we need to fine-tune it,
and this will be done soon.

## Training the model on some data to get basic knowledge
I trained the model on dataset called dollyData which is located in TrainingDat folder. 

I've laptop with gtx 1650 4GB of VRAM, and I cleaned the dollydata txt file in such a way that, after each instructions, context and response block i added a special token call <|emdoftext|> which treats this token specially  by model, this token tells when to stop the response, if you want to get the idea in generate function if the new toke is this special token it will stop to generate new token.
*** It took me about 32.5 hours to train this model on that dollyData txt file which is about 11.5MB in size, now the model has some general knowledge about the world, and I saves this model, but unfortunately I can't publish this model in GitHub due to large size, i will upload this llm im hugging face after experimenting on it. 