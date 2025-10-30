# Piano Music Composer - Transformer-Based Music Generation

A deep learning system that uses a transformer-based language model to generate piano music compositions.

## Overview

This implementation uses a decoder-only transformer architecture (similar to GPT) to learn patterns from MIDI piano music and generate new compositions. The model is trained on sequences of musical events and can produce coherent, creative piano pieces of 20+ seconds.

## Architecture

### Model Components

1. **Token Embedding Layer**: Converts music event tokens (vocab size: 382) into dense vectors
2. **Positional Encoding**: Sinusoidal encoding to capture sequential information
3. **Transformer Decoder Blocks** (6 layers):
   - Multi-head self-attention (8 heads)
   - Feed-forward networks (4-layer MLP)
   - Layer normalization
   - Residual connections
4. **Output Projection**: Maps to vocabulary for next-token prediction

### Hyperparameters

- **Embedding Dimension**: 256
- **Number of Layers**: 6
- **Attention Heads**: 8
- **Feed-forward Dimension**: 1024
- **Max Sequence Length**: 1024 tokens
- **Dropout**: 0.1
- **Learning Rate**: 3e-4
- **Total Parameters**: ~4.9M

## Installation

### 1. Create Conda Environment

```bash
conda create -n piano-composer python=3.10 -y
conda activate piano-composer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (with CUDA support if available)
- NumPy
- pretty_midi (for MIDI processing)

## Usage

### Quick Test

Run the test script to verify the implementation:

```bash
python test_composer.py
```

This will test:
- Model instantiation
- Training on a batch
- Music composition
- Randomness in generation
- Save/load functionality

### Training the Model

```python
from hw2 import Composer
from midi2seq import process_midi_seq
import torch

# Initialize composer
composer = Composer(load_trained=False)

# Load training data from MIDI files
train_data = process_midi_seq(
    datadir='.',  # Directory containing maestro-v1.0.0 folder
    n=10000,      # Number of training segments
    maxlen=100    # Segment length
)

# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        batch_tensor = torch.tensor(batch, dtype=torch.long)
        
        loss = composer.train(batch_tensor)
        
        if i % 100 == 0:
            print(f"Epoch {epoch}, Step {i}, Loss: {loss:.4f}")
```

### Generating Music

```python
from hw2 import Composer
from midi2seq import seq2piano

# Load trained model
composer = Composer(load_trained=True)

# Generate a composition
sequence = composer.compose(
    length=3000,      # Number of tokens (~20-30 seconds)
    temperature=1.0,  # Sampling temperature (0.5-1.5)
    top_k=50         # Top-k sampling (0 = disabled)
)

# Convert to MIDI and save
midi = seq2piano(sequence)
midi.write('my_composition.mid')

print(f"Generated {len(sequence)} tokens")
print(f"Duration: {midi.get_end_time():.2f} seconds")
```

### Interactive Demo

```bash
python demo.py
```

The demo provides three options:
1. Train a new model and compose music
2. Load an existing model and compose music
3. Quick composition with an untrained model

### Composition Parameters

- **length**: Number of tokens to generate (3000 ≈ 20-30 seconds)
- **temperature**: Controls randomness
  - Lower (0.5-0.8): More conservative, repetitive
  - Medium (0.9-1.1): Balanced creativity
  - Higher (1.2-2.0): More random, experimental
- **top_k**: Limits sampling to top-k most likely tokens
  - 0: No filtering (pure temperature sampling)
  - 20-50: Balanced diversity
  - 5-10: More focused, less diverse

## File Structure

```
hw2/
├── hw2.py                 # Main Composer implementation
├── model_base.py          # Base class definitions
├── midi2seq.py            # MIDI ↔ sequence conversion
├── test_composer.py       # Test suite
├── demo.py                # Interactive demo
├── requirements.txt       # Python dependencies
├── README_COMPOSER.md     # This file
├── maestro-v1.0.0/        # MIDI training data
└── composer_model.pt      # Saved model checkpoint (generated)
```

## Training Data

The model is designed to work with the MAESTRO dataset (MIDI and Audio Edited for Synchronous TRacks and Organization), which contains classical piano performances.

The dataset should be in the `maestro-v1.0.0/` directory. The `midi2seq.py` module handles:
1. Loading MIDI files
2. Converting to event sequences
3. Segmenting into fixed-length training samples

## Model Checkpointing

The model automatically saves checkpoints:
- **File**: `composer_model.pt`
- **Frequency**: Every 1000 training steps
- **Contents**: 
  - Model weights
  - Optimizer state
  - Training step counter
  - Hyperparameters

To load a saved model:

```python
composer = Composer(load_trained=True)
```

## Performance

### Training

- **Device**: CUDA (GPU) if available, otherwise CPU
- **Batch Size**: Flexible (8-32 recommended)
- **Memory**: ~2-4 GB GPU memory for batch size 16
- **Speed**: ~0.1-0.3 seconds per batch on GPU

### Generation

- **Speed**: ~1-2 seconds for 3000 tokens on GPU
- **Quality**: Improves significantly with training
- **Diversity**: Controlled by temperature and top-k parameters

## Implementation Details

### Next-Token Prediction

The model is trained using causal language modeling:
- **Input**: Token sequence [t₀, t₁, ..., t_{n-1}]
- **Target**: Shifted sequence [t₁, t₂, ..., t_n]
- **Loss**: Cross-entropy between predictions and targets

### Autoregressive Generation

Music is generated token-by-token:
1. Start with a random seed token
2. Feed through model to get next token probabilities
3. Sample next token using temperature + top-k
4. Append to sequence and repeat
5. Use sliding window for long sequences (>1024 tokens)

### Causal Masking

Self-attention uses causal masking to ensure each position can only attend to previous positions, maintaining the autoregressive property during both training and generation.

## Extending the Model

### Increasing Capacity

To improve quality, increase model size:

```python
# In hw2.py Composer.__init__()
self.d_model = 512        # Default: 256
self.num_layers = 8       # Default: 6
self.num_heads = 16       # Default: 8
self.d_ff = 2048         # Default: 1024
```

### Adding Conditioning

To condition on specific styles or composers, add conditional tokens or embeddings.

### Fine-tuning on Specific Styles

Train on a subset of MIDI files filtered by composer or style.

## Troubleshooting

### Out of Memory

Reduce batch size or model size:
```python
batch_size = 4  # Instead of 16
self.d_model = 128  # Instead of 256
```

### Poor Quality Generations

- Train for more epochs
- Adjust temperature (try 0.8-1.2)
- Increase model capacity
- Ensure sufficient training data

### CUDA Errors

Ensure PyTorch is installed with CUDA support:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## References

1. **Attention Is All You Need** - Vaswani et al. (2017)
2. **GPT: Improving Language Understanding** - Radford et al. (2018)
3. **Music Transformer** - Huang et al. (2018)
4. **MAESTRO Dataset** - Hawthorne et al. (2019)

## License

Copyright 2020 Jian Zhang. See individual file headers for details.

