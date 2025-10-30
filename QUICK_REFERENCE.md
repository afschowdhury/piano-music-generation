# Piano Music Composer - Quick Reference

## 🚀 One-Liner Commands

### Setup
```bash
conda activate piano-composer
```

### Test
```bash
python test_composer.py
```

### Train (Quick)
```bash
python train.py --epochs 5 --batch-size 16
```

### Train (Full)
```bash
python train.py --epochs 10 --batch-size 32 --num-segments 20000 --generate-sample
```

### Demo
```bash
python demo.py
```

---

## 📝 Python Code Snippets

### Create New Model
```python
from hw2 import Composer
composer = Composer(load_trained=False)
```

### Load Trained Model
```python
from hw2 import Composer
composer = Composer(load_trained=True)
```

### Train on Batch
```python
import torch
batch = torch.randint(0, 382, (16, 101))  # [batch_size, seq_len]
loss = composer.train(batch)
print(f"Loss: {loss:.4f}")
```

### Generate Music (Default)
```python
sequence = composer.compose()  # 3000 tokens, temp=1.0, top_k=50
```

### Generate Music (Custom)
```python
sequence = composer.compose(
    length=5000,      # Longer composition
    temperature=0.8,  # More conservative
    top_k=30         # More focused
)
```

### Save as MIDI
```python
from midi2seq import seq2piano

midi = seq2piano(sequence)
midi.write('output.mid')
print(f"Duration: {midi.get_end_time():.2f}s")
```

### Complete Pipeline
```python
from hw2 import Composer
from midi2seq import seq2piano

# Load model
composer = Composer(load_trained=True)

# Generate
seq = composer.compose(length=3000, temperature=1.0)

# Save
midi = seq2piano(seq)
midi.write('composition.mid')
```

---

## 🎛️ Parameter Reference

### Training Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `epochs` | 5 | 1-100 | Number of training epochs |
| `batch_size` | 16 | 4-64 | Training batch size |
| `learning_rate` | 3e-4 | 1e-5 to 1e-3 | Adam learning rate |
| `num_segments` | 10000 | 1000-50000 | Training data segments |

### Generation Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `length` | 3000 | 500-10000 | Number of tokens to generate |
| `temperature` | 1.0 | 0.5-2.0 | Sampling randomness |
| `top_k` | 50 | 0-200 | Top-k filtering (0=off) |

### Model Architecture

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | 382 | MIDI event vocabulary |
| `d_model` | 256 | Embedding dimension |
| `num_layers` | 6 | Transformer layers |
| `num_heads` | 8 | Attention heads |
| `d_ff` | 1024 | Feed-forward dimension |
| `max_seq_len` | 1024 | Maximum sequence length |
| `dropout` | 0.1 | Dropout rate |

---

## 🎨 Generation Temperature Guide

### Conservative (0.5 - 0.7)
- More predictable
- Fewer mistakes
- Can be repetitive
- Good for: Background music

### Balanced (0.8 - 1.2)
- Natural sounding
- Good variety
- Occasional surprises
- Good for: General compositions

### Creative (1.3 - 2.0)
- Very diverse
- Experimental
- May have errors
- Good for: Unique pieces

---

## 🎯 Top-K Sampling Guide

### Focused (5 - 20)
- Very consistent
- Limited variety
- Coherent structure

### Balanced (30 - 60)
- Good diversity
- Still coherent
- Recommended default

### Diverse (70 - 150)
- Maximum variety
- May lose coherence
- Experimental

### No Filtering (0)
- Pure temperature sampling
- Maximum randomness

---

## 📊 File Locations

| File | Purpose | Auto-Generated |
|------|---------|----------------|
| `hw2.py` | Main implementation | No |
| `composer_model.pt` | Model checkpoint | Yes |
| `*.mid` | Generated MIDI files | Yes |
| `maestro-v1.0.0/` | Training data | No |

---

## ⚡ Performance Tips

### Speed Up Training
- Increase batch size (if GPU allows)
- Reduce number of segments
- Use GPU (30-50x faster than CPU)

### Improve Quality
- Train for more epochs
- Use more training data
- Increase model size (edit hw2.py)

### Better Generations
- Train model longer first
- Try temperature 0.8-1.2
- Use top_k between 30-60

---

## 🔍 Common Issues

### Out of Memory
```python
# Reduce batch size
python train.py --batch-size 8
```

### Model Not Found
```python
# Create new model first
composer = Composer(load_trained=False)
composer._save_model()
```

### Poor Quality Output
```python
# Train longer or adjust temperature
composer.compose(temperature=0.8)  # More conservative
```

---

## 📞 Quick Checks

### Check CUDA
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

### Check Model
```python
from hw2 import Composer
composer = Composer(load_trained=True)
print(f"Training step: {composer.train_step}")
print(f"Parameters: {sum(p.numel() for p in composer.model.parameters()):,}")
```

### Check Generated Music
```python
from midi2seq import seq2piano
sequence = composer.compose()
midi = seq2piano(sequence)
print(f"Tokens: {len(sequence)}")
print(f"Duration: {midi.get_end_time():.2f}s")
print(f"Notes: {len(midi.instruments[0].notes)}")
```

---

## 🎵 Music Quality Metrics

After generating, check:
1. **Length**: Should be 2500-3500 tokens for 20-30s
2. **Token Distribution**: Mix of all 4 event types
3. **MIDI Duration**: Should match expected time
4. **Note Count**: Typically 200-500 notes per composition
5. **Playability**: Listen to verify it sounds musical

---

## 💡 Pro Tips

1. **Start Small**: Test with 1-2 epochs first
2. **Save Often**: Model auto-saves every 1000 steps
3. **Monitor Loss**: Should decrease to ~2-3 when trained
4. **Experiment**: Try different temperature/top_k combos
5. **Batch Generate**: Create multiple versions, pick the best

---

## 🎓 Learning Resources

### Understanding Transformers
- Multi-head attention allows parallel processing
- Causal masking ensures autoregressive property
- Positional encoding captures sequence order

### Understanding Music Generation
- Lower temperature = safer choices
- Top-k limits options to most likely
- Longer sequences need sliding window

### Debugging
- Check train_step to verify training progress
- Monitor loss to ensure learning
- Generate samples periodically to evaluate

---

## ✅ Verification Checklist

Before training:
- [ ] Conda environment activated
- [ ] maestro data available
- [ ] GPU accessible (optional but recommended)

After training:
- [ ] Loss decreased from initial value
- [ ] Model checkpoint saved
- [ ] Sample generation works

For production:
- [ ] Model trained for 5+ epochs
- [ ] Generated samples sound musical
- [ ] Multiple temperature values tested

---

**Quick Start**: `conda activate piano-composer && python test_composer.py`

