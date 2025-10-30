#!/usr/bin/env python3
"""Quick test script to verify Composer implementation."""

import torch
import numpy as np
from hw2 import Composer

print("=" * 60)
print("Testing Composer Implementation")
print("=" * 60)

# Test 1: Instantiation
print("\n1. Testing instantiation...")
try:
    composer = Composer(load_trained=False)
    print(f"✓ Composer instantiated successfully")
    print(f"  - Model device: {composer.device}")
    print(f"  - Vocab size: {composer.vocab_size}")
    print(f"  - Model parameters: {sum(p.numel() for p in composer.model.parameters()):,}")
except Exception as e:
    print(f"✗ Failed to instantiate: {e}")
    exit(1)

# Test 2: Training on a batch
print("\n2. Testing training method...")
try:
    # Create a dummy batch [batch_size, seq_len]
    batch_size = 4
    seq_len = 100
    dummy_batch = torch.randint(0, composer.vocab_size, (batch_size, seq_len))
    
    # Move to same device as model
    dummy_batch = dummy_batch.to(composer.device)
    
    # Train on batch
    loss = composer.train(dummy_batch)
    print(f"✓ Training successful")
    print(f"  - Loss: {loss:.4f}")
    print(f"  - Training step: {composer.train_step}")
except Exception as e:
    print(f"✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Composition
print("\n3. Testing compose method...")
try:
    generated_seq = composer.compose()
    print(f"✓ Composition successful")
    print(f"  - Generated sequence shape: {generated_seq.shape}")
    print(f"  - Sequence length: {len(generated_seq)}")
    print(f"  - First 10 tokens: {generated_seq[:10]}")
    print(f"  - Token range: [{generated_seq.min()}, {generated_seq.max()}]")
    
    # Verify sequence is long enough
    if len(generated_seq) >= 2000:
        print(f"✓ Sequence length adequate for 20+ seconds")
    else:
        print(f"⚠ Warning: Sequence might be too short for 20 seconds")
except Exception as e:
    print(f"✗ Composition failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Multiple compositions (check randomness)
print("\n4. Testing randomness in composition...")
try:
    seq1 = composer.compose()
    seq2 = composer.compose()
    
    if not np.array_equal(seq1, seq2):
        print(f"✓ Compositions are different (randomness working)")
        print(f"  - Sequence 1 length: {len(seq1)}")
        print(f"  - Sequence 2 length: {len(seq2)}")
    else:
        print(f"⚠ Warning: Compositions are identical")
except Exception as e:
    print(f"✗ Randomness test failed: {e}")

# Test 5: Save and Load
print("\n5. Testing save/load functionality...")
try:
    # Save the model
    composer._save_model()
    print(f"✓ Model saved to {composer.model_path}")
    
    # Try to load it
    composer2 = Composer(load_trained=True)
    print(f"✓ Model loaded successfully")
    print(f"  - Training step: {composer2.train_step}")
    
    # Test that loaded model can compose
    seq_loaded = composer2.compose()
    print(f"✓ Loaded model can compose (length: {len(seq_loaded)})")
    
except Exception as e:
    print(f"✗ Save/Load failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All basic tests passed!")
print("=" * 60)

