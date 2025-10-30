#!/usr/bin/env python3
"""
Demo script showing how to use the Composer to train on MIDI data
and generate music compositions
"""

import torch
import numpy as np
from hw2 import Composer
from midi2seq import process_midi_seq, seq2piano

def train_composer_demo(num_batches=10, batch_size=8):
    """
    Demonstration of training the Composer on MIDI data
    """
    print("=" * 60)
    print("Composer Training Demo")
    print("=" * 60)
    
    # Initialize composer
    print("\n1. Initializing Composer...")
    composer = Composer(load_trained=False)
    
    # Load training data
    print("\n2. Loading MIDI training data...")
    print("   (This may take a few minutes on first run)")
    try:
        # Process MIDI files from maestro dataset
        train_data = process_midi_seq(
            datadir='.',  # Assumes maestro-v1.0.0 is in current directory
            n=num_batches * batch_size * 51,  # Get enough segments
            maxlen=100
        )
        print(f"   Loaded {len(train_data)} training segments")
    except Exception as e:
        print(f"   Warning: Could not load MIDI data: {e}")
        print("   Using synthetic data for demo...")
        # Create synthetic data if MIDI loading fails
        train_data = np.random.randint(0, composer.vocab_size, (num_batches * batch_size, 101))
    
    # Training loop
    print(f"\n3. Training for {num_batches} batches...")
    losses = []
    
    for batch_idx in range(num_batches):
        # Get batch
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch = train_data[start_idx:end_idx]
        
        # Convert to torch tensor
        batch_tensor = torch.tensor(batch, dtype=torch.long)
        
        # Train on batch
        loss = composer.train(batch_tensor)
        losses.append(loss)
        
        print(f"   Batch {batch_idx + 1}/{num_batches} - Loss: {loss:.4f}")
    
    print(f"\n   Average loss: {np.mean(losses):.4f}")
    print(f"   Final loss: {losses[-1]:.4f}")
    
    return composer


def compose_demo(composer=None):
    """
    Demonstration of music composition
    """
    print("\n" + "=" * 60)
    print("Music Composition Demo")
    print("=" * 60)
    
    # Load or create composer
    if composer is None:
        print("\n1. Loading trained Composer...")
        try:
            composer = Composer(load_trained=True)
        except:
            print("   No trained model found, creating new one...")
            composer = Composer(load_trained=False)
    
    # Generate music
    print("\n2. Generating music composition...")
    print("   Length: ~3000 tokens (20+ seconds)")
    print("   Temperature: 1.0 (balanced creativity)")
    print("   Top-k: 50 (diverse sampling)")
    
    sequence = composer.compose(length=3000, temperature=1.0, top_k=50)
    
    print(f"\n   ✓ Generated {len(sequence)} tokens")
    print(f"   Token range: [{sequence.min()}, {sequence.max()}]")
    print(f"   First 20 tokens: {sequence[:20]}")
    
    # Convert to MIDI
    print("\n3. Converting to MIDI...")
    try:
        midi = seq2piano(sequence)
        output_file = 'generated_composition.mid'
        midi.write(output_file)
        print(f"   ✓ Saved to: {output_file}")
        print(f"   Duration: ~{midi.get_end_time():.2f} seconds")
    except Exception as e:
        print(f"   Warning: Could not save MIDI: {e}")
    
    return sequence


def main():
    """Main demo function"""
    print("\n" + "=" * 60)
    print("Piano Music Composer - Complete Demo")
    print("=" * 60)
    
    # Option 1: Train and compose
    print("\nOption 1: Train a new model and compose music")
    print("Option 2: Load existing model and compose music")
    print("Option 3: Just compose with untrained model (random output)")
    
    choice = input("\nEnter choice (1/2/3) [default=3]: ").strip() or "3"
    
    if choice == "1":
        # Train and compose
        composer = train_composer_demo(num_batches=10, batch_size=8)
        compose_demo(composer)
    elif choice == "2":
        # Load and compose
        compose_demo()
    else:
        # Quick compose demo
        print("\n[Quick demo with untrained model]")
        composer = Composer(load_trained=False)
        compose_demo(composer)
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

