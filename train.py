#!/usr/bin/env python3
"""
Training script for the Piano Music Composer
Trains on MAESTRO dataset with proper batching and evaluation
"""

import argparse
import time
import torch
import numpy as np
from hw2 import Composer
from midi2seq import process_midi_seq, seq2piano


def train(args):
    """Train the composer model"""
    
    print("=" * 70)
    print("Piano Music Composer - Training")
    print("=" * 70)
    
    # Initialize composer
    print(f"\n1. Initializing Composer...")
    print(f"   Loading from checkpoint: {args.load_trained}")
    composer = Composer(load_trained=args.load_trained)
    print(f"   Device: {composer.device}")
    print(f"   Model parameters: {sum(p.numel() for p in composer.model.parameters()):,}")
    print(f"   Starting from step: {composer.train_step}")
    
    # Load training data
    print(f"\n2. Loading training data...")
    print(f"   Data directory: {args.datadir}")
    print(f"   Number of segments: {args.num_segments}")
    print(f"   Sequence length: {args.seq_len}")
    print("   (This may take several minutes...)")
    
    start_time = time.time()
    try:
        train_data = process_midi_seq(
            datadir=args.datadir,
            n=args.num_segments,
            maxlen=args.seq_len
        )
        load_time = time.time() - start_time
        print(f"   ✓ Loaded {len(train_data)} segments in {load_time:.1f}s")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        print("\n   Make sure maestro-v1.0.0 directory exists in datadir!")
        return
    
    # Training loop
    print(f"\n3. Training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Total batches per epoch: {len(train_data) // args.batch_size}")
    print(f"   Evaluation interval: every {args.eval_interval} steps")
    print()
    
    best_loss = float('inf')
    step = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)
        
        # Shuffle data
        indices = np.random.permutation(len(train_data))
        train_data_shuffled = train_data[indices]
        
        epoch_losses = []
        epoch_start = time.time()
        
        # Batch training
        num_batches = len(train_data) // args.batch_size
        
        for batch_idx in range(num_batches):
            batch_start = time.time()
            
            # Get batch
            start_idx = batch_idx * args.batch_size
            end_idx = start_idx + args.batch_size
            batch = train_data_shuffled[start_idx:end_idx]
            
            # Convert to tensor
            batch_tensor = torch.tensor(batch, dtype=torch.long)
            
            # Train
            loss = composer.train(batch_tensor)
            epoch_losses.append(loss)
            step += 1
            
            batch_time = time.time() - batch_start
            
            # Log progress
            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = np.mean(epoch_losses[-args.log_interval:])
                print(f"  Step {step:6d} | Batch {batch_idx+1:4d}/{num_batches:4d} | "
                      f"Loss: {loss:.4f} | Avg: {avg_loss:.4f} | "
                      f"Time: {batch_time:.3f}s")
            
            # Evaluate
            if step % args.eval_interval == 0:
                eval_loss = evaluate(composer, train_data_shuffled, args.batch_size)
                print(f"\n  >>> Evaluation at step {step}: Loss = {eval_loss:.4f}")
                
                # Save best model
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    print(f"  >>> New best model! Saving...")
                    composer._save_model()
                print()
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = np.mean(epoch_losses)
        print(f"\n  Epoch {epoch + 1} Summary:")
        print(f"    Average Loss: {avg_epoch_loss:.4f}")
        print(f"    Time: {epoch_time:.1f}s")
        print(f"    Batches/sec: {num_batches / epoch_time:.2f}")
    
    # Final save
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"  Total steps: {step}")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Final loss: {epoch_losses[-1]:.4f}")
    composer._save_model()
    print(f"  Model saved to: {composer.model_path}")
    print("=" * 70)
    
    # Generate sample
    if args.generate_sample:
        print("\n4. Generating sample composition...")
        generate_sample(composer)


def evaluate(composer, data, batch_size, num_batches=10):
    """Evaluate the model on a subset of data"""
    composer.model.eval()
    
    losses = []
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            if start_idx + batch_size > len(data):
                break
            
            batch = data[start_idx:start_idx + batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.long, device=composer.device)
            
            # Forward pass
            input_seq = batch_tensor[:, :-1]
            target_seq = batch_tensor[:, 1:]
            mask = composer.model.generate_causal_mask(input_seq.size(1), composer.device)
            
            logits = composer.model(input_seq, mask)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, composer.vocab_size),
                target_seq.reshape(-1)
            )
            losses.append(loss.item())
    
    composer.model.train()
    return np.mean(losses)


def generate_sample(composer, output_file='sample_composition.mid'):
    """Generate a sample composition"""
    print(f"  Generating music...")
    sequence = composer.compose(length=3000, temperature=1.0, top_k=50)
    
    print(f"  ✓ Generated {len(sequence)} tokens")
    
    try:
        midi = seq2piano(sequence)
        midi.write(output_file)
        print(f"  ✓ Saved to: {output_file}")
        print(f"  Duration: {midi.get_end_time():.2f} seconds")
    except Exception as e:
        print(f"  Warning: Could not save MIDI: {e}")


def main():
    parser = argparse.ArgumentParser(description='Train Piano Music Composer')
    
    # Model options
    parser.add_argument('--load-trained', action='store_true',
                        help='Load existing trained model')
    
    # Data options
    parser.add_argument('--datadir', type=str, default='.',
                        help='Directory containing maestro-v1.0.0 folder (default: .)')
    parser.add_argument('--num-segments', type=int, default=10000,
                        help='Number of training segments to load (default: 10000)')
    parser.add_argument('--seq-len', type=int, default=100,
                        help='Length of training sequences (default: 100)')
    
    # Training options
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log every N batches (default: 10)')
    parser.add_argument('--eval-interval', type=int, default=500,
                        help='Evaluate every N steps (default: 500)')
    
    # Output options
    parser.add_argument('--generate-sample', action='store_true',
                        help='Generate a sample composition after training')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()

