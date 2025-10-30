#!/usr/bin/env python3
"""
Piano Music Composer using Transformer-based Language Model
Implements a decoder-only transformer (GPT-style) for music generation
"""

import os
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_base import ComposerBase

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


# ============================================================================
# Transformer Architecture Components
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(batch_size, seq_len, d_model)
        
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer decoder block with self-attention and feed-forward"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_out)
        
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerMusicModel(nn.Module):
    """Full transformer model for music generation"""
    
    def __init__(self, vocab_size, d_model=256, num_layers=4, num_heads=4, 
                 d_ff=1024, max_seq_len=1024, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, mask=None):
        """
        Forward pass
        Args:
            x: [batch_size, seq_len] - token indices
            mask: [1, 1, seq_len, seq_len] - causal mask
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Embed tokens
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output projection
        x = self.norm(x)
        logits = self.output(x)
        
        return logits
    
    def generate_causal_mask(self, seq_len, device):
        """Generate causal mask for autoregressive generation"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]


# ============================================================================
# Composer Class
# ============================================================================

class Composer(ComposerBase):
    """
    Transformer-based music composer that can be trained on MIDI sequences
    and generate new piano compositions
    """
    
    def __init__(self, load_trained=False):
        """
        Initialize the Composer
        
        Args:
            load_trained: If True, load a trained model from checkpoint
        """
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {self.device}')
        
        # Define vocabulary size (from midi2seq.py)
        from math import ceil
        velo_inc = 5
        self.vocab_size = 128*2 + 100 + int(ceil(126/velo_inc))  # 382
        
        # Model hyperparameters
        self.d_model = 256
        self.num_layers = 6
        self.num_heads = 8
        self.d_ff = 1024
        self.max_seq_len = 1024
        self.dropout = 0.1
        
        # Training parameters
        self.learning_rate = 3e-4
        self.train_step = 0
        self.save_interval = 1000  # Save every 1000 steps
        
        # Model path
        self.model_path = 'composer_model.pt'
        
        # Initialize model
        self.model = TransformerMusicModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Load trained model if requested
        if load_trained:
            self._load_model()
        else:
            logging.info('Initialized new model')
            logging.info(f'Model parameters: {sum(p.numel() for p in self.model.parameters()):,}')
    
    def train(self, x):
        """
        Train the model on one batch of data
        
        Args:
            x: [batch_size, seq_len] - batch of token sequences
        
        Returns:
            loss: float - mean loss on the batch
        """
        self.model.train()
        
        # Move to device if not already
        if x.device != self.device:
            x = x.to(self.device)
        
        batch_size, seq_len = x.shape
        
        # Prepare input and target
        # Input: all tokens except the last
        # Target: all tokens except the first (next token prediction)
        input_seq = x[:, :-1]
        target_seq = x[:, 1:]
        
        # Generate causal mask
        mask = self.model.generate_causal_mask(seq_len - 1, self.device)
        
        # Forward pass
        logits = self.model(input_seq, mask)  # [batch_size, seq_len-1, vocab_size]
        
        # Calculate loss
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_seq.reshape(-1)
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Increment training step
        self.train_step += 1
        
        # Periodically save model
        if self.train_step % self.save_interval == 0:
            self._save_model()
            logging.info(f'Model saved at step {self.train_step}')
        
        return loss.item()
    
    def compose(self, length=3000, temperature=1.0, top_k=50):
        """
        Generate a music sequence
        
        Args:
            length: Target length of generated sequence (for 20+ seconds, need ~2500-3000 tokens)
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter (0 = no top-k filtering)
        
        Returns:
            seq: numpy array of generated token sequence
        """
        self.model.eval()
        
        with torch.no_grad():
            # Start with a random seed token from the valid range
            # Use a reasonable starting token (e.g., a time shift token)
            start_token = 128*2 + np.random.randint(0, 50)  # Random time shift
            generated = torch.tensor([[start_token]], dtype=torch.long, device=self.device)
            
            # Generate tokens autoregressively
            for _ in range(length - 1):
                # Get the last max_seq_len tokens (sliding window)
                input_seq = generated[:, -self.max_seq_len:]
                seq_len = input_seq.size(1)
                
                # Generate mask
                mask = self.model.generate_causal_mask(seq_len, self.device)
                
                # Forward pass
                logits = self.model(input_seq, mask)  # [1, seq_len, vocab_size]
                
                # Get logits for the last position
                logits = logits[:, -1, :] / temperature  # [1, vocab_size]
                
                # Apply top-k filtering if specified
                if top_k > 0:
                    # Zero out all logits except top-k
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(1, top_k_indices, top_k_logits)
                    logits = logits_filtered
                
                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
            
            # Convert to numpy array
            seq = generated.squeeze(0).cpu().numpy()
        
        return seq
    
    def _save_model(self):
        """Save model checkpoint to file"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_step': self.train_step,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate
        }
        
        torch.save(checkpoint, self.model_path)
        logging.info(f'Model checkpoint saved to {self.model_path}')
    
    def _load_model(self):
        """Load model checkpoint from file"""
        if not os.path.exists(self.model_path):
            logging.warning(f'Model checkpoint not found at {self.model_path}')
            logging.info('Initializing new model instead')
            return
        
        logging.info(f'Loading model from {self.model_path}')
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Load hyperparameters
        self.vocab_size = checkpoint.get('vocab_size', self.vocab_size)
        self.d_model = checkpoint.get('d_model', self.d_model)
        self.num_layers = checkpoint.get('num_layers', self.num_layers)
        self.num_heads = checkpoint.get('num_heads', self.num_heads)
        self.d_ff = checkpoint.get('d_ff', self.d_ff)
        self.max_seq_len = checkpoint.get('max_seq_len', self.max_seq_len)
        self.dropout = checkpoint.get('dropout', self.dropout)
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
        
        # Recreate model with saved hyperparameters
        self.model = TransformerMusicModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout
        ).to(self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Recreate and load optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training step
        self.train_step = checkpoint.get('train_step', 0)
        
        logging.info(f'Model loaded successfully (training step: {self.train_step})')

