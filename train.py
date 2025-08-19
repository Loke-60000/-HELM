"""
Human-like AI Multi-Model System
Includes: Memory Classification, Mood Tracking, Social Context Analysis, Memory Management

IMPORTANT: This system requires REAL datasets for proper training!

QUICK START:
1) Auto-download raw datasets:
   python train.py --download-datasets

2) Process raw datasets into training CSVs:
   python train.py --process-datasets
   # or explicitly:
   # python train.py --process-datasets \
   #   --custom-meld-dir ./datasets/MELD \
   #   --custom-goemotions-dir ./datasets/GoEmotions \
   #   --custom-mood-file ./datasets/mood_data.csv

3) Train:
   python train.py train

4) Inference:
   python train.py inference --inference-text "I'm feeling great today!"
"""

import os
import io
import json
import shutil
import logging
import argparse
import tarfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---------------------------
# CLI
# ---------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description='Human-like AI Multi-Model Training & Inference System')

    # Mode
    parser.add_argument('mode', choices=['train', 'inference', 'evaluate'], nargs='?',
                        help='Mode to run: train, inference, or evaluate')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=100, help='Warmup steps')

    # Model parameters
    parser.add_argument('--model-name', type=str, default='roberta-base', help='HF model name')
    parser.add_argument('--hidden-dim', type=int, default=768, help='Hidden dim')
    parser.add_argument('--sequence-length', type=int, default=7, help='Sequence length for mood tracking')

    # Data paths (processed)
    parser.add_argument('--meld-path', type=str, default='processed_meld.csv', help='Processed MELD CSV')
    parser.add_argument('--goemotions-path', type=str, default='processed_goemotions.csv', help='Processed GoEmotions CSV')
    parser.add_argument('--mood-path', type=str, default='processed_mood.csv', help='Processed Mood CSV')

    # Raw data base dir
    parser.add_argument('--raw-data-dir', type=str, default='./raw_data', help='Raw datasets dir')

    # Checkpoints and outputs
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint dir')
    parser.add_argument('--load-checkpoint', type=str, default=None, help='Path to load checkpoint from')
    parser.add_argument('--save-interval', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Outputs dir')

    # Device and performance
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'], help='Device')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--pin-memory', action='store_true', help='Pin memory')
    parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--gradient-clip', type=float, default=1.0, help='Grad clipping threshold')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')

    # Special commands
    parser.add_argument('--process-datasets', action='store_true', help='Process raw datasets')
    parser.add_argument('--download-instructions', action='store_true', help='Show dataset setup info')
    parser.add_argument('--download-datasets', action='store_true', help='Auto-download datasets')

    # Custom dataset raw paths for processing
    parser.add_argument('--custom-meld-dir', type=str, help='Dir with MELD CSVs (train/dev/test)')
    parser.add_argument('--custom-goemotions-dir', type=str, help='Dir with GoEmotions TSVs (train/dev/test)')
    parser.add_argument('--custom-mood-file', type=str, help='Path to mood tracking CSV')

    # Inference
    parser.add_argument('--inference-text', type=str, nargs='+', help='Texts for inference')
    parser.add_argument('--inference-file', type=str, help='File with one text per line for inference')

    # Evaluation
    parser.add_argument('--eval-datasets', type=str, nargs='+', help='Datasets to evaluate on')

    return parser.parse_args()

def setup_device(device_arg: str) -> str:
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = device_arg
    logger.info(f"Using device: {device}")
    return device

def setup_directories(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.raw_data_dir, exist_ok=True)
    os.makedirs('./datasets', exist_ok=True)
    logger.info(f"Directories created: {args.checkpoint_dir}, {args.output_dir}, {args.raw_data_dir}, ./datasets")

# ---------------------------
# Shared State and DataClasses
# ---------------------------
@dataclass
class ConversationTurn:
    text: str
    speaker_id: str
    timestamp: datetime
    emotion: Optional[str] = None
    sentiment: Optional[str] = None

@dataclass
class Memory:
    content: str
    importance: float
    emotional_weight: float
    topic_category: str
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    decay_rate: float = 0.1

@dataclass
class MoodState:
    valence: float  # -1 to 1
    arousal: float  # 0 to 1
    timestamp: datetime
    confidence: float = 1.0

@dataclass
class SocialContext:
    relationship_type: str  # friend, family, colleague, stranger
    formality_level: float  # 0 to 1
    group_size: int
    intimacy_level: float  # 0 to 1

class SharedState:
    def __init__(self):
        self.current_mood = MoodState(0.0, 0.5, datetime.now())
        self.active_memories: List[Memory] = []
        self.social_context = SocialContext("unknown", 0.5, 1, 0.5)
        self.conversation_history: List[ConversationTurn] = []
        self.subscribers = {}

    def update_mood(self, new_mood: MoodState):
        self.current_mood = new_mood
        self.notify_subscribers('mood_updated')

    def add_memory(self, memory: Memory):
        self.active_memories.append(memory)
        self.notify_subscribers('memory_added')

    def update_social_context(self, context: SocialContext):
        self.social_context = context
        self.notify_subscribers('social_updated')

    def subscribe(self, event: str, callback):
        if event not in self.subscribers:
            self.subscribers[event] = []
        self.subscribers[event].append(callback)

    def notify_subscribers(self, event: str):
        if event in self.subscribers:
            for callback in self.subscribers[event]:
                callback(self)

# ---------------------------
# Datasets
# ---------------------------
class MELDDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_length: int = 256):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.emotion_map = {
            'neutral': 0, 'joy': 1, 'sadness': 2, 'anger': 3,
            'fear': 4, 'disgust': 5, 'surprise': 6
        }
        self.sentiment_map = {'neutral': 0, 'positive': 1, 'negative': 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer(
            row['Utterance'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'emotion': torch.tensor(self.emotion_map.get(row['Emotion'], 0), dtype=torch.long),
            'sentiment': torch.tensor(self.sentiment_map.get(row['Sentiment'], 0), dtype=torch.long),
            'speaker': row.get('Speaker', ''),
            'dialogue_id': row.get('Dialogue_ID', -1),
        }

class GoEmotionsDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise'
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer(
            row['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        emotions = torch.zeros(len(self.emotion_labels))
        if 'emotions' in row and pd.notna(row['emotions']):
            try:
                emotion_list = eval(row['emotions']) if isinstance(row['emotions'], str) else row['emotions']
                if isinstance(emotion_list, list):
                    for emotion in emotion_list:
                        if emotion in self.emotion_labels:
                            emotions[self.emotion_labels.index(emotion)] = 1
            except Exception:
                emotions[0] = 1
        else:
            emotions[0] = 1

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'emotions': emotions
        }

class MoodDataset(Dataset):
    def __init__(self, mood_data_path: str, sequence_length: int = 7):
        self.data = pd.read_csv(mood_data_path)
        self.sequence_length = sequence_length
        self.mood_map = {'bad': 1, 'meh': 2, 'ok': 3, 'good': 4, 'great': 5}
        self.sequences = self._create_sequences()

    def _create_sequences(self):
        sequences = []
        for user_id in self.data['user_id'].unique():
            user_data = self.data[self.data['user_id'] == user_id].sort_values('date')
            if len(user_data) <= self.sequence_length:
                continue

            for i in range(len(user_data) - self.sequence_length):
                sequence = user_data.iloc[i:i+self.sequence_length]
                target = user_data.iloc[i+self.sequence_length]
                mood_sequence = [self.mood_map.get(mood, 3) for mood in sequence['mood']]
                target_mood = self.mood_map.get(target['mood'], 3)

                uplifting_effect = sequence.get('total_uplifting_effect_of_events', pd.Series([0])).fillna(0).mean()
                depressive_effect = sequence.get('total_depressive_effect_of_events', pd.Series([0])).fillna(0).mean()

                sequences.append({
                    'mood_sequence': torch.tensor(mood_sequence, dtype=torch.float32),
                    'target_mood': torch.tensor(target_mood, dtype=torch.float32),
                    'features': torch.tensor([uplifting_effect, depressive_effect], dtype=torch.float32)
                })
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

# ---------------------------
# Models
# ---------------------------
class ConversationEncoder(nn.Module):
    def __init__(self, model_name: str = 'roberta-base', hidden_dim: int = 768):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        self.conversation_lstm = nn.LSTM(hidden_dim, hidden_dim//2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None \
                        else sequence_output[:, 0, :]
        lstm_out, _ = self.conversation_lstm(sequence_output)
        return {
            'sequence_features': lstm_out,
            'pooled_features': self.dropout(pooled_output),
            'attention_mask': attention_mask
        }

class MemoryClassifier(nn.Module):
    """Includes emotion head for MELD and topic head for GoEmotions."""
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim//2, 64), nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.emotional_weight_head = nn.Sequential(
            nn.Linear(hidden_dim//2, 64), nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.topic_head = nn.Sequential(
            nn.Linear(hidden_dim//2, 64), nn.ReLU(),
            nn.Linear(64, 27)  # GoEmotions multi-label logits
        )
        self.decay_rate_head = nn.Sequential(
            nn.Linear(hidden_dim//2, 32), nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        # Emotion classification for MELD (7 classes)
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim//2, 64), nn.ReLU(),
            nn.Linear(64, 7)
        )

    def forward(self, features):
        shared = self.shared_layers(features)
        return {
            'importance': self.importance_head(shared),
            'emotional_weight': self.emotional_weight_head(shared),
            'topic': self.topic_head(shared),
            'decay_rate': self.decay_rate_head(shared),
            'emotion': self.emotion_head(shared),
        }

class MoodTracker(nn.Module):
    def __init__(self, text_input_dim: int = 768, numerical_input_dim: int = 2, sequence_length: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.text_encoder = nn.Linear(text_input_dim, hidden_dim)
        self.numerical_encoder = nn.Linear(numerical_input_dim, hidden_dim)
        self.mood_lstm = nn.LSTM(hidden_dim + 1, hidden_dim, batch_first=True)
        self.valence_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())
        self.arousal_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.persistence_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, features, mood_history=None, feature_type='text'):
        if feature_type == 'text':
            encoded = self.text_encoder(features)
        else:
            encoded = self.numerical_encoder(features)

        if mood_history is not None:
            if len(mood_history.shape) == 2:
                mood_history = mood_history.unsqueeze(1)
            combined = torch.cat([encoded.unsqueeze(1).expand(-1, mood_history.shape[1], -1), mood_history], dim=2)
            lstm_out, _ = self.mood_lstm(combined)
            features_final = lstm_out[:, -1, :]
        else:
            features_final = encoded

        return {
            'valence': self.valence_head(features_final),
            'arousal': self.arousal_head(features_final),
            'persistence': self.persistence_head(features_final)
        }

class SocialContextAnalyzer(nn.Module):
    """Adds sentiment head (3 classes) to match MELD sentiment labels."""
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        self.relationship_head = nn.Sequential(
            nn.Linear(hidden_dim//2, 64), nn.ReLU(),
            nn.Linear(64, 5)  # stranger, acquaintance, friend, family, romantic
        )
        self.formality_head = nn.Sequential(
            nn.Linear(hidden_dim//2, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.intimacy_head = nn.Sequential(
            nn.Linear(hidden_dim//2, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.group_size_head = nn.Sequential(
            nn.Linear(hidden_dim//2, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        # Sentiment head (3 classes) for MELD sentiment
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_dim//2, 64), nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, features):
        extracted = self.feature_extractor(features)
        return {
            'relationship': self.relationship_head(extracted),
            'formality': self.formality_head(extracted),
            'intimacy': self.intimacy_head(extracted),
            'group_size': self.group_size_head(extracted),
            'sentiment': self.sentiment_head(extracted),
        }

class MemoryManager(nn.Module):
    def __init__(self, feature_dim: int = 768, context_dim: int = 6):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim + context_dim, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.retrieval_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.storage_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.pruning_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, text_features, context_features):
        combined = torch.cat([text_features, context_features], dim=-1)
        fused = self.fusion_layer(combined)
        return {
            'retrieval_score': self.retrieval_head(fused),
            'storage_probability': self.storage_head(fused),
            'pruning_probability': self.pruning_head(fused)
        }

# ---------------------------
# Main System
# ---------------------------
class HumanLikeAISystem:
    def __init__(self, args=None):
        if args is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model_name = 'roberta-base'
            self.hidden_dim = 768
        else:
            self.device = setup_device(args.device)
            self.model_name = args.model_name
            self.hidden_dim = args.hidden_dim
            self.args = args

        self.shared_state = SharedState()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.conversation_encoder = ConversationEncoder(self.model_name, self.hidden_dim).to(self.device)
        self.memory_classifier = MemoryClassifier(self.hidden_dim).to(self.device)
        self.mood_tracker = MoodTracker().to(self.device)
        self.social_analyzer = SocialContextAnalyzer(self.hidden_dim).to(self.device)
        self.memory_manager = MemoryManager(self.hidden_dim).to(self.device)

        self._setup_model_communication()
        logger.info(f"Initialized HumanLikeAISystem on {self.device}")
        logger.info(f"Using model: {self.model_name}")

    def _setup_model_communication(self):
        self.shared_state.subscribe('mood_updated', self._on_mood_updated)
        self.shared_state.subscribe('memory_added', self._on_memory_added)
        self.shared_state.subscribe('social_updated', self._on_social_updated)

    def _on_mood_updated(self, state: SharedState):
        logger.info(f"Mood updated: valence={state.current_mood.valence:.2f}, arousal={state.current_mood.arousal:.2f}")

    def _on_memory_added(self, state: SharedState):
        logger.info(f"New memory added. Total memories: {len(state.active_memories)}")

    def _on_social_updated(self, state: SharedState):
        logger.info(f"Social context updated: {state.social_context.relationship_type}")

    def train_models(self, meld_path: str = None, goemotions_path: str = None, mood_path: str = None,
                     num_epochs: int = 10, batch_size: int = 16, learning_rate: float = 2e-5,
                     num_workers: int = 4, pin_memory: bool = False, mixed_precision: bool = False,
                     gradient_clip: float = 1.0, weight_decay: float = 0.01, save_interval: int = 5):

        logger.info("Starting model training...")

        if hasattr(self, 'args'):
            meld_path = meld_path or self.args.meld_path
            goemotions_path = goemotions_path or self.args.goemotions_path
            mood_path = mood_path or self.args.mood_path
            num_epochs = self.args.epochs
            batch_size = self.args.batch_size
            learning_rate = self.args.learning_rate
            num_workers = self.args.num_workers
            pin_memory = self.args.pin_memory
            mixed_precision = self.args.mixed_precision
            gradient_clip = self.args.gradient_clip
            weight_decay = self.args.weight_decay
            save_interval = self.args.save_interval

        meld_dataset = MELDDataset(meld_path, self.tokenizer)
        goemotions_dataset = GoEmotionsDataset(goemotions_path, self.tokenizer)
        mood_dataset = MoodDataset(mood_path)

        meld_loader = DataLoader(meld_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=pin_memory)
        goemotions_loader = DataLoader(goemotions_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, pin_memory=pin_memory)
        mood_loader = DataLoader(mood_dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=pin_memory)

        if len(meld_dataset) == 0:
            logger.warning("MELD dataset is empty, skipping MELD training")
        if len(goemotions_dataset) == 0:
            logger.warning("GoEmotions dataset is empty, skipping GoEmotions training")
        if len(mood_dataset) == 0:
            logger.warning("Mood dataset is empty, skipping mood training")

        optimizers = {
            'encoder': optim.AdamW(self.conversation_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay),
            'memory': optim.AdamW(self.memory_classifier.parameters(), lr=learning_rate, weight_decay=weight_decay),
            'mood': optim.AdamW(self.mood_tracker.parameters(), lr=learning_rate, weight_decay=weight_decay),
            'social': optim.AdamW(self.social_analyzer.parameters(), lr=learning_rate, weight_decay=weight_decay),
            'manager': optim.AdamW(self.memory_manager.parameters(), lr=learning_rate, weight_decay=weight_decay)
        }
        scaler = torch.amp.GradScaler('cuda') if mixed_precision and self.device == 'cuda' else None

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")

            if len(meld_dataset) > 0:
                self._train_epoch_meld(meld_loader, optimizers, epoch, scaler, gradient_clip)

            if len(goemotions_dataset) > 0:
                self._train_epoch_goemotions(goemotions_loader, optimizers, epoch, scaler, gradient_clip)

            if len(mood_dataset) > 0:
                self._train_epoch_mood(mood_loader, optimizers, epoch, scaler, gradient_clip)

            if (epoch + 1) % save_interval == 0:
                checkpoint_name = f'checkpoint_epoch_{epoch+1}'
                if hasattr(self, 'args'):
                    checkpoint_name = os.path.join(self.args.checkpoint_dir, checkpoint_name)
                self.save_models(checkpoint_name, include_data=False)

        logger.info("Training completed!")

        if hasattr(self, 'args'):
            logger.info(f"Training Summary:")
            logger.info(f"  - Epochs: {num_epochs}")
            logger.info(f"  - Batch size: {batch_size}")
            logger.info(f"  - Learning rate: {learning_rate}")
            logger.info(f"  - MELD samples: {len(meld_dataset)}")
            logger.info(f"  - GoEmotions samples: {len(goemotions_dataset)}")
            logger.info(f"  - Mood samples: {len(mood_dataset)}")
            logger.info(f"  - Device: {self.device}")
            logger.info(f"  - Mixed precision: {scaler is not None}")
            logger.info(f"  - Checkpoints saved every {save_interval} epochs")

    def _train_epoch_meld(self, data_loader, optimizers, epoch, scaler=None, gradient_clip=1.0):
        self.conversation_encoder.train()
        self.memory_classifier.train()
        self.social_analyzer.train()

        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"MELD Epoch {epoch+1}")):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            emotions = batch['emotion'].to(self.device)     # 7 classes
            sentiments = batch['sentiment'].to(self.device) # 3 classes

            # Zero only the optimizers we'll use
            optimizers['encoder'].zero_grad()
            optimizers['memory'].zero_grad()
            optimizers['social'].zero_grad()

            def forward():
                encoded = self.conversation_encoder(input_ids, attention_mask)
                pooled = encoded['pooled_features']
                mem_out = self.memory_classifier(pooled)
                soc_out = self.social_analyzer(pooled)
                emotion_loss = nn.CrossEntropyLoss()(mem_out['emotion'], emotions)
                sentiment_loss = nn.CrossEntropyLoss()(soc_out['sentiment'], sentiments)
                return emotion_loss + sentiment_loss

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    loss = forward()
                scaler.scale(loss).backward()
                
                if gradient_clip > 0:
                    scaler.unscale_(optimizers['encoder'])
                    scaler.unscale_(optimizers['memory'])
                    scaler.unscale_(optimizers['social'])
                    torch.nn.utils.clip_grad_norm_(self.conversation_encoder.parameters(), gradient_clip)
                    torch.nn.utils.clip_grad_norm_(self.memory_classifier.parameters(), gradient_clip)
                    torch.nn.utils.clip_grad_norm_(self.social_analyzer.parameters(), gradient_clip)
                
                scaler.step(optimizers['encoder'])
                scaler.step(optimizers['memory'])
                scaler.step(optimizers['social'])
                scaler.update()
            else:
                loss = forward()
                loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.conversation_encoder.parameters(), gradient_clip)
                    torch.nn.utils.clip_grad_norm_(self.memory_classifier.parameters(), gradient_clip)
                    torch.nn.utils.clip_grad_norm_(self.social_analyzer.parameters(), gradient_clip)
                optimizers['encoder'].step()
                optimizers['memory'].step()
                optimizers['social'].step()

            if batch_idx % 100 == 0:
                logger.info(f'MELD Batch {batch_idx}, Loss: {loss.item():.4f}')

    def _train_epoch_goemotions(self, data_loader, optimizers, epoch, scaler=None, gradient_clip=1.0):
        self.conversation_encoder.train()
        self.memory_classifier.train()

        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"GoEmotions Epoch {epoch+1}")):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            emotions = batch['emotions'].to(self.device)  # multi-label 27

            optimizers['encoder'].zero_grad()
            optimizers['memory'].zero_grad()

            def forward():
                encoded = self.conversation_encoder(input_ids, attention_mask)
                mem_out = self.memory_classifier(encoded['pooled_features'])
                loss = nn.BCEWithLogitsLoss()(mem_out['topic'], emotions)
                return loss

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    loss = forward()
                scaler.scale(loss).backward()
                
                if gradient_clip > 0:
                    scaler.unscale_(optimizers['encoder'])
                    scaler.unscale_(optimizers['memory'])
                    torch.nn.utils.clip_grad_norm_(self.conversation_encoder.parameters(), gradient_clip)
                    torch.nn.utils.clip_grad_norm_(self.memory_classifier.parameters(), gradient_clip)
                
                scaler.step(optimizers['encoder'])
                scaler.step(optimizers['memory'])
                scaler.update()
            else:
                loss = forward()
                loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.conversation_encoder.parameters(), gradient_clip)
                    torch.nn.utils.clip_grad_norm_(self.memory_classifier.parameters(), gradient_clip)
                optimizers['encoder'].step()
                optimizers['memory'].step()

            if batch_idx % 100 == 0:
                logger.info(f'GoEmotions Batch {batch_idx}, Loss: {loss.item():.4f}')

    def _train_epoch_mood(self, data_loader, optimizers, epoch, scaler=None, gradient_clip=1.0):
        self.mood_tracker.train()

        for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Mood Epoch {epoch+1}")):
            mood_sequence = batch['mood_sequence'].to(self.device)
            target_mood = batch['target_mood'].to(self.device)
            features = batch['features'].to(self.device)

            optimizers['mood'].zero_grad()

            def forward():
                mood_outputs = self.mood_tracker(
                    features,
                    mood_history=mood_sequence.unsqueeze(-1),
                    feature_type='numerical'
                )
                target_valence = (target_mood - 3.0) / 2.0  # map [1..5] -> [-1..1]
                loss = nn.MSELoss()(mood_outputs['valence'].squeeze(), target_valence)
                return loss

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    loss = forward()
                scaler.scale(loss).backward()
                
                if gradient_clip > 0:
                    scaler.unscale_(optimizers['mood'])
                    torch.nn.utils.clip_grad_norm_(self.mood_tracker.parameters(), gradient_clip)
                
                scaler.step(optimizers['mood'])
                scaler.update()
            else:
                loss = forward()
                loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.mood_tracker.parameters(), gradient_clip)
                optimizers['mood'].step()

            if batch_idx % 50 == 0:
                logger.info(f'Mood Batch {batch_idx}, Loss: {loss.item():.4f}')

    def inference(self, text: str, speaker_id: str = "user", context: Dict = None) -> Dict:
        self.conversation_encoder.eval()
        self.memory_classifier.eval()
        self.mood_tracker.eval()
        self.social_analyzer.eval()
        self.memory_manager.eval()

        with torch.no_grad():
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                return_tensors='pt'
            ).to(self.device)

            encoded = self.conversation_encoder(
                encoding['input_ids'], encoding['attention_mask']
            )

            memory_results = self.memory_classifier(encoded['pooled_features'])
            mood_results = self.mood_tracker(encoded['pooled_features'], feature_type='text')
            social_results = self.social_analyzer(encoded['pooled_features'])

            context_features = torch.cat([
                mood_results['valence'],
                mood_results['arousal'],
                mood_results['persistence'],
                social_results['formality'],
                social_results['intimacy'],
                social_results['group_size']
            ], dim=-1)

            memory_mgmt_results = self.memory_manager(
                encoded['pooled_features'], context_features
            )

            results = {
                'memory': {
                    'importance': torch.softmax(memory_results['importance'], dim=-1).cpu().numpy(),
                    'emotional_weight': memory_results['emotional_weight'].cpu().numpy(),
                    'topic': torch.softmax(memory_results['topic'], dim=-1).cpu().numpy(),
                    'decay_rate': memory_results['decay_rate'].cpu().numpy()
                },
                'mood': {
                    'valence': mood_results['valence'].cpu().numpy(),
                    'arousal': mood_results['arousal'].cpu().numpy(),
                    'persistence': mood_results['persistence'].cpu().numpy()
                },
                'social': {
                    'relationship': torch.softmax(social_results['relationship'], dim=-1).cpu().numpy(),
                    'formality': social_results['formality'].cpu().numpy(),
                    'intimacy': social_results['intimacy'].cpu().numpy(),
                    'group_size': social_results['group_size'].cpu().numpy()
                },
                'memory_management': {
                    'retrieval_score': memory_mgmt_results['retrieval_score'].cpu().numpy(),
                    'storage_probability': memory_mgmt_results['storage_probability'].cpu().numpy(),
                    'pruning_probability': memory_mgmt_results['pruning_probability'].cpu().numpy()
                }
            }

            new_mood = MoodState(
                valence=float(results['mood']['valence'][0].item()),
                arousal=float(results['mood']['arousal'][0].item()),
                timestamp=datetime.now()
            )
            self.shared_state.update_mood(new_mood)

            if float(results['memory_management']['storage_probability'][0].item()) > 0.7:
                memory = Memory(
                    content=text,
                    importance=float(np.argmax(results['memory']['importance'][0])),
                    emotional_weight=float(results['memory']['emotional_weight'][0].item()),
                    topic_category=str(np.argmax(results['memory']['topic'][0])),
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    decay_rate=float(results['memory']['decay_rate'][0].item())
                )
                self.shared_state.add_memory(memory)

            return results

    def save_models(self, checkpoint_name: str, include_data: bool = True):
        model_dir = checkpoint_name.replace('.pth', '') if checkpoint_name.endswith('.pth') else checkpoint_name
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, 'models.pth')
        checkpoint = {
            'conversation_encoder': self.conversation_encoder.state_dict(),
            'memory_classifier': self.memory_classifier.state_dict(),
            'mood_tracker': self.mood_tracker.state_dict(),
            'social_analyzer': self.social_analyzer.state_dict(),
            'memory_manager': self.memory_manager.state_dict(),
        }
        torch.save(checkpoint, model_path)

        config = {
            'model_name': self.model_name,
            'hidden_dim': self.hidden_dim,
            'device': str(self.device),
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
            'model_architecture': {
                'conversation_encoder': str(self.conversation_encoder),
                'memory_classifier': str(self.memory_classifier),
                'mood_tracker': str(self.mood_tracker),
                'social_analyzer': str(self.social_analyzer),
                'memory_manager': str(self.memory_manager)
            }
        }
        if hasattr(self, 'args'):
            config['training_args'] = {
                'epochs': self.args.epochs,
                'batch_size': self.args.batch_size,
                'learning_rate': self.args.learning_rate,
                'mixed_precision': self.args.mixed_precision,
                'gradient_clip': self.args.gradient_clip,
                'weight_decay': self.args.weight_decay
            }
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        tokenizer_dir = os.path.join(model_dir, 'tokenizer')
        self.tokenizer.save_pretrained(tokenizer_dir)

        state_data = {
            'current_mood': {
                'valence': self.shared_state.current_mood.valence,
                'arousal': self.shared_state.current_mood.arousal,
                'timestamp': self.shared_state.current_mood.timestamp.isoformat(),
                'confidence': self.shared_state.current_mood.confidence
            },
            'active_memories': [
                {
                    'content': mem.content,
                    'importance': mem.importance,
                    'emotional_weight': mem.emotional_weight,
                    'topic_category': mem.topic_category,
                    'created_at': mem.created_at.isoformat(),
                    'last_accessed': mem.last_accessed.isoformat(),
                    'access_count': mem.access_count,
                    'decay_rate': mem.decay_rate
                }
                for mem in self.shared_state.active_memories
            ],
            'social_context': {
                'relationship_type': self.shared_state.social_context.relationship_type,
                'formality_level': self.shared_state.social_context.formality_level,
                'group_size': self.shared_state.social_context.group_size,
                'intimacy_level': self.shared_state.social_context.intimacy_level
            },
            'conversation_history': [
                {
                    'text': turn.text,
                    'speaker_id': turn.speaker_id,
                    'timestamp': turn.timestamp.isoformat(),
                    'emotion': turn.emotion,
                    'sentiment': turn.sentiment
                }
                for turn in self.shared_state.conversation_history
            ]
        }
        with open(os.path.join(model_dir, 'shared_state.json'), 'w') as f:
            json.dump(state_data, f, indent=2)

        if include_data and hasattr(self, 'args'):
            data_dir = os.path.join(model_dir, 'training_data')
            os.makedirs(data_dir, exist_ok=True)
            for dataset_name, dataset_path in [
                ('meld.csv', self.args.meld_path),
                ('goemotions.csv', self.args.goemotions_path),
                ('mood.csv', self.args.mood_path)
            ]:
                if os.path.exists(dataset_path):
                    shutil.copy2(dataset_path, os.path.join(data_dir, dataset_name))

        readme_content = f"""# Human-like AI Model Checkpoint

Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {self.model_name}
Device: {self.device}
Hidden Dimension: {self.hidden_dim}

Files:
- models.pth
- config.json
- shared_state.json
- tokenizer/
- training_data/ (if included)
"""
        with open(os.path.join(model_dir, 'README.md'), 'w') as f:
            f.write(readme_content)

        logger.info(f"Model saved to {model_dir}/ with complete structure")
        return model_dir

    def load_models(self, checkpoint_path: str):
        if os.path.isdir(checkpoint_path):
            model_file = os.path.join(checkpoint_path, 'models.pth')
            config_file = os.path.join(checkpoint_path, 'config.json')
            state_file = os.path.join(checkpoint_path, 'shared_state.json')

            if os.path.exists(model_file):
                checkpoint = torch.load(model_file, map_location=self.device)
                self.conversation_encoder.load_state_dict(checkpoint['conversation_encoder'])
                self.memory_classifier.load_state_dict(checkpoint['memory_classifier'])
                self.mood_tracker.load_state_dict(checkpoint['mood_tracker'])
                self.social_analyzer.load_state_dict(checkpoint['social_analyzer'])
                self.memory_manager.load_state_dict(checkpoint['memory_manager'])
                logger.info(f"Loaded model weights from {model_file}")

            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration: {config.get('timestamp', 'unknown time')}")

            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                mood_data = state_data.get('current_mood', {})
                if mood_data:
                    self.shared_state.current_mood = MoodState(
                        valence=mood_data.get('valence', 0.0),
                        arousal=mood_data.get('arousal', 0.5),
                        timestamp=datetime.fromisoformat(mood_data.get('timestamp', datetime.now().isoformat())),
                        confidence=mood_data.get('confidence', 1.0)
                    )
                self.shared_state.active_memories = []
                for mem_data in state_data.get('active_memories', []):
                    memory = Memory(
                        content=mem_data.get('content', ''),
                        importance=mem_data.get('importance', 0.0),
                        emotional_weight=mem_data.get('emotional_weight', 0.0),
                        topic_category=mem_data.get('topic_category', ''),
                        created_at=datetime.fromisoformat(mem_data.get('created_at', datetime.now().isoformat())),
                        last_accessed=datetime.fromisoformat(mem_data.get('last_accessed', datetime.now().isoformat())),
                        access_count=mem_data.get('access_count', 0),
                        decay_rate=mem_data.get('decay_rate', 0.1)
                    )
                    self.shared_state.active_memories.append(memory)
                logger.info(f"Restored {len(self.shared_state.active_memories)} memories")
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.conversation_encoder.load_state_dict(checkpoint['conversation_encoder'])
            self.memory_classifier.load_state_dict(checkpoint['memory_classifier'])
            self.mood_tracker.load_state_dict(checkpoint['mood_tracker'])
            self.social_analyzer.load_state_dict(checkpoint['social_analyzer'])
            self.memory_manager.load_state_dict(checkpoint['memory_manager'])
            logger.info(f"Loaded models from {checkpoint_path}")

        logger.info("Model loading completed")

# ---------------------------
# Data Preprocessing
# ---------------------------
class DataPreprocessor:
    @staticmethod
    def process_meld_data(meld_dir: str, output_path: str):
        logger.info(f"Processing MELD data from {meld_dir}")

        required_splits = ['train', 'dev', 'test']
        required_cols = {'Utterance', 'Speaker', 'Emotion', 'Sentiment', 'Dialogue_ID'}
        all_data = []

        for split in required_splits:
            file_path = os.path.join(meld_dir, f'{split}_sent_emo.csv')
            if not os.path.exists(file_path):
                logger.warning(f"  Missing MELD file: {file_path}")
                continue

            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                raise ValueError(f"Failed to read MELD {split} CSV at {file_path}: {e}")

            if df is None or df.empty:
                raise ValueError(f"MELD {split} CSV appears empty: {file_path} "
                                 f"(did the download save an HTML error page?)")

            present = set(df.columns)
            if not required_cols.issubset(present):
                raise ValueError(
                    f"MELD {split} CSV missing required columns.\n"
                    f"  required: {sorted(list(required_cols))}\n"
                    f"  found:    {sorted(list(present))}\n"
                    f"  file:     {file_path}"
                )

            df['split'] = split
            all_data.append(df)
            logger.info(f"Loaded {len(df)} samples from {split} split")

        if not all_data:
            raise FileNotFoundError(f"No valid MELD CSV files found in {meld_dir}. Re-download the dataset.")

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Processed MELD data saved to {output_path} ({len(combined_df)} total samples)")

        try:
            emo_counts = combined_df['Emotion'].value_counts().to_dict()
            sent_counts = combined_df['Sentiment'].value_counts().to_dict()
            logger.info(f"Emotion distribution: {emo_counts}")
            logger.info(f"Sentiment distribution: {sent_counts}")
        except Exception as e:
            logger.warning(f"Could not compute MELD distributions: {e}")

        return output_path

    @staticmethod
    def process_goemotions_data(goemotions_dir: str, output_path: str):
        logger.info(f"Processing GoEmotions data from {goemotions_dir}")

        emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise'
        ]

        all_data = []
        for split in ['train', 'dev', 'test']:
            file_path = os.path.join(goemotions_dir, f'{split}.tsv')
            if not os.path.exists(file_path):
                logger.warning(f"  Missing GoEmotions file: {file_path}")
                continue

            try:
                df = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'emotion_ids', 'id'])
            except Exception as e:
                raise ValueError(f"Failed to read GoEmotions {split} TSV at {file_path}: {e}")

            if df is None or df.empty or 'emotion_ids' not in df.columns or 'text' not in df.columns:
                raise ValueError(f"GoEmotions {split} TSV invalid/empty: {file_path}")

            processed_data = []
            for _, row in df.iterrows():
                text = row['text']
                raw_ids = str(row['emotion_ids']) if pd.notna(row['emotion_ids']) else ''
                try:
                    emotion_ids = [int(x) for x in raw_ids.split(',') if x.strip() != '']
                except Exception:
                    emotion_ids = []
                emotions = [emotion_labels[i] for i in emotion_ids if 0 <= i < len(emotion_labels)]
                processed_data.append({'text': text, 'emotions': str(emotions), 'split': split})

            split_df = pd.DataFrame(processed_data)
            all_data.append(split_df)
            logger.info(f"Loaded {len(split_df)} samples from {split} split")

        if not all_data:
            raise FileNotFoundError(f"No valid GoEmotions TSV files found in {goemotions_dir}. Re-download the dataset.")

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Processed GoEmotions data saved to {output_path} ({len(combined_df)} total samples)")
        return output_path

    @staticmethod
    def process_mood_data(mood_file_path: str, output_path: str):
        logger.info(f"Processing mood data from {mood_file_path}")

        if not os.path.exists(mood_file_path):
            logger.info("Mood Dataset Options:")
            logger.info("1. Daylio exports: https://daylio.com")
            logger.info("2. Mood Meter data: https://moodmeterapp.com")
            logger.info("3. LoCoMo dataset: https://arxiv.org/abs/2402.17753")
            logger.info("4. K-EmoCon dataset: https://www.nature.com/articles/s41597-020-00630-y")
            logger.info("5. Your own mood tracking CSV with columns: user_id, date, mood, events")
            raise FileNotFoundError(f"Mood data file not found: {mood_file_path}")

        df = pd.read_csv(mood_file_path)

        column_mapping = {
            'user': 'user_id',
            'date': 'date',
            'mood': 'mood',
            'valence': 'valence',
            'arousal': 'arousal',
            'events': 'total_uplifting_effect_of_events'
        }
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})

        if 'user_id' not in df.columns:
            df['user_id'] = 'user_1'
        if 'date' not in df.columns:
            df['date'] = datetime.now().strftime('%Y-%m-%d')
        if 'mood' not in df.columns:
            df['mood'] = 'ok'
        if 'total_uplifting_effect_of_events' not in df.columns:
            df['total_uplifting_effect_of_events'] = 0.0
        if 'total_depressive_effect_of_events' not in df.columns:
            df['total_depressive_effect_of_events'] = 0.0

        df.to_csv(output_path, index=False)
        logger.info(f"Processed mood data saved to {output_path} ({len(df)} samples)")
        return output_path

# ---------------------------
# Dataset Downloader
# ---------------------------
class DatasetDownloader:
    @staticmethod
    def download_file(url: str, dest_path: str, desc: str = "Downloading", min_bytes: int = 10_000):
        """Download a file with progress, verify HTTP status and minimum size."""
        try:
            resp = requests.get(url, stream=True, allow_redirects=True, timeout=60)
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to request {url}: {e}")

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code} for {url}")

        total_size = int(resp.headers.get('content-length', 0)) or None
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        with open(dest_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        size = os.path.getsize(dest_path)
        if size < min_bytes:
            try:
                os.remove(dest_path)
            except Exception:
                pass
            raise RuntimeError(f"Downloaded file too small ({size} bytes): {dest_path} from {url}")
        return dest_path

    @staticmethod
    def download_and_extract_tar(url: str, dest_dir: str, desc: str = "Downloading TAR", min_bytes: int = 100_000):
        """Download a tar.gz into memory (size-checked) and extract to dest_dir."""
        try:
            resp = requests.get(url, stream=True, allow_redirects=True, timeout=120)
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to request {url}: {e}")
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code} for {url}")

        data = io.BytesIO()
        total_size = int(resp.headers.get('content-length', 0)) or None
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    data.write(chunk)
                    pbar.update(len(chunk))
        size = data.tell()
        if size < min_bytes:
            raise RuntimeError(f"TAR too small ({size} bytes) from {url}")

        data.seek(0)
        os.makedirs(dest_dir, exist_ok=True)
        with tarfile.open(fileobj=data, mode='r:gz') as tar:
            tar.extractall(path=dest_dir)
        return dest_dir

    @staticmethod
    def download_meld_dataset(data_dir: str = './datasets'):
        logger.info("Downloading MELD dataset...")
        meld_dir = os.path.join(data_dir, 'MELD')
        os.makedirs(meld_dir, exist_ok=True)

        # Attempt 1: GitHub raw CSVs
        base_url = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/"
        files_to_download = {
            'train_sent_emo.csv': base_url + 'train_sent_emo.csv',
            'dev_sent_emo.csv':   base_url + 'dev_sent_emo.csv',
            'test_sent_emo.csv':  base_url + 'test_sent_emo.csv'
        }

        ok = True
        for filename, url in files_to_download.items():
            dest_path = os.path.join(meld_dir, filename)
            min_bytes = 50_000  # MELD CSVs are much larger than this
            if os.path.exists(dest_path):
                if os.path.getsize(dest_path) < min_bytes:
                    logger.warning(f"{filename} is suspiciously small; re-downloading...")
                    try:
                        os.remove(dest_path)
                    except Exception:
                        pass
                else:
                    logger.info(f"  {filename} already exists, size OK.")
                    continue

            try:
                DatasetDownloader.download_file(url, dest_path, f"MELD/{filename}", min_bytes=min_bytes)
                logger.info(f"  ✓ Downloaded {filename}")
            except Exception as e:
                logger.error(f"  ✗ Failed to download {filename}: {e}")
                ok = False

        # If any failed, try tarball fallback (UMich mirror)
        if not all(os.path.exists(os.path.join(meld_dir, f)) for f in files_to_download.keys()):
            logger.info("Trying MELD tarball fallback (UMich mirror)...")
            try:
                tmp_extract = os.path.join(meld_dir, "_tmp_extract")
                DatasetDownloader.download_and_extract_tar(
                    "https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz",
                    tmp_extract,
                    desc="MELD.Raw.tar.gz",
                    min_bytes=5_000_000  # expect several MB+
                )
                # Find and copy the CSVs
                def find_file(root, name):
                    for r, _, files in os.walk(root):
                        for f in files:
                            if f == name:
                                return os.path.join(r, f)
                    return None

                for name in files_to_download.keys():
                    src = find_file(tmp_extract, name)
                    if src:
                        shutil.copy2(src, os.path.join(meld_dir, name))
                        logger.info(f"  ✓ Extracted {name} from tarball")
                    else:
                        ok = False
                        logger.error(f"  ✗ Could not find {name} inside tarball contents")
            except Exception as e:
                logger.error(f"  ✗ Tarball fallback failed: {e}")
                ok = False
            finally:
                # Clean temp extract
                tmp_dir = os.path.join(meld_dir, "_tmp_extract")
                if os.path.isdir(tmp_dir):
                    shutil.rmtree(tmp_dir, ignore_errors=True)

        downloaded_files = [f for f in files_to_download.keys() if os.path.exists(os.path.join(meld_dir, f))]
        if downloaded_files:
            logger.info(f"MELD dataset ready: {len(downloaded_files)}/3 files")
        else:
            logger.warning("MELD dataset not ready. Please provide the CSVs manually in ./datasets/MELD/")
        return len(downloaded_files) == 3

    @staticmethod
    def download_goemotions_dataset(data_dir: str = './datasets'):
        logger.info("Downloading GoEmotions dataset...")
        goemotions_dir = os.path.join(data_dir, 'GoEmotions')
        os.makedirs(goemotions_dir, exist_ok=True)

        base_url = "https://storage.googleapis.com/gresearch/goemotions/data/"
        mirror = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"
        files_to_download = ['train.tsv', 'dev.tsv', 'test.tsv']

        ok = True
        for filename in files_to_download:
            dest_path = os.path.join(goemotions_dir, filename)
            min_bytes = 10_000

            if os.path.exists(dest_path):
                if os.path.getsize(dest_path) < min_bytes:
                    logger.warning(f"{filename} is too small; re-downloading...")
                    try:
                        os.remove(dest_path)
                    except Exception:
                        pass
                else:
                    logger.info(f"  {filename} already exists, size OK.")
                    continue

            try:
                DatasetDownloader.download_file(base_url + filename, dest_path, f"GoEmotions/{filename}", min_bytes=min_bytes)
                logger.info(f"  ✓ Downloaded {filename}")
            except Exception as e:
                logger.error(f"  ✗ Failed from GCS: {e}")
                logger.info(f"  Trying mirror...")
                try:
                    DatasetDownloader.download_file(mirror + filename, dest_path, f"GoEmotions/{filename}", min_bytes=min_bytes)
                    logger.info(f"  ✓ Downloaded {filename} from mirror")
                except Exception as e2:
                    logger.error(f"  ✗ Failed from mirror: {e2}")
                    ok = False

        downloaded_files = [f for f in files_to_download if os.path.exists(os.path.join(goemotions_dir, f))]
        if downloaded_files:
            logger.info(f"GoEmotions dataset ready: {len(downloaded_files)}/3 files")
        return len(downloaded_files) == 3 and ok

    @staticmethod
    def download_sample_mood_dataset(data_dir: str = './datasets'):
        logger.info("Creating sample mood dataset...")
        mood_file = os.path.join(data_dir, 'mood_data.csv')

        if os.path.exists(mood_file):
            if os.path.getsize(mood_file) >= 100:  # sanity
                logger.info("  Mood dataset already exists")
                return True

        sample_data = []
        moods = ['bad', 'meh', 'ok', 'good', 'great']

        for user_id in range(1, 11):
            base_date = datetime.now() - timedelta(days=30)
            for day in range(30):
                date = base_date + timedelta(days=day)
                mood = np.random.choice(moods)
                uplifting = float(np.random.uniform(0, 5))
                depressive = float(np.random.uniform(0, 3))
                sample_data.append({
                    'user_id': f'user_{user_id}',
                    'date': date.strftime('%Y-%m-%d'),
                    'mood': mood,
                    'total_uplifting_effect_of_events': uplifting,
                    'total_depressive_effect_of_events': depressive
                })
        df = pd.DataFrame(sample_data)
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(mood_file, index=False)
        logger.info(f"  ✓ Created sample mood dataset with {len(df)} entries")
        return True

    @staticmethod
    def download_all_datasets(data_dir: str = './datasets'):
        logger.info("=" * 60)
        logger.info("AUTOMATIC DATASET DOWNLOAD")
        logger.info("=" * 60)

        success = []
        failed = []

        if DatasetDownloader.download_meld_dataset(data_dir):
            success.append('MELD')
        else:
            failed.append('MELD')

        if DatasetDownloader.download_goemotions_dataset(data_dir):
            success.append('GoEmotions')
        else:
            failed.append('GoEmotions')

        if DatasetDownloader.download_sample_mood_dataset(data_dir):
            success.append('Mood')
        else:
            failed.append('Mood')

        logger.info("=" * 60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Successfully downloaded: {success}")
        if failed:
            logger.warning(f"Failed to download: {failed}")
            logger.info("Please download failed datasets manually")
        else:
            logger.info("All datasets downloaded successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Process datasets: python train.py --process-datasets")
            logger.info("2. Train model: python train.py train")

        return len(failed) == 0

    @staticmethod
    def setup_datasets(data_dir: str = './datasets'):
        os.makedirs(data_dir, exist_ok=True)

        logger.info("=" * 60)
        logger.info("DATASET SETUP REQUIRED")
        logger.info("=" * 60)

        datasets_info = {
            'MELD': {
                'description': 'Emotion recognition in conversations',
                'source': 'https://github.com/declare-lab/MELD',
                'files': ['train_sent_emo.csv', 'dev_sent_emo.csv', 'test_sent_emo.csv'],
                'location': os.path.join(data_dir, 'MELD'),
                'paper': 'https://arxiv.org/abs/1810.02508'
            },
            'GoEmotions': {
                'description': 'Fine-grained emotion classification',
                'source': 'https://github.com/google-research/google-research/tree/master/goemotions',
                'files': ['train.tsv', 'dev.tsv', 'test.tsv'],
                'location': os.path.join(data_dir, 'GoEmotions'),
                'paper': 'https://arxiv.org/abs/2005.00547'
            }
        }

        for dataset_name, info in datasets_info.items():
            logger.info(f"\n{dataset_name}:")
            logger.info(f"  Description: {info['description']}")
            logger.info(f"  Source: {info['source']}")
            logger.info(f"  Paper: {info['paper']}")
            logger.info(f"  Required files: {info['files']}")
            logger.info(f"  Location: {info['location']}")
            os.makedirs(info['location'], exist_ok=True)

            existing_files = []
            for file in info['files']:
                if os.path.exists(os.path.join(info['location'], file)):
                    existing_files.append(file)

            if existing_files:
                logger.info(f"  ✓ Found: {existing_files}")
            else:
                logger.info(f"  ✗ Missing: {info['files']}")

        logger.info("=" * 60)
        logger.info("AUTOMATIC DOWNLOAD OPTION:")
        logger.info("=" * 60)
        logger.info("Run: python train.py --download-datasets")
        logger.info("This will automatically download all required datasets")

        return datasets_info

# ---------------------------
# Data checks (processed)
# ---------------------------
def check_and_prepare_data(args):
    data_paths = {
        'meld': args.meld_path,
        'goemotions': args.goemotions_path,
        'mood': args.mood_path
    }

    missing = [name for name, path in data_paths.items() if not os.path.exists(path)]
    for name, path in data_paths.items():
        if not os.path.exists(path):
            logger.warning(f"Dataset not found: {path}")

    if missing:
        logger.error(f"Missing datasets: {missing}")
        logger.info("\nTo get the real datasets:")
        if 'meld' in missing:
            logger.info("\nMELD Dataset:")
            logger.info("  1. Ensure you have: train_sent_emo.csv, dev_sent_emo.csv, test_sent_emo.csv in ./datasets/MELD/")
            logger.info("  2. Process: python train.py --process-datasets")
        if 'goemotions' in missing:
            logger.info("\nGoEmotions Dataset:")
            logger.info("  1. Ensure you have: train.tsv, dev.tsv, test.tsv in ./datasets/GoEmotions/")
            logger.info("  2. Process: python train.py --process-datasets")
        if 'mood' in missing:
            logger.info("\nMood Dataset Options:")
            logger.info("  1. Use your own Daylio export (CSV format)")
            logger.info("  2. Or rely on generated sample at ./datasets/mood_data.csv")
            logger.info("  3. Process: python train.py --process-datasets")

        logger.info("\nComplete workflow:")
        logger.info("python train.py --download-instructions")
        logger.info("python train.py --download-datasets")
        logger.info("python train.py --process-datasets")
        logger.info("python train.py train")
        raise FileNotFoundError("Required datasets not found. Please download/process datasets.")
    return args

# ---------------------------
# Advanced Inference (optional)
# ---------------------------
class AdvancedInference:
    def __init__(self, system: HumanLikeAISystem):
        self.system = system
        self.conversation_context = []
        self.memory_bank = []

    def contextual_inference(self, text: str, speaker_id: str = "user", conversation_history: List[ConversationTurn] = None) -> Dict:
        turn = ConversationTurn(text, speaker_id, datetime.now())
        self.conversation_context.append(turn)

        results = self.system.inference(text, speaker_id)
        relevant_memories = self._retrieve_memories(text, results)
        adjusted_results = self._adjust_with_context(results, relevant_memories)
        self._update_memory_bank(turn, adjusted_results)

        return {
            **adjusted_results,
            'retrieved_memories': relevant_memories,
            'conversation_context_length': len(self.conversation_context)
        }

    def _retrieve_memories(self, query: str, current_results: Dict) -> List[Dict]:
        relevant_memories = []
        for memory in self.memory_bank:
            sim = self._calculate_similarity(query, memory['content'])
            time_diff = (datetime.now() - memory['created_at']).days
            decay_factor = np.exp(-memory['decay_rate'] * time_diff)
            relevance_score = sim * decay_factor * memory['importance']
            if relevance_score > 0.3:
                relevant_memories.append({**memory, 'relevance_score': relevance_score})
        relevant_memories.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_memories[:5]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1.intersection(words2)) / len(words1.union(words2))

    def _adjust_with_context(self, results: Dict, memories: List[Dict]) -> Dict:
        adjusted = results.copy()
        if len(self.conversation_context) > 1:
            recent_mood_trend = self._calculate_mood_trend()
            adjusted['mood']['valence'] = (adjusted['mood']['valence'] + recent_mood_trend) / 2
        if memories:
            avg_memory_importance = np.mean([m['importance'] for m in memories])
            if avg_memory_importance > 1.5:
                adjusted['memory']['importance_boost'] = 0.2
        return adjusted

    def _calculate_mood_trend(self) -> float:
        if len(self.conversation_context) < 2:
            return 0.0
        recent_turns = self.conversation_context[-3:]
        positive_words = ['good', 'great', 'happy', 'excited', 'love', 'amazing', 'wonderful']
        negative_words = ['bad', 'terrible', 'sad', 'angry', 'hate', 'awful', 'worried']
        sentiment_score = 0
        for turn in recent_turns:
            words = turn.text.lower().split()
            for w in words:
                if w in positive_words:
                    sentiment_score += 1
                elif w in negative_words:
                    sentiment_score -= 1
        return np.tanh(sentiment_score / len(recent_turns))

    def _update_memory_bank(self, turn: ConversationTurn, results: Dict):
        storage_prob = results['memory_management']['storage_probability'][0]
        if storage_prob > 0.5:
            memory_entry = {
                'content': turn.text,
                'speaker': turn.speaker_id,
                'created_at': turn.timestamp,
                'last_accessed': turn.timestamp,
                'importance': float(np.argmax(results['memory']['importance'][0])),
                'emotional_weight': float(results['memory']['emotional_weight'][0]),
                'topic_category': int(np.argmax(results['memory']['topic'][0])),
                'decay_rate': float(results['memory']['decay_rate'][0]),
                'access_count': 1,
                'mood_at_creation': {
                    'valence': float(results['mood']['valence'][0].item()),
                    'arousal': float(results['mood']['arousal'][0].item())
                }
            }
            self.memory_bank.append(memory_entry)
            logger.info(f"Stored new memory: {turn.text[:50]}...")

        if len(self.memory_bank) > 1000:
            self._prune_memories()

    def _prune_memories(self):
        current_time = datetime.now()
        pruning_candidates = []
        for i, memory in enumerate(self.memory_bank):
            time_diff = (current_time - memory['last_accessed']).days
            pruning_score = (
                time_diff * memory['decay_rate'] * 0.1 +
                (1.0 - memory['importance'] / 2.0) * 0.3 +
                (1.0 / (memory['access_count'] + 1)) * 0.2
            )
            pruning_candidates.append((i, pruning_score))
        pruning_candidates.sort(key=lambda x: x[1], reverse=True)
        num_to_prune = len(self.memory_bank) // 5
        indices_to_remove = [idx for idx, _ in pruning_candidates[:num_to_prune]]
        indices_to_remove.sort(reverse=True)
        for idx in indices_to_remove:
            removed = self.memory_bank.pop(idx)
            logger.info(f"Pruned memory: {removed['content'][:30]}...")
        logger.info(f"Pruned {num_to_prune} memories. Memory bank size: {len(self.memory_bank)}")

# ---------------------------
# Evaluation
# ---------------------------
class ModelEvaluator:
    def __init__(self, system: HumanLikeAISystem):
        self.system = system

    def evaluate_emotion_classification(self, test_data_path: str) -> Dict:
        test_data = pd.read_csv(test_data_path)
        predictions = []
        ground_truth = []
        emotion_map = {'neutral': 0, 'joy': 1, 'sadness': 2, 'anger': 3, 'fear': 4, 'disgust': 5, 'surprise': 6}

        for _, row in test_data.iterrows():
            results = self.system.inference(row['Utterance'])
            # Placeholder: mapping importance to "emotion" for demo purposes
            predicted_emotion = int(np.argmax(results['memory']['importance'][0]))
            true_emotion = emotion_map.get(row['Emotion'], 0)
            predictions.append(predicted_emotion)
            ground_truth.append(true_emotion)

        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        accuracy = accuracy_score(ground_truth, predictions)
        report = classification_report(ground_truth, predictions)
        cm = confusion_matrix(ground_truth, predictions)
        return {'accuracy': accuracy, 'classification_report': report, 'confusion_matrix': cm.tolist()}

    def evaluate_mood_prediction(self, test_data_path: str) -> Dict:
        test_data = pd.read_csv(test_data_path)
        mse_valence = []
        mse_arousal = []

        for _, row in test_data.iterrows():
            text_content = row['text'] if 'text' in row else (row['content'] if 'content' in row else str(row.iloc[0]))
            results = self.system.inference(text_content)
            predicted_valence = results['mood']['valence'][0].item()
            predicted_arousal = results['mood']['arousal'][0].item()

            if 'true_valence' in row and 'true_arousal' in row:
                mse_valence.append((predicted_valence - row['true_valence']) ** 2)
                mse_arousal.append((predicted_arousal - row['true_arousal']) ** 2)

        return {
            'mse_valence': float(np.mean(mse_valence)) if mse_valence else None,
            'mse_arousal': float(np.mean(mse_arousal)) if mse_arousal else None,
            'rmse_valence': float(np.sqrt(np.mean(mse_valence))) if mse_valence else None,
            'rmse_arousal': float(np.sqrt(np.mean(mse_arousal))) if mse_arousal else None
        }

    def run_comprehensive_evaluation(self, test_datasets: Dict[str, str]) -> Dict:
        results = {}
        if 'emotion' in test_datasets:
            results['emotion_classification'] = self.evaluate_emotion_classification(test_datasets['emotion'])
        if 'mood' in test_datasets:
            results['mood_prediction'] = self.evaluate_mood_prediction(test_datasets['mood'])
        return results

# ---------------------------
# Mode Runners
# ---------------------------
def run_training(args):
    logger.info("Starting training mode...")
    setup_directories(args)
    args = check_and_prepare_data(args)

    system = HumanLikeAISystem(args)

    if args.load_checkpoint:
        system.load_models(args.load_checkpoint)
        logger.info(f"Loaded checkpoint: {args.load_checkpoint}")

    system.train_models()

    final_model_dir = os.path.join(args.checkpoint_dir, 'final_model')
    system.save_models(final_model_dir, include_data=True)

    evaluator = ModelEvaluator(system)
    eval_results = evaluator.run_comprehensive_evaluation({
        'emotion': args.meld_path,
        'mood': args.mood_path
    })
    results_path = os.path.join(args.output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2, default=str)
    logger.info(f"Training completed! Results saved to {results_path}")

def run_inference(args):
    logger.info("Starting inference mode...")
    system = HumanLikeAISystem(args)

    if args.load_checkpoint:
        system.load_models(args.load_checkpoint)
        logger.info(f"Loaded checkpoint: {args.load_checkpoint}")

    texts = []
    if args.inference_text:
        texts.extend(args.inference_text)
    if args.inference_file and os.path.exists(args.inference_file):
        with open(args.inference_file, 'r') as f:
            texts.extend([line.strip() for line in f if line.strip()])

    if not texts:
        texts = [
            "I'm really excited about this new project!",
            "I had a terrible day at work today.",
            "Let's grab coffee sometime this week.",
            "I'm worried about the presentation tomorrow.",
            "That movie was absolutely amazing!"
        ]

    results = []
    for i, text in enumerate(texts):
        logger.info(f"Processing text {i+1}/{len(texts)}: {text[:50]}...")
        result = system.inference(text)
        processed_result = {
            'text': text,
            'mood': {
                'valence': float(result['mood']['valence'][0].item()),
                'arousal': float(result['mood']['arousal'][0].item()),
                'persistence': float(result['mood']['persistence'][0].item())
            },
            'memory': {
                'importance': int(np.argmax(result['memory']['importance'][0])),
                'emotional_weight': float(result['memory']['emotional_weight'][0].item()),
                'topic': int(np.argmax(result['memory']['topic'][0])),
                'decay_rate': float(result['memory']['decay_rate'][0].item())
            },
            'social': {
                'relationship': int(np.argmax(result['social']['relationship'][0])),
                'formality': float(result['social']['formality'][0].item()),
                'intimacy': float(result['social']['intimacy'][0].item()),
                'group_size': float(result['social']['group_size'][0].item())
            },
            'memory_management': {
                'retrieval_score': float(result['memory_management']['retrieval_score'][0].item()),
                'storage_probability': float(result['memory_management']['storage_probability'][0].item()),
                'pruning_probability': float(result['memory_management']['pruning_probability'][0].item())
            }
        }
        results.append(processed_result)
        print(f"\nInput: {text}")
        print(f"Mood - Valence: {processed_result['mood']['valence']:.3f}, Arousal: {processed_result['mood']['arousal']:.3f}")
        print(f"Memory - Importance: {processed_result['memory']['importance']}, Storage Prob: {processed_result['memory_management']['storage_probability']:.3f}")
        print(f"Social - Relationship: {processed_result['social']['relationship']}, Formality: {processed_result['social']['formality']:.3f}")
        print("-" * 50)

    if hasattr(args, 'output_dir'):
        results_path = os.path.join(args.output_dir, 'inference_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Inference results saved to {results_path}")

def run_evaluation(args):
    logger.info("Starting evaluation mode...")
    system = HumanLikeAISystem(args)

    if args.load_checkpoint:
        system.load_models(args.load_checkpoint)
        logger.info(f"Loaded checkpoint: {args.load_checkpoint}")
    else:
        logger.error("Evaluation mode requires a trained model. Use --load-checkpoint")
        return

    evaluator = ModelEvaluator(system)

    eval_datasets = {}
    if args.eval_datasets:
        for dataset_path in args.eval_datasets:
            dataset_name = os.path.basename(dataset_path).split('.')[0]
            eval_datasets[dataset_name] = dataset_path
    else:
        eval_datasets = {
            'emotion': args.meld_path,
            'mood': args.mood_path
        }

    results = evaluator.run_comprehensive_evaluation(eval_datasets)
    print("\nEvaluation Results:")
    print(json.dumps(results, indent=2, default=str))

    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Evaluation results saved to {results_path}")

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    try:
        args = parse_arguments()

        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        if args.download_instructions:
            DatasetDownloader.setup_datasets()
            raise SystemExit(0)

        if args.download_datasets:
            if DatasetDownloader.download_all_datasets():
                logger.info("\nDatasets downloaded! Now process them:")
                logger.info("python train.py --process-datasets")
            else:
                logger.warning("\nSome datasets failed to download. Check the logs above.")
            raise SystemExit(0)

        if args.process_datasets:
            logger.info("Processing datasets...")
            preprocessor = DataPreprocessor()

            if not args.custom_meld_dir and os.path.exists('./datasets/MELD'):
                args.custom_meld_dir = './datasets/MELD'
            if not args.custom_goemotions_dir and os.path.exists('./datasets/GoEmotions'):
                args.custom_goemotions_dir = './datasets/GoEmotions'
            if not args.custom_mood_file and os.path.exists('./datasets/mood_data.csv'):
                args.custom_mood_file = './datasets/mood_data.csv'

            if args.custom_meld_dir:
                args.meld_path = preprocessor.process_meld_data(args.custom_meld_dir, args.meld_path)
            if args.custom_goemotions_dir:
                args.goemotions_path = preprocessor.process_goemotions_data(args.custom_goemotions_dir, args.goemotions_path)
            if args.custom_mood_file:
                args.mood_path = preprocessor.process_mood_data(args.custom_mood_file, args.mood_path)

            logger.info("Dataset processing completed!")
            logger.info("\nNext step: Train the model")
            logger.info("python train.py train")
            raise SystemExit(0)

        if not args.mode:
            logger.error("\nError: mode argument is required for training/inference/evaluation")
            logger.info("\nQuick start:")
            logger.info("1. Download datasets: python train.py --download-datasets")
            logger.info("2. Process datasets: python train.py --process-datasets")
            logger.info("3. Train model: python train.py train")
            logger.info("4. Run inference: python train.py inference --inference-text 'Hello, how are you?'")
            raise SystemExit(1)

        if args.mode == 'train':
            run_training(args)
        elif args.mode == 'inference':
            run_inference(args)
        elif args.mode == 'evaluate':
            run_evaluation(args)

    except SystemExit:
        raise
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        raise
