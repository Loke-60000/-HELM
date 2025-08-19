import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass
import logging
from shutil import rmtree

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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

class HumanLikeAI:
    def __init__(self, model_name: str = 'roberta-base', device: str = 'auto', hidden_dim: int = 768):
        self.device = setup_device(device)
        self.model_name = model_name
        self.hidden_dim = hidden_dim

        self.shared_state = SharedState()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.conversation_encoder = ConversationEncoder(self.model_name, self.hidden_dim).to(self.device)
        self.memory_classifier = MemoryClassifier(self.hidden_dim).to(self.device)
        self.mood_tracker = MoodTracker().to(self.device)
        self.social_analyzer = SocialContextAnalyzer(self.hidden_dim).to(self.device)
        self.memory_manager = MemoryManager(self.hidden_dim).to(self.device)

        self._setup_model_communication()
        logger.info(f"Initialized HumanLikeAI on {self.device}")
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

    def inference(self, text: str, speaker_id: str = "user") -> Dict:
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