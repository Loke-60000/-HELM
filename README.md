# HELM Core - Human-like Emotional Learning Model

A multi-model neural architecture that simulates human-like emotional intelligence through memory classification, mood tracking, social context awareness, and dynamic memory management. HELM Core serves as the foundational technology for building emotionally intelligent conversational AI systems.

## Overview

HELM Core implements a modular approach to human-like AI behavior using five specialized neural networks that communicate through a shared state system:

- **Memory Classifier**: Categorizes conversation importance, emotional weight, topics, and decay rates
- **Mood Tracker**: Predicts emotional valence, arousal, and mood persistence over time  
- **Social Context Analyzer**: Analyzes relationships, formality levels, and group dynamics
- **Memory Manager**: Handles memory storage, retrieval, and pruning decisions
- **Conversation Encoder**: Shared transformer-based feature extraction for all models

## Architecture

The system uses a hub-and-spoke architecture where models process conversations in parallel and share information through a centralized state manager. This enables cross-model learning and emergent human-like behaviors.

```
Conversation Input → Conversation Encoder → Shared Feature Space
                                          ↓
    ┌─────────────────────────────────────────────────────────┐
    │  Memory      Mood        Social       Memory            │
    │  Classifier  Tracker     Analyzer     Manager           │
    └─────────────────────────────────────────────────────────┘
                                          ↓
                    Unified Human-like Response
```

## Key Features

- **Multi-task Learning**: Joint training across emotion recognition, mood prediction, and social understanding
- **Dynamic Memory Management**: Automatic storage, retrieval, and pruning based on importance and decay
- **Contextual Adaptation**: Responses adapt based on conversation history and relationship context
- **Cross-model Communication**: Models influence each other through shared emotional and social state
- **Temporal Modeling**: Tracks mood changes and relationship evolution over time

## Quick Start

### Training

```python
from helm_core import HumanLikeAISystem

# Initialize system
system = HumanLikeAISystem()

# Train on your datasets
system.train_models(
    meld_path='data/meld_conversations.csv',
    goemotions_path='data/goemotions_emotions.csv', 
    mood_path='data/mood_tracking.csv',
    num_epochs=10
)

# Save trained models
system.save_models('trained_helm_core')
```

### Inference

```python
# Load trained system
system = HumanLikeAISystem()
system.load_models('trained_helm_core.pth')

# Run inference
results = system.inference("I'm really excited about this project!")
print(f"Mood: {results['mood']['valence'][0]:.3f} valence, {results['mood']['arousal'][0]:.3f} arousal")
print(f"Memory importance: {results['memory']['importance']}")
print(f"Social context: {results['social']['relationship']}")
```

### Integration Example

Here's how HELM Core could be integrated with existing language models like GPT-4.1:

```python
import openai
from helm_core import HumanLikeAISystem

# Initialize HELM Core
helm = HumanLikeAISystem()
helm.load_models('trained_helm_core.pth')

# Initialize OpenAI client
client = openai.OpenAI(api_key="your-api-key")

def enhanced_conversation(user_message, conversation_history=[]):
    # Analyze message with HELM Core
    analysis = helm.inference(user_message, speaker_id="user")
    
    # Extract emotional and social context
    mood = analysis['mood']
    social_context = analysis['social'] 
    memory_info = analysis['memory']
    
    # Build context-aware prompt for GPT-4.1
    system_prompt = f"""You are having a conversation with someone who appears to be:
    - Emotional state: {mood['valence'][0]:.2f} valence (negative to positive), {mood['arousal'][0]:.2f} arousal (calm to excited)
    - Relationship context: formality level {social_context['formality'][0]:.2f}, intimacy level {social_context['intimacy'][0]:.2f}
    - Message importance: {memory_info['importance'].argmax()} (0=low, 1=medium, 2=high)
    
    Respond appropriately to their emotional state and relationship context."""
    
    # Get GPT-4.1 response
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            *conversation_history,
            {"role": "user", "content": user_message}
        ]
    )
    
    # Analyze the response for memory storage
    response_analysis = helm.inference(response.choices[0].message.content, speaker_id="assistant")
    
    return {
        "response": response.choices[0].message.content,
        "user_analysis": analysis,
        "response_analysis": response_analysis
    }

# Example usage
result = enhanced_conversation("I'm feeling really anxious about my presentation tomorrow")
```

### Expected Output

```python
{
    "response": "I understand you're feeling anxious about your presentation. That's completely normal - presentations can feel overwhelming. Would it help to talk through what specifically is making you most nervous? Sometimes breaking down our worries can make them feel more manageable.",
    
    "user_analysis": {
        "mood": {
            "valence": [-0.6],  # Negative emotional state
            "arousal": [0.7],   # High arousal (anxious energy)
            "persistence": [0.4] # Moderate persistence
        },
        "memory": {
            "importance": [0.1, 0.2, 0.7],  # High importance (index 2)
            "emotional_weight": [0.8],       # Strong emotional content
            "topic": [0.0, 0.0, 0.9, ...],   # Work/stress category
            "decay_rate": [0.3]              # Slower decay for important memory
        },
        "social": {
            "relationship": [0.1, 0.3, 0.6, 0.0, 0.0],  # Friend-level (index 2)
            "formality": [0.3],              # Casual conversation
            "intimacy": [0.6],               # Moderate intimacy
            "group_size": [1.0]              # One-on-one conversation
        },
        "memory_management": {
            "retrieval_score": [0.8],        # High relevance for retrieval
            "storage_probability": [0.9],    # Definitely store this
            "pruning_probability": [0.1]     # Don't prune
        }
    },
    
    "response_analysis": {
        "mood": {
            "valence": [0.3],   # Supportive, slightly positive
            "arousal": [0.2],   # Calm response
            "persistence": [0.6] # Supportive mood should persist
        }
        # ... additional analysis
    }
}
```

## Training Data

HELM Core is designed to work with multiple dataset types:

- **MELD**: Multimodal emotion recognition in conversations
- **GoEmotions**: Fine-grained emotion classification (27 categories)
- **Mood tracking datasets**: Temporal mood patterns and influences
- **Social context datasets**: Relationship and formality classification

## Model Architecture Details

### Conversation Encoder
- Base model: RoBERTa-base (768-dimensional embeddings)
- Additional bidirectional LSTM for conversation context
- Shared across all downstream tasks

### Memory Classifier
- Multi-task architecture with shared backbone
- Outputs: importance (3 classes), emotional weight (continuous), topic (10 categories), decay rate (continuous)

### Mood Tracker  
- LSTM-based temporal modeling
- Outputs: valence (-1 to 1), arousal (0 to 1), persistence (0 to 1)
- Integrates conversation history for context

### Social Context Analyzer
- Multi-output architecture for relationship understanding
- Outputs: relationship type (5 classes), formality (continuous), intimacy (continuous), group size (continuous)

### Memory Manager
- Fusion network combining all model outputs
- Decisions: storage probability, retrieval relevance, pruning probability

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

## Evaluation

The system includes comprehensive evaluation utilities:

```python
from helm_core import ModelEvaluator

evaluator = ModelEvaluator(system)
results = evaluator.run_comprehensive_evaluation({
    'emotion': 'test_emotions.csv',
    'mood': 'test_mood.csv'
})
```

## Advanced Usage

### Contextual Inference with Memory

```python
from helm_core import AdvancedInference

advanced = AdvancedInference(system)

# Maintains conversation context and memory across turns
results = advanced.contextual_inference(
    "I'm worried about tomorrow", 
    speaker_id="user"
)

# Returns enhanced results with:
# - Retrieved relevant memories
# - Context-adjusted predictions  
# - Automatic memory management
```
