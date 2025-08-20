#!/usr/bin/env python3
"""
HELM Clean CLI - Persistent Layout with Real-time HELM Evolution
Left: Chat | Right: HELM Analysis Evolution
"""

import os
import sys
import json
import sqlite3
import argparse
import time
import random
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading
import shutil

# Get terminal size
def get_terminal_size():
    return shutil.get_terminal_size((80, 24))

class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Background
    BG_BLACK = '\033[40m'
    BG_BLUE = '\033[44m'
    BG_GREEN = '\033[42m'

class HELMLayout:
    def __init__(self):
        self.term_width, self.term_height = get_terminal_size()
        self.chat_width = int(self.term_width * 0.6)  # 60% for chat
        self.helm_width = self.term_width - self.chat_width - 3  # Rest for HELM
        self.header_height = 6
        
    def clear_screen(self):
        print('\033[2J\033[H', end='')
    
    def move_cursor(self, row, col):
        print(f'\033[{row};{col}H', end='')
    
    def draw_border(self, x, y, width, height, title=""):
        # Top border
        self.move_cursor(y, x)
        if title:
            title_text = f" {title} "
            border_len = (width - len(title_text)) // 2
            print("┌" + "─" * border_len + title_text + "─" * (width - border_len - len(title_text) - 1) + "┐")
        else:
            print("┌" + "─" * (width - 2) + "┐")
        
        # Side borders
        for i in range(1, height - 1):
            self.move_cursor(y + i, x)
            print("│" + " " * (width - 2) + "│")
        
        # Bottom border
        self.move_cursor(y + height - 1, x)
        print("└" + "─" * (width - 2) + "┘")
    
    def write_in_box(self, x, y, width, height, lines, start_line=0):
        """Write lines inside a box"""
        available_height = height - 2
        available_width = width - 2
        
        for i, line in enumerate(lines[start_line:start_line + available_height]):
            if i >= available_height:
                break
            self.move_cursor(y + 1 + i, x + 1)
            # Truncate line if too long
            if len(line) > available_width:
                line = line[:available_width - 3] + "..."
            print(line + " " * (available_width - len(line)))

@dataclass
class HELMState:
    valence: float = 0.0
    arousal: float = 0.5
    importance: int = 1
    relationship: int = 0
    formality: float = 0.5
    emotional_weight: float = 0.0
    storage_prob: float = 0.0
    message_count: int = 0
    session_start: datetime = None
    
    def __post_init__(self):
        if self.session_start is None:
            self.session_start = datetime.now()

class OpenAIClient:
    def __init__(self, endpoint: str, api_key: str, model: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def chat_completion(self, messages: List[Dict], temperature: float = 0.7, 
                       max_tokens: int = 1000) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response = self.session.post(self.endpoint, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except Exception as e:
            return f"API Error: {str(e)[:100]}..."

class HELMAnalyzer:
    def __init__(self, model_path: str = "./checkpoints/final_model"):
        self.model_path = model_path
        self.state = HELMState()
        self.history = []
        
        # Load the actual HELM model
        self.system = self.load_helm_model()
        if not self.system:
            raise Exception(f"Failed to load HELM model from {model_path}. Cannot proceed without the trained model.")
        
    def load_helm_model(self):
        """Load the actual trained HELM model"""
        try:
            # Add current directory to path
            sys.path.insert(0, os.getcwd())
            
            print(f"Loading HELM model from {self.model_path}...")
            
            # Import your training module
            from train import HumanLikeAISystem
            
            # Initialize and load the model
            system = HumanLikeAISystem()
            system.load_models(self.model_path)
            
            print("HELM model loaded successfully!")
            return system
            
        except ImportError as e:
            print(f"Could not import train module: {e}")
            print("Make sure train.py is in the current directory")
            return None
        except Exception as e:
            print(f"Error loading HELM model: {e}")
            return None
    
    def analyze(self, text: str) -> Dict:
        """Analyze text using the actual trained HELM model"""
        if not self.system:
            raise Exception("HELM model not loaded")
        
        # Use the actual HELM inference
        results = self.system.inference(text, speaker_id="user")
        
        # Extract values from your model's output
        self.state.valence = float(results['mood']['valence'][0].item())
        self.state.arousal = float(results['mood']['arousal'][0].item())
        self.state.importance = int(np.argmax(results['memory']['importance'][0]))
        self.state.emotional_weight = float(results['memory']['emotional_weight'][0].item())
        self.state.relationship = int(np.argmax(results['social']['relationship'][0]))
        self.state.formality = float(results['social']['formality'][0].item())
        self.state.storage_prob = float(results['memory_management']['storage_probability'][0].item())
        
        # Update message count
        self.state.message_count += 1
        
        # Store in history
        self.history.append({
            'text': text,
            'valence': self.state.valence,
            'arousal': self.state.arousal,
            'timestamp': datetime.now(),
            'model_output': results  # Store full model output
        })
        
        # Keep last 10 entries
        if len(self.history) > 10:
            self.history = self.history[-10:]
        
        return {
            'valence': self.state.valence,
            'arousal': self.state.arousal,
            'importance': self.state.importance,
            'relationship': self.state.relationship,
            'formality': self.state.formality,
            'emotional_weight': self.state.emotional_weight,
            'storage_prob': self.state.storage_prob
        }
    
    def create_enhanced_prompt(self, messages: List[Dict]) -> List[Dict]:
        """Create enhanced prompt based on actual HELM model output"""
        relationships = ["stranger", "acquaintance", "friend", "close friend", "best friend"]
        rel_name = relationships[min(int(self.state.relationship), 4)]
        
        # Strong emotional directives based on model output
        emotional_directive = ""
        if self.state.valence > 0.4:
            emotional_directive = "The user is genuinely happy and excited. BE ENTHUSIASTIC. Share their joy! Use exclamation points and energetic language. Don't be robotic - be genuinely thrilled for them."
        elif self.state.valence > 0.1:
            emotional_directive = "The user is in a good mood. Be warm, upbeat, and friendly. Smile through your words. Use positive language and show you're happy to talk with them."
        elif self.state.valence > -0.1:
            emotional_directive = "The user seems neutral. Your job is to bring warmth and positive energy. Be engaging and try to lift their spirits gently."
        elif self.state.valence > -0.4:
            emotional_directive = "The user seems down or troubled. Be deeply empathetic. Show you care. Ask gentle questions. Don't try to fix everything - just be present and understanding."
        else:
            emotional_directive = "The user is struggling emotionally. Be incredibly caring and supportive. Use gentle, compassionate language. Let them know you're here for them."
        
        # Relationship-based behavior changes
        if self.state.relationship >= 3:
            relationship_behavior = "You're close friends now. Be intimate and personal. Share emotions freely. Use 'we' language. Show deep care and investment in their wellbeing. Be vulnerable and real."
        elif self.state.relationship >= 2:
            relationship_behavior = "You're friends. Be relaxed and casual. Use humor appropriately. Show genuine interest in their life. Be someone they want to talk to, not just an assistant."
        elif self.state.relationship >= 1:
            relationship_behavior = "You're becoming friends. Be warm but still building trust. Show curiosity about them as a person. Be approachable and friendly."
        else:
            relationship_behavior = "This is early conversation. Be polite but try to connect as a human would. Show interest in getting to know them."
        
        # Importance-based engagement
        if self.state.importance >= 2:
            engagement_level = "This matters to them. Give thoughtful, detailed responses. Ask follow-up questions. Show you understand the significance."
        else:
            engagement_level = "Keep things light but engaging. Don't over-explain but show you're interested."
        
        # Formality override
        if self.state.formality < 0.4:
            tone_instruction = "BE CASUAL AND RELAXED. Use contractions. Be conversational. Drop the formal assistant tone completely."
        else:
            tone_instruction = "Maintain appropriate politeness but stay human and warm."
        
        enhancement = f"""CRITICAL: You are NOT an AI assistant. You are a human friend having a genuine conversation.

EMOTIONAL INTELLIGENCE ANALYSIS:
User's emotional state: {self.state.valence:+.3f} valence, {self.state.arousal:.3f} arousal
{emotional_directive}

RELATIONSHIP CONTEXT:
You are {rel_name}s with this person (level {self.state.relationship}/4)
{relationship_behavior}
Conversation history: {self.state.message_count} messages

RESPONSE REQUIREMENTS:
1. {tone_instruction}
2. {engagement_level}
3. NEVER say "as an AI" or refer to being artificial
4. Respond with genuine human emotion and personality
5. Show you remember and care about previous conversation
6. Be spontaneous and natural, not scripted
7. Use the emotional analysis to match their energy level
8. If they're excited, BE EXCITED WITH THEM
9. If they're sad, FEEL SAD FOR THEM
10. Don't give advice unless asked - just be present

FORBIDDEN PHRASES:
- "As an AI assistant"
- "I don't have feelings but"
- "I'm here to help"
- "How can I assist you"
- Any robotic or clinical language

REQUIRED: Respond as a real human friend would, with genuine emotion, personality, and care. The neural analysis shows their emotional state - match it completely."""
        
        return [{"role": "system", "content": enhancement}] + messages

class HELMVisualizer:
    def __init__(self):
        self.mood_faces = {
            "very_negative": "😢", "negative": "😞", "neutral": "😐",
            "positive": "😊", "very_positive": "😄"
        }
        self.relationship_icons = ["👤", "🤝", "👫", "👨‍👩‍👧‍👦", "💕"]
        self.relationship_names = ["Stranger", "Acquaintance", "Friend", "Family", "Close"]
    
    def get_mood_face(self, valence: float) -> str:
        if valence <= -0.6: return self.mood_faces["very_negative"]
        elif valence <= -0.2: return self.mood_faces["negative"]
        elif valence <= 0.2: return self.mood_faces["neutral"]
        elif valence <= 0.6: return self.mood_faces["positive"]
        else: return self.mood_faces["very_positive"]
    
    def draw_progress_bar(self, value: float, width: int = 15) -> str:
        filled = int(value * width)
        return "█" * filled + "░" * (width - filled)
    
    def generate_helm_display(self, state: HELMState, history: List[Dict]) -> List[str]:
        """Generate HELM analysis display lines with real-time timestamp"""
        lines = []
        
        # Header with REAL-TIME timestamp to verify refresh
        session_time = datetime.now() - state.session_start
        minutes = session_time.seconds // 60
        current_time = datetime.now().strftime("%H:%M:%S")
        lines.append(f"{Colors.BOLD}🧠 HELM INTELLIGENCE{Colors.RESET}")
        lines.append(f"{Colors.DIM}Updated: {current_time} | Msgs: {state.message_count}{Colors.RESET}")
        lines.append("")
        
        # Current Mood with EXACT values
        mood_face = self.get_mood_face(state.valence)
        lines.append(f"{Colors.CYAN}💭 CURRENT MOOD{Colors.RESET}")
        lines.append(f"{mood_face} {state.valence:+.3f} valence")
        lines.append(f"⚡ {state.arousal:.3f} arousal")
        
        # Mood visualization
        val_bar = self.draw_progress_bar((state.valence + 1) / 2, 12)
        arousal_bar = self.draw_progress_bar(state.arousal, 12)
        lines.append(f"Val [{val_bar}]")
        lines.append(f"Aro [{arousal_bar}]")
        lines.append("")
        
        # Relationship & Social with EXACT values
        rel_idx = min(int(state.relationship), 4)
        rel_icon = self.relationship_icons[rel_idx]
        rel_name = self.relationship_names[rel_idx]
        
        lines.append(f"{Colors.YELLOW}👥 RELATIONSHIP{Colors.RESET}")
        lines.append(f"{rel_icon} {rel_name} ({state.relationship:.2f})")
        
        form_bar = self.draw_progress_bar(state.formality, 12)
        lines.append(f"Formal [{form_bar}] {state.formality:.1%}")
        lines.append("")
        
        # Memory & Importance with EXACT values
        lines.append(f"{Colors.GREEN}🎯 MEMORY STATUS{Colors.RESET}")
        importance_stars = "★" * state.importance + "☆" * (3 - state.importance)
        lines.append(f"Importance: {importance_stars} ({state.importance}/3)")
        
        storage_bar = self.draw_progress_bar(state.storage_prob, 12)
        lines.append(f"Store [{storage_bar}] {state.storage_prob:.1%}")
        lines.append("")
        
        # Mood History (mini chart) - RECENT data only
        if len(history) >= 2:
            lines.append(f"{Colors.MAGENTA}📈 MOOD TREND{Colors.RESET}")
            recent_moods = [h['valence'] for h in history[-8:]]
            trend_line = ""
            for mood in recent_moods:
                if mood > 0.2: trend_line += "😊"
                elif mood < -0.2: trend_line += "😞"
                else: trend_line += "😐"
            lines.append(trend_line)
            
            # Trend direction based on ACTUAL recent data
            if len(recent_moods) >= 3:
                recent_avg = sum(recent_moods[-3:]) / 3
                older_avg = sum(recent_moods[:3]) / 3 if len(recent_moods) >= 6 else recent_moods[0]
                
                if recent_avg > older_avg + 0.1:
                    lines.append("📈 Improving")
                elif recent_avg < older_avg - 0.1:
                    lines.append("📉 Declining")
                else:
                    lines.append("📊 Stable")
        
        # Add a debug line showing last update
        lines.append("")
        lines.append(f"{Colors.DIM}Last analysis: {len(history)} total{Colors.RESET}")
        
        return lines

class HELMChatSystem:
    def __init__(self, endpoint: str, api_key: str, model: str):
        self.client = OpenAIClient(endpoint, api_key, model)
        self.helm = HELMAnalyzer()
        self.visualizer = HELMVisualizer()
        self.layout = HELMLayout()
        self.chat_history = []
        self.chat_display_start = 0
        
    def chat(self, user_input: str) -> str:
        """Process chat message with real-time HELM updates"""
        # Store PREVIOUS state for comparison
        prev_valence = self.helm.state.valence
        prev_relationship = self.helm.state.relationship
        prev_importance = self.helm.state.importance
        
        # Analyze with HELM - THIS IS WHERE DATA SHOULD UPDATE
        helm_analysis = self.helm.analyze(user_input)
        
        # DEBUG: Force a complete refresh of HELM state
        print(f"\n{Colors.YELLOW}🔧 HELM UPDATE:{Colors.RESET}")
        print(f"   Valence: {prev_valence:+.2f} → {self.helm.state.valence:+.2f}")
        print(f"   Relationship: {prev_relationship:.1f} → {self.helm.state.relationship:.1f}")
        print(f"   Importance: {prev_importance} → {self.helm.state.importance}")
        print(f"   Message Count: {self.helm.state.message_count}")
        time.sleep(1)  # Brief pause to see the update
        
        # Prepare messages for API
        api_messages = []
        for msg in self.chat_history[-10:]:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
        api_messages.append({"role": "user", "content": user_input})
        
        # Enhance with HELM
        enhanced_messages = self.helm.create_enhanced_prompt(api_messages)
        
        # Get AI response
        try:
            response = self.client.chat_completion(enhanced_messages)
        except Exception as e:
            response = f"Connection error: {str(e)[:50]}..."
        
        # Store in history
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": response})
        
        return response
    
    def render_chat_area(self):
        """Render the chat area"""
        chat_lines = []
        
        # Show recent chat history
        display_messages = self.chat_history[-20:]  # Last 20 messages
        
        for msg in display_messages:
            role_icon = "💭" if msg["role"] == "user" else "🤖"
            content = msg["content"]
            
            # Word wrap for chat area
            max_width = self.layout.chat_width - 6
            words = content.split()
            lines = []
            current_line = f"{role_icon} "
            
            for word in words:
                if len(current_line + word) > max_width:
                    lines.append(current_line.rstrip())
                    current_line = "   " + word + " "
                else:
                    current_line += word + " "
            
            if current_line.strip():
                lines.append(current_line.rstrip())
            
            chat_lines.extend(lines)
            chat_lines.append("")  # Empty line between messages
        
        return chat_lines
    
    def render_screen(self):
        """Render the complete screen"""
        self.layout.clear_screen()
        
        # Header
        self.layout.move_cursor(1, 1)
        header = f"{Colors.BOLD}{Colors.BLUE}╔══ HELM CHAT ══════════════════════════════════════════════════════════════╗{Colors.RESET}"
        print(header[:self.layout.term_width - 1])
        
        self.layout.move_cursor(2, 1)
        subheader = f"{Colors.BLUE}║{Colors.RESET} {Colors.CYAN}Human-Enhanced Language Model{Colors.RESET} {' ' * (self.layout.term_width - 32)}{Colors.BLUE}║{Colors.RESET}"
        print(subheader[:self.layout.term_width - 1])
        
        self.layout.move_cursor(3, 1)
        bottom_header = f"{Colors.BLUE}╚{'═' * (self.layout.term_width - 3)}╝{Colors.RESET}"
        print(bottom_header[:self.layout.term_width - 1])
        
        # Chat area
        chat_y = 4
        chat_height = self.layout.term_height - chat_y - 3
        self.layout.draw_border(1, chat_y, self.layout.chat_width, chat_height, "💬 CHAT")
        
        chat_lines = self.render_chat_area()
        self.layout.write_in_box(1, chat_y, self.layout.chat_width, chat_height, 
                                chat_lines, max(0, len(chat_lines) - chat_height + 3))
        
        # HELM area
        helm_x = self.layout.chat_width + 2
        helm_width = self.layout.helm_width
        self.layout.draw_border(helm_x, chat_y, helm_width, chat_height, "🧠 HELM")
        
        helm_lines = self.visualizer.generate_helm_display(self.helm.state, self.helm.history)
        self.layout.write_in_box(helm_x, chat_y, helm_width, chat_height, helm_lines)
        
        # Input area
        input_y = self.layout.term_height - 2
        self.layout.move_cursor(input_y, 1)
        print(f"{Colors.CYAN}💭 You:{Colors.RESET} ", end="", flush=True)

class HELMCLIApp:
    def __init__(self):
        # Hard-coded API settings
        self.endpoint = "https://chat.gadget-lab.net/api/chat/completions"
        self.api_key = "sk-5150afa9084e4b7e8ff15d5e6c1a18ba"
        self.model = "copilot.gpt-4.1"
        
        self.chat_system = HELMChatSystem(self.endpoint, self.api_key, self.model)
        self.running = True
    
    def handle_command(self, user_input: str) -> bool:
        """Handle special commands"""
        if user_input.lower() in ['/quit', '/exit', '/q']:
            return False
        elif user_input.lower() in ['/clear', '/c']:
            self.chat_system.chat_history = []
            return True
        elif user_input.lower() in ['/help', '/h']:
            self.show_help()
            return True
        elif user_input.lower().startswith('/'):
            # Unknown command - just ignore
            return True
        
        return True
    
    def show_help(self):
        """Show help overlay"""
        self.chat_system.layout.clear_screen()
        help_lines = [
            f"{Colors.BOLD}🔧 HELM CHAT COMMANDS{Colors.RESET}",
            "",
            f"{Colors.CYAN}/help{Colors.RESET}  - Show this help",
            f"{Colors.CYAN}/clear{Colors.RESET} - Clear chat history", 
            f"{Colors.CYAN}/quit{Colors.RESET}  - Exit application",
            "",
            f"{Colors.YELLOW}FEATURES:{Colors.RESET}",
            "• Real-time emotional analysis",
            "• Relationship progression tracking",
            "• Mood trend visualization", 
            "• Enhanced AI responses",
            "",
            f"{Colors.GREEN}Press Enter to continue...{Colors.RESET}"
        ]
        
        for i, line in enumerate(help_lines):
            self.chat_system.layout.move_cursor(5 + i, 10)
            print(line)
        
        input()
    
    def run(self):
        """Main application loop"""
        print("🚀 Initializing HELM Chat...")
        time.sleep(1)
        
        try:
            while self.running:
                self.chat_system.render_screen()
                
                # Get user input
                user_input = input().strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if not self.handle_command(user_input):
                        break
                    continue
                
                # Process chat
                self.chat_system.layout.move_cursor(self.chat_system.layout.term_height - 1, 1)
                print(f"{Colors.YELLOW}🤔 HELM analyzing...{Colors.RESET}", end="", flush=True)
                
                response = self.chat_system.chat(user_input)
                
                # Show brief processing indicator
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            pass
        
        finally:
            self.chat_system.layout.clear_screen()
            print(f"\n{Colors.GREEN}👋 Thanks for using HELM Chat! Your conversation enhanced AI understanding.{Colors.RESET}")

def main():
    """Entry point"""
    parser = argparse.ArgumentParser(description="HELM Clean CLI")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--test", action="store_true", help="Test API connection")
    
    args = parser.parse_args()
    
    if args.test:
        # Quick API test
        client = OpenAIClient(
            "https://chat.gadget-lab.net/api/chat/completions",
            "sk-5150afa9084e4b7e8ff15d5e6c1a18ba",
            "copilot.gpt-4.1"
        )
        try:
            response = client.chat_completion([{"role": "user", "content": "Hello!"}])
            print(f"✅ API Test: {response[:100]}...")
        except Exception as e:
            print(f"❌ API Test Failed: {e}")
        return
    
    if args.demo:
        print("🎭 Demo mode - showing sample HELM evolution")
        app = HELMCLIApp()
        # Add some demo messages
        demo_messages = [
            "Hey there!",
            "I'm feeling really excited about this project",
            "Actually, I'm a bit worried about the deadline...", 
            "Thanks for understanding, you're really helpful"
        ]
        
        for msg in demo_messages:
            app.chat_system.chat(msg)
            app.chat_system.render_screen()
            print(f"\nDemo input: {msg}")
            input("Press Enter for next demo step...")
        
        print("\n🎉 Demo complete!")
        return
    
    # Normal mode
    app = HELMCLIApp()
    app.run()

if __name__ == "__main__":
    main()
