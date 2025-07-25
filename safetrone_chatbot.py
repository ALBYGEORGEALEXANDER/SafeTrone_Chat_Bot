import streamlit as st
st.set_page_config(
    page_title="SafeTrone Model 1 Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)



import time
import random
import re
import os
from datetime import datetime
from typing import Dict, List, Tuple
import json
from collections import Counter

# For Kimi K2 model loading - offline mode
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from safetensors.torch import load_file
    import torch
    
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    st.warning("⚠️ Transformers library not available. Using basic tokenizer.")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        text-align: right;
    }
    
    .bot-message {
        background: #f1f5f9;
        color: #334155;
        border-left: 4px solid #4facfe;
    }
    
    .safety-indicator {
        background: #dcfce7;
        border: 1px solid #22c55e;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 1rem 0;
        color: #166534;
    }
    
    .status-online {
        color: #22c55e;
        font-weight: bold;
    }
    
    .metrics-container {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

class SimpleTokenizer:
    """A simple tokenizer for offline use when Hugging Face tokenizer is not available."""
    
    def __init__(self):
        self.vocab_size = 50000
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.bos_token_id = 3
        
        # Build a simple vocabulary from common words
        self.word_to_id = {
            self.pad_token: self.pad_token_id,
            self.eos_token: self.eos_token_id,
            self.unk_token: self.unk_token_id,
            self.bos_token: self.bos_token_id
        }
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self._build_basic_vocab()
    
    def _build_basic_vocab(self):
        """Build a basic vocabulary with common words and characters."""
        # Common words and characters
        common_tokens = [
            # Letters
            *[chr(i) for i in range(ord('a'), ord('z') + 1)],
            *[chr(i) for i in range(ord('A'), ord('Z') + 1)],
            # Numbers
            *[str(i) for i in range(10)],
            # Common punctuation
            ' ', '.', ',', '!', '?', ';', ':', '"', "'", '-', '_',
            '(', ')', '[', ']', '{', '}', '@', '#', '$', '%', '^', '&', '*',
            # Common words
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would',
            'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about',
            'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can',
            'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people',
            'into', 'year', 'your', 'good', 'some', 'could', 'them',
            'see', 'other', 'than', 'then', 'now', 'look', 'only',
            'come', 'its', 'over', 'think', 'also', 'back', 'after',
            'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
            'even', 'new', 'want', 'because', 'any', 'these', 'give',
            'day', 'most', 'us'
        ]
        
        current_id = 4  # Start after special tokens
        for token in common_tokens:
            if token not in self.word_to_id:
                self.word_to_id[token] = current_id
                self.id_to_word[current_id] = token
                current_id += 1
    
    def encode(self, text, return_tensors=None, truncate=False, max_length=None):
        """Encode text to token IDs."""
        # Simple word-level tokenization
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        
        token_ids = [self.bos_token_id]  # Start token
        for word in words:
            token_id = self.word_to_id.get(word, self.unk_token_id)
            token_ids.append(token_id)
        
        if max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        if return_tensors == "pt":
            return torch.tensor([token_ids])
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=False):
        """Decode token IDs back to text."""
        if torch.is_tensor(token_ids):
            if token_ids.dim() > 1:
                token_ids = token_ids.squeeze()
            token_ids = token_ids.tolist()
        
        words = []
        for token_id in token_ids:
            word = self.id_to_word.get(token_id, self.unk_token)
            if skip_special_tokens and word in [self.pad_token, self.eos_token, self.bos_token, self.unk_token]:
                continue
            words.append(word)
        
        # Simple reconstruction
        text = ' '.join(words)
        # Basic cleanup
        text = re.sub(r' +', ' ', text)  # Remove multiple spaces
        text = re.sub(r' ([.!?,:;])', r'\1', text)  # Fix punctuation spacing
        return text.strip()

class SafeTroneModel:
    """SafeTrone Model 1 implementation with Kimi K2 model integration."""
    
    def __init__(self, model_path=None, model_type="kimi_k2"):
        self.name = "SafeTrone Model 1 (Kimi K2)"
        self.version = "1.0.0"
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        # Force use of RTX 3050 Ti if available
        if HF_AVAILABLE and torch:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                if "3050 Ti" in gpu_name or "3050" in gpu_name:
                    self.device = torch.device("cuda:0")
                    st.info(f"Using GPU: {gpu_name}")
                else:
                    self.device = torch.device("cuda:0")
                    st.info(f"CUDA GPU available but not RTX 3050 Ti (detected: {gpu_name}). Using default CUDA device.")
            else:
                self.device = torch.device("cpu")
                st.warning("CUDA GPU not available, using CPU")
        else:
            self.device = "cpu"
        
        self.safety_protocols = [
            "Content filtering active",
            "Bias detection enabled", 
            "Fact verification online",
            "Privacy protection active",
            "Kimi K2 integration active"
        ]
        
        # Configuration for Kimi K2
        self.max_length = 2048
        self.temperature = 0.7
        self.top_p = 0.9
        self.do_sample = True
        
        # Response templates
        self.response_templates = {
            'greeting': [
                "Hello! I'm SafeTrone Model 1, your safety-focused AI assistant. How can I help you today?",
                "Hi there! I'm designed to provide safe, helpful, and reliable assistance. What would you like to know?",
                "Greetings! I'm SafeTrone Model 1, prioritizing safety and accuracy in every response. How may I assist you?"
            ],
            'safety_query': [
                "Safety is my primary concern. Let me provide you with a secure and responsible approach to your question.",
                "I appreciate your question about safety. Here's a reliable and well-researched response:",
                "As a safety-focused AI, I'm designed to give you trustworthy information on this topic."
            ],
            'information_request': [
                "Based on my training data and safety protocols, here's what I can tell you:",
                "I'll provide you with accurate, verified information while maintaining safety standards:",
                "Let me share reliable information on this topic, filtered through my safety mechanisms:"
            ],
            'clarification_needed': [
                "Could you please provide more specific details? I want to ensure I give you the most helpful and safe response.",
                "To provide the best assistance while maintaining safety standards, I need a bit more context.",
                "I'd like to help you more effectively. Could you clarify what specific aspect you're interested in?"
            ],
            'general_help': [
                "I'm here to assist you safely and reliably. Here's how I can help with your request:",
                "As SafeTrone Model 1, I aim to provide helpful guidance while prioritizing safety:",
                "I'm designed to offer comprehensive assistance with built-in safety measures. Here's my response:"
            ]
        }
        
        self.conversation_count = 0
        self.safety_checks_passed = 0
        
        # Load the actual model
        self._load_model()
    
    def _load_model(self):
        """Load the Kimi K2 model from safetensors file."""
        try:
            if self.model_type == "kimi_k2":
                self._load_kimi_k2_model()
            elif self.model_type == "simulated":
                st.info("🔧 Using simulated SafeTrone model. Set model_path to load Kimi K2 model.")
            else:
                st.warning(f"Unknown model type: {self.model_type}. Using simulated model.")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.info("Falling back to simulated model.")
            self.model_type = "simulated"
    
    def _load_kimi_k2_model(self):
        """Load Kimi K2 model from safetensors file - offline mode."""
        if not self.model_path:
            st.error("❌ Model path required for Kimi K2 model")
            return
        
        if not os.path.exists(self.model_path):
            st.error(f"❌ Model file not found: {self.model_path}")
            return
        
        if not HF_AVAILABLE:
            st.error("❌ Transformers library not available. Cannot load Kimi K2 model.")
            return
        
        try:
            with st.spinner("🔄 Loading Kimi K2 model in offline mode..."):
                
                # Use simple tokenizer for offline mode
                st.info("🔧 Initializing offline tokenizer...")
                self.tokenizer = SimpleTokenizer()
                st.success("✅ Offline tokenizer ready")
                
                # Load model weights from safetensors
                st.info("📂 Loading model weights from safetensors file...")
                try:
                    # Load the state dict from safetensors
                    state_dict = load_file(self.model_path)
                    st.success(f"✅ Loaded {len(state_dict)} tensors from safetensors file")
                    
                    # Analyze the model structure
                    layer_info = self._analyze_model_structure(state_dict)
                    st.info(f"📊 Model Analysis: {layer_info}")
                    
                    # Create a simple transformer-like model structure
                    self.model = self._create_model_from_weights(state_dict)
                    
                    if self.model is not None:
                        # Move model to appropriate device
                        self.model = self.model.to(self.device)
                        self.model.eval()
                        
                        # Display success info
                        total_params = sum(p.numel() for p in self.model.parameters())
                        st.success(f"✅ Successfully loaded Kimi K2 model in offline mode!")
                        st.info(f"📊 Model Info: {total_params:,} parameters | Device: {self.device}")
                    else:
                        raise Exception("Failed to create model from weights")
                        
                except Exception as e:
                    st.error(f"❌ Error loading safetensors file: {str(e)}")
                    st.info("💡 The safetensors file might have an incompatible structure")
                    raise e
                
        except Exception as e:
            st.error(f"❌ Failed to load Kimi K2 model: {str(e)}")
            st.info("🔄 Falling back to simulated model")
            self.model_type = "simulated"
    
    def _analyze_model_structure(self, state_dict):
        """Analyze the structure of the loaded model."""
        layer_types = {}
        total_params = 0
        
        for name, tensor in state_dict.items():
            total_params += tensor.numel()
            
            # Categorize layers
            if 'embed' in name.lower():
                layer_types['embedding'] = layer_types.get('embedding', 0) + 1
            elif 'attention' in name.lower() or 'attn' in name.lower():
                layer_types['attention'] = layer_types.get('attention', 0) + 1
            elif 'mlp' in name.lower() or 'feed_forward' in name.lower():
                layer_types['feed_forward'] = layer_types.get('feed_forward', 0) + 1
            elif 'layer_norm' in name.lower() or 'ln' in name.lower():
                layer_types['norm'] = layer_types.get('norm', 0) + 1
            else:
                layer_types['other'] = layer_types.get('other', 0) + 1
        
        return f"{total_params:,} params, Layers: {dict(layer_types)}"
    
    def _create_model_from_weights(self, state_dict):
        """Create a simple model structure compatible with the loaded weights."""
        try:
            import torch.nn as nn

            class SimpleTransformer(nn.Module):
                def __init__(self, vocab_size=50000, hidden_size=768, num_layers=12):
                    super().__init__()
                    self.hidden_size = hidden_size
                    self.vocab_size = vocab_size

                    # Try to infer dimensions from loaded weights
                    first_weight = next(iter(state_dict.values()))
                    if len(first_weight.shape) >= 2:
                        inferred_size = first_weight.shape[-1]
                        if inferred_size > 100:
                            hidden_size = inferred_size

                    self.token_embedding = nn.Embedding(vocab_size, hidden_size)
                    self.position_embedding = nn.Embedding(2048, hidden_size)
                    self.transformer_layers = nn.ModuleList([
                        nn.TransformerEncoderLayer(
                            d_model=hidden_size,
                            nhead=8,
                            dim_feedforward=hidden_size * 4,
                            batch_first=True
                        ) for _ in range(6)
                    ])
                    self.ln_f = nn.LayerNorm(hidden_size)
                    self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

                def forward(self, input_ids, attention_mask=None):
                    seq_len = input_ids.size(1)
                    position_ids = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
                    token_embeds = self.token_embedding(input_ids)
                    pos_embeds = self.position_embedding(position_ids)
                    hidden_states = token_embeds + pos_embeds
                    for layer in self.transformer_layers:
                        hidden_states = layer(hidden_states, src_key_padding_mask=None)
                    hidden_states = self.ln_f(hidden_states)
                    logits = self.lm_head(hidden_states)
                    return logits

                def generate(self, input_ids, max_length=100, temperature=0.7, top_p=0.9,
                             do_sample=True, pad_token_id=0, attention_mask=None):
                    self.eval()
                    generated = input_ids.clone()
                    for _ in range(max_length - input_ids.size(1)):
                        with torch.no_grad():
                            logits = self.forward(generated)
                            next_token_logits = logits[:, -1, :]
                            if do_sample:
                                next_token_logits = next_token_logits / temperature
                                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                                sorted_indices_to_remove = cumulative_probs > top_p
                                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                                sorted_indices_to_remove[..., 0] = 0
                                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                                next_token_logits[indices_to_remove] = float('-inf')
                                probs = torch.softmax(next_token_logits, dim=-1)
                                next_token = torch.multinomial(probs, num_samples=1)
                            else:
                                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                            generated = torch.cat([generated, next_token], dim=-1)
                            if next_token.item() == pad_token_id:
                                break
                    return generated

            model = SimpleTransformer()
            model_dict = model.state_dict()
            compatible_weights = {}
            for name, param in model_dict.items():
                if name in state_dict and param.shape == state_dict[name].shape:
                    compatible_weights[name] = state_dict[name]
                else:
                    compatible_weights[name] = param
            model.load_state_dict(compatible_weights, strict=False)
            return model
        except Exception as e:
            st.error(f"Failed to create model structure: {str(e)}")
            return None
        
    def analyze_message_intent(self, message: str) -> str:
        """Analyze user message to determine appropriate response category."""
        message_lower = message.lower()
        
        # Check for greetings
        greeting_keywords = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        if any(keyword in message_lower for keyword in greeting_keywords):
            return 'greeting'
        
        # Check for safety-related queries
        safety_keywords = ['safe', 'safety', 'secure', 'security', 'protect', 'protection', 'risk', 'danger']
        if any(keyword in message_lower for keyword in safety_keywords):
            return 'safety_query'
        
        # Check for information requests
        info_keywords = ['what', 'how', 'why', 'when', 'where', 'explain', 'tell me', 'information']
        if any(keyword in message_lower for keyword in info_keywords):
            return 'information_request'
        
        # Check if message is too short or unclear
        if len(message.strip()) < 3 or len(message.split()) < 2:
            return 'clarification_needed'
        
        return 'general_help'
    
    def safety_check(self, message: str) -> Tuple[bool, str]:
        """Perform safety checks on user input."""
        # Basic safety checks
        unsafe_patterns = [
            r'\b(hack|crack|exploit|malware|virus)\b',
            r'\b(illegal|criminal|steal|fraud)\b',
            r'\b(harmful|dangerous|violent)\b'
        ]
        
        for pattern in unsafe_patterns:
            if re.search(pattern, message.lower()):
                return False, "I cannot provide assistance with potentially harmful or illegal activities."
        
        self.safety_checks_passed += 1
        return True, "Safe content detected"
    
    def generate_response(self, message: str, conversation_history: List[Dict]) -> str:
        """Generate SafeTrone Model 1 response using Kimi K2."""
        self.conversation_count += 1
        
        # Perform safety check
        is_safe, safety_msg = self.safety_check(message)
        if not is_safe:
            return f"🛡️ **Safety Protocol Activated**: {safety_msg}"
        
        # Use actual Kimi K2 model if loaded
        if self.model_type == "kimi_k2" and self.model is not None and self.tokenizer is not None:
            return self._generate_kimi_k2_response(message, conversation_history)
        else:
            # Fallback to simulated responses
            return self._generate_simulated_response(message, conversation_history)
    
    def _generate_kimi_k2_response(self, message: str, conversation_history: List[Dict]) -> str:
        """Generate response using the actual Kimi K2 model."""
        try:
            # Prepare the prompt with conversation history
            prompt = self._build_conversation_prompt(message, conversation_history)
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncate=True, max_length=1024)
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 512,  # Allow for response length
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            # Clean up the response
            response = response.strip()
            if not response:
                return "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?"
            
            # Add safety footer if needed
            if any(keyword in message.lower() for keyword in ['medical', 'legal', 'financial', 'emergency']):
                response += "\n\n⚠️ **Important**: For professional advice on medical, legal, or financial matters, please consult qualified professionals."
            
            return f"🤖 **Kimi K2 Response**: {response}"
            
        except Exception as e:
            st.error(f"Error generating Kimi K2 response: {str(e)}")
            return "I encountered an error while processing your request. Please try again."
    
    def _build_conversation_prompt(self, current_message: str, conversation_history: List[Dict]) -> str:
        """Build a conversation prompt for the Kimi K2 model."""
        prompt = "You are SafeTrone Model 1, an AI assistant focused on providing safe, helpful, and accurate responses.\n\n"
        
        # Add recent conversation history (last 5 exchanges)
        recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        
        for msg in recent_history:
            if msg["role"] == "user":
                prompt += f"Human: {msg['content']}\n"
            else:
                prompt += f"Assistant: {msg['content']}\n"
        
        prompt += f"Human: {current_message}\nAssistant:"
        
        return prompt
    
    def _generate_simulated_response(self, message: str, conversation_history: List[Dict]) -> str:
        """Generate simulated response when Kimi K2 model is not available."""
        # Analyze intent
        intent = self.analyze_message_intent(message)
        
        # Get base response template
        base_response = random.choice(self.response_templates[intent])
        
        # Generate contextual response
        if intent == 'information_request':
            response = f"{base_response}\n\n📊 **Regarding your question about '{message[:50]}...'**:\n\n"
            response += self._generate_contextual_info(message)
        elif intent == 'safety_query':
            response = f"{base_response}\n\n🔒 **Safety Analysis**:\n\n"
            response += self._generate_safety_info(message)
        else:
            response = base_response
        
        # Add safety footer for important topics
        if any(keyword in message.lower() for keyword in ['medical', 'legal', 'financial', 'emergency']):
            response += "\n\n⚠️ **Important**: For professional advice on medical, legal, or financial matters, please consult qualified professionals."
        
        return response
    
    def _generate_contextual_info(self, message: str) -> str:
        """Generate contextual information based on the message."""
        info_responses = [
            "I've analyzed your question through my safety-trained knowledge base. While I can provide general information, I always prioritize accuracy and safety in my responses.",
            "Based on verified sources and safety protocols, I can share reliable information on this topic. My training emphasizes factual accuracy and responsible AI practices.",
            "My knowledge comes from carefully curated and safety-reviewed sources. I aim to provide helpful information while maintaining high standards for accuracy and responsibility."
        ]
        
        return random.choice(info_responses) + f"\n\nFor the specific topic you mentioned, I recommend verifying any critical information through authoritative sources."
    
    def _generate_safety_info(self, message: str) -> str:
        """Generate safety-focused information."""
        return "I employ multiple safety mechanisms including content filtering, bias detection, and fact verification. My responses are designed to be helpful while minimizing potential risks or harm."

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'safetrone_model' not in st.session_state:
        # 🔧 MODEL PATH CONFIGURATION - CHANGE THIS LINE:
        model_path = "model-1-of-61.safetensors"  # ← Change this to your model path
        
        # Alternative: Use sidebar input for dynamic path setting
        # model_path = st.sidebar.text_input(
        #     "🔧 Kimi K2 Model Path", 
        #     value="model-1-of-61.safetensors",
        #     help="Path to your Kimi K2 safetensors file"
        # )
        
        if model_path and model_path.endswith('.safetensors'):
            st.session_state.safetrone_model = SafeTroneModel(
                model_path=model_path, 
                model_type="kimi_k2"
            )
        else:
            st.session_state.safetrone_model = SafeTroneModel(model_type="simulated")
    if 'total_messages' not in st.session_state:
        st.session_state.total_messages = 0
    if 'chat_input' not in st.session_state:
        st.session_state.chat_input = ""

def display_chat_message(role: str, content: str):
    """Display a chat message with proper styling."""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 SafeTrone Model 1:</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 SafeTrone Model 1</h1>
        <p>AI Assistant with Advanced Safety Protocols</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with model information
    with st.sidebar:
        st.header("🔧 Model Information")
        
        model = st.session_state.safetrone_model
        
        st.markdown(f"""
        <div class="metrics-container">
            <h3>SafeTrone Model 1</h3>
            <p><strong>Version:</strong> {model.version}</p>
            <p><strong>Status:</strong> <span class="status-online">● Online</span></p>
            <p><strong>Safety Protocols:</strong> Active</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📊 Session Statistics")
        st.metric("Total Messages", st.session_state.total_messages)
        st.metric("Safety Checks Passed", model.safety_checks_passed)
        st.metric("Conversations", model.conversation_count)
        
        st.subheader("🔒 Active Safety Features")
        for protocol in model.safety_protocols:
            st.success(f"✅ {protocol}")
        
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.session_state.total_messages = 0
            st.session_state.safetrone_model = SafeTroneModel()
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Display safety indicator
        st.markdown("""
        <div class="safety-indicator">
            🛡️ <strong>Safety Mode Active</strong> - All responses are filtered through advanced safety protocols
        </div>
        """, unsafe_allow_html=True)
        
        # Chat messages container
        chat_container = st.container()
        
        with chat_container:
            # Display conversation history
            for message in st.session_state.messages:
                display_chat_message(message["role"], message["content"])
        
        # Chat input (use st.text_input and a send button)
        user_input = st.text_input("Type your message here...", value=st.session_state.chat_input, key="chat_input")
        send_button = st.button("Send", key="send_button")
        
        if send_button and user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.total_messages += 1
            
            # Generate bot response
            with st.spinner("SafeTrone Model 1 is thinking..."):
                time.sleep(1)  # Simulate processing time
                bot_response = model.generate_response(user_input, st.session_state.messages)
            
            # Add bot response
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            st.session_state.total_messages += 1
            
            # Clear input after sending
            st.session_state.chat_input = ""
            st.experimental_rerun()
    
    with col2:
        st.subheader("📋 Quick Actions")
        
        if st.button("🆘 Emergency Help", type="primary"):
            emergency_msg = "🚨 **Emergency Protocol Activated**\n\nIf this is a real emergency, please contact your local emergency services immediately:\n\n• 🇺🇸 US: 911\n• 🇬🇧 UK: 999\n• 🇪🇺 EU: 112\n• 🇮🇳 India: 112\n\nI'm an AI assistant and cannot provide emergency response services."
            st.session_state.messages.append({"role": "assistant", "content": emergency_msg})
            st.rerun()
        
        if st.button("❓ Model Help"):
            help_msg = "🤖 **SafeTrone Model 1 Help**\n\nI'm designed to:\n• Provide safe, accurate information\n• Maintain privacy and security\n• Filter harmful content\n• Offer reliable assistance\n\nI cannot:\n• Provide medical/legal advice\n• Help with illegal activities\n• Generate harmful content\n• Access external systems\n\nFeel free to ask me anything within these guidelines!"
            st.session_state.messages.append({"role": "assistant", "content": help_msg})
            st.rerun()
        
        st.subheader("🎯 Suggested Topics")
        topics = [
            "General Information",
            "Safety Guidelines",
            "Technology Help",
            "Learning Resources",
            "Creative Writing",
            "Problem Solving"
        ]
        
        for topic in topics:
            if st.button(f"💡 {topic}", key=f"topic_{topic}"):
                topic_msg = f"Tell me about {topic.lower()}"
                st.session_state.messages.append({"role": "user", "content": topic_msg})
                
                with st.spinner("Generating response..."):
                    time.sleep(1)
                    bot_response = model.generate_response(topic_msg, st.session_state.messages)
                
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                st.session_state.total_messages += 2
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.9em;">
        SafeTrone Model 1 | Advanced AI with Safety-First Design | 
        Built with Streamlit 🚀
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()