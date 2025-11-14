"""
Simple LLM client using transformers (HuggingFace models) or Gemini API
"""
import os
import logging
import random
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger("main." + __name__.split(".")[-1])

# Optional imports for Gemini API
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not available. Gemini API will not work.")

# Optional imports for HuggingFace models
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Optional bitsandbytes quantization
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
        BNB_AVAILABLE = True
    except Exception:
        BNB_AVAILABLE = False
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available. HuggingFace models will not work.")


class LLMClient:
    """LLM client using transformers (HuggingFace models) or Gemini API"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self.use_gemini = False
        self.gemini_model = None
        self.gemini_api_keys = []
        self.current_api_key_index = 0
        
        # Rate limiting for Gemini API
        self.last_api_call_time = 0
        self.min_delay_between_calls = float(os.getenv('GEMINI_MIN_DELAY', '1.0'))  # seconds between calls
        self.max_retries = int(os.getenv('GEMINI_MAX_RETRIES', '3'))
        self.retry_delay = float(os.getenv('GEMINI_RETRY_DELAY', '5.0'))  # initial retry delay
        
        # Fallback to HuggingFace if Gemini fails persistently
        self.gemini_failure_count = 0
        self.gemini_failure_threshold = int(os.getenv('GEMINI_FAILURE_THRESHOLD', '5'))
        self.fallback_model_path = os.getenv('FALLBACK_MODEL_PATH', 'Qwen/Qwen2.5-7B-Instruct')  # e.g., "Qwen/Qwen2.5-7B-Instruct"
        self.has_fallen_back = False
        
        # Check if model_path is a Gemini model
        if model_path and model_path.startswith("gemini/"):
            self.use_gemini = True
            self._init_gemini()
        elif model_path and TRANSFORMERS_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._load_model()
    
    def _init_gemini(self):
        """Initialize Gemini API client"""
        if not GEMINI_AVAILABLE:
            logger.error("google-generativeai not available. Install with: pip install google-generativeai")
            return
        
        # Get API keys from environment
        api_token = os.getenv('LLM_API_TOKEN', '')
        if api_token:
            self.gemini_api_keys = [key.strip() for key in api_token.split(',') if key.strip()]
        
        if not self.gemini_api_keys:
            logger.error("No Gemini API keys found in LLM_API_TOKEN environment variable")
            return
        
        # Extract model name (e.g., "gemini/gemini-2.5-pro" -> "gemini-2.5-pro")
        model_name = self.model_path.replace("gemini/", "")
        # Ensure model name has "models/" prefix if not already present
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"
        
        # Try to initialize with first API key
        if self._try_init_gemini(model_name):
            logger.info(f"Initialized Gemini API with model: {model_name}")
        else:
            logger.error(f"Failed to initialize Gemini API with model: {model_name}")
    
    def _try_init_gemini(self, model_name: str) -> bool:
        """Try to initialize Gemini with current API key"""
        if not self.gemini_api_keys:
            return False
        
        # Safety settings - allow all content for medical research
        if not GEMINI_AVAILABLE:
            return False
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Try each API key until one works
        for i, api_key in enumerate(self.gemini_api_keys):
            try:
                genai.configure(api_key=api_key)
                test_model = genai.GenerativeModel(
                    model_name,
                    safety_settings=safety_settings
                )
                # Test if the API key works with a simple call
                # Use a simple test that won't trigger safety filters
                test_response = test_model.generate_content(
                    "Say hello",
                    safety_settings=safety_settings
                )
                if test_response and test_response.text:
                    # This API key works
                    self.gemini_model = test_model
                    self.current_api_key_index = i
                    logger.info(f"Using Gemini API key {i+1}/{len(self.gemini_api_keys)}")
                    return True
                else:
                    logger.warning(f"API key {i+1} returned empty response, trying next...")
                    continue
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini with API key {i+1}: {e}")
                if i < len(self.gemini_api_keys) - 1:
                    logger.info(f"Trying next API key...")
                    continue
                else:
                    logger.error(f"All Gemini API keys failed")
                    return False
        return False
    
    def _load_model(self):
        """Load the LLM model"""
        if not self.model_path:
            logger.warning("No model path provided for LLM client")
            return
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers not available. Cannot load model.")
            return
        
        try:
            # Get HuggingFace token for gated models
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            
            # Quantization preference (env): 4bit / 8bit / none / auto
            # - "4bit": Force 4-bit quantization (requires GPU + bitsandbytes)
            # - "8bit": Force 8-bit quantization (requires GPU + bitsandbytes)
            # - "none": No quantization, load full precision model
            # - "auto" or empty: Auto-select based on available memory
            # Default: "none" (pure model without quantization)
            quant_pref = (os.getenv('LLM_QUANT', 'none') or 'none').strip().lower()
            
            # Only use quantization if explicitly requested AND bitsandbytes is available
            use_4bit = self.device == 'cuda' and BNB_AVAILABLE and quant_pref == '4bit'
            use_8bit = self.device == 'cuda' and BNB_AVAILABLE and quant_pref == '8bit'
            use_auto_quant = self.device == 'cuda' and BNB_AVAILABLE and quant_pref == 'auto'
            
            logger.info(f"Loading HuggingFace model from {self.model_path}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Quantization mode: {quant_pref}")

            quantization_config = None
            from_pretrained_kwargs = {
                'device_map': 'auto' if self.device == 'cuda' else None,
                'token': hf_token,
                'trust_remote_code': True,
            }

            if use_4bit:
                logger.info("✓ Using 4-bit quantization (bitsandbytes, nf4)")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                from_pretrained_kwargs['quantization_config'] = quantization_config
            elif use_8bit:
                logger.info("✓ Using 8-bit quantization (bitsandbytes)")
                from_pretrained_kwargs['load_in_8bit'] = True
            elif use_auto_quant:
                logger.info("✓ Using auto 4-bit quantization (memory-efficient)")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                from_pretrained_kwargs['quantization_config'] = quantization_config
            else:
                # No quantization - pure model
                logger.info("✓ Using full precision model (no quantization)")
                from_pretrained_kwargs['torch_dtype'] = torch.float16 if self.device == 'cuda' else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **from_pretrained_kwargs,
            )
            
            if self.device == 'cpu' and self.model is not None:
                self.model = self.model.to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                token=hf_token,
                trust_remote_code=True,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("HuggingFace model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {e}")
            self.model = None
            self.tokenizer = None
    
    def call(self, messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
        """Make a single LLM call with automatic fallback"""
        # Check if we should use Gemini or have fallen back to HuggingFace
        if self.use_gemini and not self.has_fallen_back:
            result = self._call_gemini(messages, max_tokens)
            
            # If result is empty and we've exceeded failure threshold, try fallback
            if not result and self.gemini_failure_count >= self.gemini_failure_threshold:
                logger.warning(f"Gemini failed {self.gemini_failure_count} times. Attempting fallback to HuggingFace...")
                if self._fallback_to_huggingface():
                    logger.info("Successfully fell back to HuggingFace model")
                    return self._call_huggingface(messages, max_tokens)
            
            # Reset failure count on successful call
            if result:
                self.gemini_failure_count = 0
            
            return result
        else:
            return self._call_huggingface(messages, max_tokens)
    
    def _call_gemini(self, messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
        """Make a call using Gemini API with rate limiting"""
        if not self.gemini_model:
            logger.error("Gemini model not initialized. Cannot make call.")
            return ""
        
        # Rate limiting: ensure minimum delay between API calls
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        if time_since_last_call < self.min_delay_between_calls:
            sleep_time = self.min_delay_between_calls - time_since_last_call
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_api_call_time = time.time()
        
        try:
            # Convert messages to Gemini format
            if isinstance(messages, str):
                prompt = messages
            else:
                # Gemini expects a single prompt or chat history
                # For now, concatenate all user messages
                prompt_parts = []
                for msg in messages:
                    if msg.get("role") == "user":
                        prompt_parts.append(msg.get("content", ""))
                    elif msg.get("role") == "assistant":
                        # Gemini can handle chat history, but for simplicity, we'll just use user messages
                        pass
                prompt = "\n\n".join(prompt_parts) if prompt_parts else messages[0].get("content", "")
            
            # Generate response
            generation_config = {
                "max_output_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
            }
            
            # Also pass safety settings to generate_content to ensure they're applied
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Check response - try to get text directly first
            try:
                if response and hasattr(response, 'text') and response.text:
                    return response.text
            except Exception:
                pass
            
            # Fallback: check candidates
            if response and response.candidates:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason
                
                # Check if blocked by safety
                if finish_reason == 2:  # SAFETY
                    logger.warning(f"Response blocked by safety filter. Finish reason: {finish_reason}")
                    if candidate.safety_ratings:
                        for rating in candidate.safety_ratings:
                            if rating.probability > 1:  # BLOCKED or HIGH
                                logger.warning(f"  Blocked by {rating.category}: {rating.probability}")
                
                if candidate.content and candidate.content.parts:
                    text = candidate.content.parts[0].text
                    if text:
                        return text
                    else:
                        logger.warning(f"Empty text in Gemini response. Finish reason: {finish_reason}")
                else:
                    logger.warning(f"No content parts in Gemini response. Finish reason: {finish_reason}")
            else:
                logger.warning("No candidates in Gemini API response")
            return ""
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in Gemini API call: {error_msg}")
            
            # Increment failure count
            self.gemini_failure_count += 1
            
            # Handle rate limit errors with exponential backoff
            if "quota" in error_msg.lower() or "rate" in error_msg.lower() or "429" in error_msg:
                logger.warning(f"Rate limit or quota exceeded (failure #{self.gemini_failure_count}). Attempting to handle...")
                
                # Try switching to next API key first
                if "API key" in error_msg or "quota" in error_msg.lower() or "permission" in error_msg.lower():
                    logger.info("Trying next API key...")
                    model_name = self.model_path.replace("gemini/", "")
                    if not model_name.startswith("models/"):
                        model_name = f"models/{model_name}"
                    
                    if self._try_next_api_key(model_name):
                        # Retry with new key after a delay
                        time.sleep(self.min_delay_between_calls)
                        try:
                            response = self.gemini_model.generate_content(
                                prompt,
                                generation_config=generation_config,
                                safety_settings=safety_settings
                            )
                            if response and hasattr(response, 'text') and response.text:
                                self.gemini_failure_count = 0  # Reset on success
                                return response.text
                        except Exception as retry_e:
                            logger.error(f"Retry with new API key also failed: {retry_e}")
                            self.gemini_failure_count += 1
                
                # If no more keys or key switch failed, try exponential backoff
                for retry_num in range(self.max_retries):
                    retry_wait = self.retry_delay * (2 ** retry_num)  # Exponential backoff
                    logger.info(f"Retry {retry_num + 1}/{self.max_retries}: waiting {retry_wait:.1f}s before retry...")
                    time.sleep(retry_wait)
                    
                    try:
                        response = self.gemini_model.generate_content(
                            prompt,
                            generation_config=generation_config,
                            safety_settings=safety_settings
                        )
                        if response and hasattr(response, 'text') and response.text:
                            logger.info(f"Retry {retry_num + 1} succeeded")
                            self.gemini_failure_count = 0  # Reset on success
                            return response.text
                    except Exception as retry_e:
                        logger.warning(f"Retry {retry_num + 1} failed: {retry_e}")
                        if retry_num == self.max_retries - 1:
                            logger.error("All retries exhausted")
                            self.gemini_failure_count += 1
            
            # Check if we should trigger fallback
            if self.gemini_failure_count >= self.gemini_failure_threshold:
                logger.warning(f"Gemini failure threshold reached ({self.gemini_failure_count}/{self.gemini_failure_threshold})")
            
            return ""
    
    def _fallback_to_huggingface(self) -> bool:
        """Fallback to HuggingFace model when Gemini persistently fails"""
        if self.has_fallen_back:
            return True  # Already fallen back
        
        if not self.fallback_model_path:
            logger.error("No fallback model path specified. Set FALLBACK_MODEL_PATH environment variable.")
            return False
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers not available. Cannot fallback to HuggingFace.")
            return False
        
        logger.warning(f"Falling back to HuggingFace model: {self.fallback_model_path}")
        
        try:
            # Save original model path and switch
            original_model_path = self.model_path
            self.model_path = self.fallback_model_path
            self.use_gemini = False
            self.has_fallen_back = True
            
            # Initialize HuggingFace model
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._load_model()
            
            if self.model and self.tokenizer:
                logger.info(f"Successfully initialized fallback model: {self.fallback_model_path}")
                logger.info(f"Original Gemini model ({original_model_path}) will not be used further")
                return True
            else:
                logger.error("Failed to load fallback model")
                self.has_fallen_back = False
                self.use_gemini = True
                self.model_path = original_model_path
                return False
                
        except Exception as e:
            logger.error(f"Error during fallback to HuggingFace: {e}")
            self.has_fallen_back = False
            return False
    
    def _try_next_api_key(self, model_name: str) -> bool:
        """Try to switch to next API key"""
        if not self.gemini_api_keys:
            return False
        
        # Try next key
        next_index = (self.current_api_key_index + 1) % len(self.gemini_api_keys)
        if next_index == self.current_api_key_index:
            return False  # Only one key available
        
        try:
            if not GEMINI_AVAILABLE:
                return False
            # Safety settings - allow all content for medical research
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            api_key = self.gemini_api_keys[next_index]
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(
                model_name,
                safety_settings=safety_settings
            )
            self.current_api_key_index = next_index
            logger.info(f"Switched to Gemini API key {next_index+1}/{len(self.gemini_api_keys)}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch to next API key: {e}")
            return False
    
    def _call_huggingface(self, messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
        """Make a call using HuggingFace model"""
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded. Cannot make call.")
            return ""
        
        try:
            # Format messages as chat template
            if isinstance(messages, str):
                text = messages
            else:
                text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device if hasattr(self.model, 'device') else self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_p=0.9,
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            return response
            
        except Exception as e:
            logger.error(f"Error in LLM call: {e}")
            return ""

