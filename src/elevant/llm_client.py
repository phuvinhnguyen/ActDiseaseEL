"""
Simple LLM client using transformers (HuggingFace models)
"""
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("main." + __name__.split(".")[-1])

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
    """LLM client using transformers (HuggingFace models)"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        
        if TRANSFORMERS_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_path and TRANSFORMERS_AVAILABLE:
            self._load_model()
    
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
            # Quantization preference (env): 4bit / 8bit / none
            quant_pref = (os.getenv('LLM_QUANT') or '').strip().lower()  # e.g., "4bit", "8bit"
            use_4bit = self.device == 'cuda' and BNB_AVAILABLE and (quant_pref == '4bit' or (quant_pref == '' and True))
            use_8bit = self.device == 'cuda' and BNB_AVAILABLE and (quant_pref == '8bit')
            
            logger.info(f"Loading HuggingFace model from {self.model_path}")

            quantization_config = None
            from_pretrained_kwargs = {
                'device_map': 'auto' if self.device == 'cuda' else None,
                'token': hf_token,
                'trust_remote_code': True,
            }

            if use_4bit:
                logger.info("Using 4-bit quantization (bitsandbytes, nf4)")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                from_pretrained_kwargs['quantization_config'] = quantization_config
            elif use_8bit:
                logger.info("Using 8-bit quantization (bitsandbytes)")
                # 8-bit path uses load_in_8bit flag
                from_pretrained_kwargs['load_in_8bit'] = True
            else:
                # No quantization
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
        """Make a single LLM call"""
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

