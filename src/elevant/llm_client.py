"""
Simple LLM client using transformers (HuggingFace models)
"""
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("main." + __name__.split(".")[-1])

# Optional imports for HuggingFace models
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
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
        
        if model_path and TRANSFORMERS_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._load_model()
        elif model_path:
            logger.error("transformers not available. Cannot load model.")
    
    def _load_model(self):
        """Load the LLM model"""
        if not self.model_path:
            logger.warning("No model path provided for LLM client")
            return
        
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers not available. Cannot load model.")
            return
        
        try:
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            
            # Quantization preference: 4bit / 8bit / none
            quant_pref = (os.getenv('LLM_QUANT', '4bit') or '4bit').strip().lower()
            
            use_4bit = self.device == 'cuda' and BNB_AVAILABLE and quant_pref == '4bit'
            use_8bit = self.device == 'cuda' and BNB_AVAILABLE and quant_pref == '8bit'
            
            logger.info(f"Loading HuggingFace model: {self.model_path}")
            logger.info(f"Device: {self.device}, Quantization: {quant_pref}")

            load_kwargs = {
                'token': hf_token,
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
            }

            if use_4bit:
                logger.info("Using 4-bit quantization")
                load_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                load_kwargs['device_map'] = 'auto'
            elif use_8bit:
                logger.info("Using 8-bit quantization")
                load_kwargs['load_in_8bit'] = True
                load_kwargs['device_map'] = 'auto'
            else:
                logger.info("Using full precision model")
                load_kwargs['torch_dtype'] = torch.float16 if self.device == 'cuda' else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **load_kwargs,
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
            
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.tokenizer = None
    
    def call(self, messages: List[Dict[str, str]], max_tokens: int = 100000) -> str:
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
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            
            # Get device from model
            device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else self.device
            inputs = self.tokenizer(text, return_tensors="pt").to(device)
            input_length = inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    top_p=0.9,
                )
            
            response = self.tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True
            )
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error in LLM call: {e}")
            return ""
    
    def call_batch(self, messages_batch: List[List[Dict[str, str]]], max_tokens: int = 100000, batch_size: int = 8) -> List[str]:
        """Make batch LLM calls"""
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded. Cannot make batch calls.")
            return [""] * len(messages_batch) if messages_batch else []
        
        if not messages_batch:
            return []
        
        batch_size = max(1, batch_size)
        responses: List[str] = []
        
        try:
            device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else self.device
            
            for start in range(0, len(messages_batch), batch_size):
                chunk = messages_batch[start:start + batch_size]
                
                # Format chunk messages as chat templates
                texts = []
                for messages in chunk:
                    if isinstance(messages, str):
                        text = messages
                    else:
                        text = self.tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    texts.append(text)
                
                # Tokenize chunk with padding
                inputs = self.tokenizer(
                    texts, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ).to(device)
                
                input_lengths = inputs['attention_mask'].sum(dim=1).tolist()
                
                # Generate for chunk
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        top_p=0.9,
                    )
                
                # Decode each output
                for i, output in enumerate(outputs):
                    input_len = int(input_lengths[i]) if i < len(input_lengths) else inputs['input_ids'].shape[1]
                    response = self.tokenizer.decode(
                        output[input_len:], 
                        skip_special_tokens=True
                    )
                    responses.append(response.strip())
            
            return responses
            
        except Exception as e:
            logger.error(f"Error in batch LLM call: {e}")
            return [""] * len(messages_batch)
