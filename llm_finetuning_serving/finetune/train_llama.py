"""
Script finetune Llama-3.1-8B using Unsloth with LoRA
Optimized for Vietnamese Legal QA
"""

import os
import json
import logging
import torch
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from do_spaces_manager import DOSpacesManager

from unsloth import FastLanguageModel

from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
import wandb

import sys
sys.path.append('..')
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
@dataclass
class FineTuneConfig:
    """Fine-tuning configuration optimized for Unsloth and H200 GPU - MAXIMUM UTILIZATION"""

    model_name: str = "unsloth/Llama-3.1-8B-Instruct"  
    max_seq_length: int = 8192
    dtype: Optional[torch.dtype] = None
    load_in_4bit: bool = False

    # LoRA settings
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.0  
    bias: str = "none"  
    use_gradient_checkpointing: str = "unsloth"  
    random_state: int = 3407  
    use_rslora: bool = True  
    
    # Training settings 
    per_device_train_batch_size: int = 32  
    per_device_eval_batch_size: int = 32  
    gradient_accumulation_steps: int = 16   
    warmup_steps: int = 100 
    num_train_epochs: int = 4
    max_steps: int = -1  
    learning_rate: float = 3e-4  
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"  
    seed: int = 3407
    
    # Optimization 
    optim: str = "adamw_torch"  
    fp16: bool = False  
    bf16: bool = True  
    dataloader_num_workers: int = 8
    dataloader_pin_memory: bool = True  
    
    # Logging and saving
    logging_steps: int = 1
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Data settings
    dataset_text_field: str = "text"
    packing: bool = False  
    
    # Output settings
    output_dir: str = "./outputs"
    run_name: str = field(default_factory=lambda: f"vietnamese-legal-llama-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

class LlamaFineTuner:
    def __init__(self, config: FineTuneConfig, data_dir: str):
        self.config = config
        # Chuẩn hóa đường dẫn dữ liệu và output cho phù hợp workspace
        self.data_dir = Path(data_dir).resolve()
        self.model = None
        self.tokenizer = None
        self.trainer = None

        # Setup output directory (outputs/ nằm trong finetune/)
        self.output_dir = (Path(__file__).parent / "outputs").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_dir / "config.json", 'w') as f:
            config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
            if config_dict.get('dtype'):
                config_dict['dtype'] = str(config_dict['dtype'])
            json.dump(config_dict, f, indent=2)
    
    def load_model(self):
        """Load and setup model with Unsloth optimizations"""
        logger.info(f"🦥 Loading Unsloth model: {self.config.model_name}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype,  
            load_in_4bit=self.config.load_in_4bit,
            attn_implementation="flash_attention_2",  
            device_map="auto",  
        )
        
        logger.info("🔧 Applying HIGH-RANK LoRA configuration with Unsloth for H200")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,  
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",
                          "embed_tokens", "lm_head"],  
            lora_alpha=self.config.lora_alpha, 
            lora_dropout=self.config.lora_dropout,  
            bias=self.config.bias,  
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,  
            random_state=self.config.random_state,
            use_rslora=self.config.use_rslora,  
            max_seq_length=self.config.max_seq_length,  
        )
        
        logger.info("📝 Setting up Llama-3.1 chat template")
        
        logger.info("✅ Model loaded successfully with Unsloth optimizations")
        
        # Print memory usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"📊 GPU Memory: {gpu_allocated:.1f}GB / {gpu_memory:.1f}GB used")
        
    def load_datasets(self) -> Dict[str, Dataset]:
        """Load train, validation, and test datasets from local splits directory"""
        logger.info("Loading train, validation, and test datasets from local splits directory...")

        datasets = {}
        splits_dir = (Path(__file__).parent.parent / "data_processing" / "splits").resolve()

        # Load train dataset
        train_path = splits_dir / "train.jsonl"
        if train_path.exists():
            train_data = []
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    train_data.append(json.loads(line))
            datasets['train'] = Dataset.from_list(train_data)
            logger.info(f"✅ Loaded {len(train_data)} training examples")
        else:
            raise FileNotFoundError(f"Training data not found: {train_path}")

        # Load validation dataset
        val_path = splits_dir / "valid.jsonl"
        if val_path.exists():
            val_data = []
            with open(val_path, 'r', encoding='utf-8') as f:
                for line in f:
                    val_data.append(json.loads(line))
            datasets['val'] = Dataset.from_list(val_data)
            logger.info(f"✅ Loaded {len(val_data)} validation examples")
        else:
            logger.warning("Validation data not found - training without validation")

        # Load test dataset
        test_path = splits_dir / "test.jsonl"
        if test_path.exists():
            test_data = []
            with open(test_path, 'r', encoding='utf-8') as f:
                for line in f:
                    test_data.append(json.loads(line))
            datasets['test'] = Dataset.from_list(test_data)
            logger.info(f"✅ Loaded {len(test_data)} test examples")
        else:
            logger.warning("Test data not found - will skip final testing")

        # Summary
        logger.info("📊 Dataset Summary:")
        logger.info(f"   🏋️ Train: {len(datasets.get('train', []))} examples")
        logger.info(f"   📝 Validation: {len(datasets.get('val', []))} examples") 
        logger.info(f"   🧪 Test: {len(datasets.get('test', []))} examples")
        total_examples = sum(len(ds) for ds in datasets.values())
        logger.info(f"   📊 Total: {total_examples} examples")

        return datasets
    
    def apply_chat_template_to_datasets(self, datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """Apply chat template to format datasets for training"""
        logger.info("🔧 Applying chat template to datasets...")
        
        def formatting_prompts_func(examples):
            """Format examples into conversation format for Llama"""
            texts = []
            for i in range(len(examples['instruction'])):
                # Combine instruction and input for the user message
                user_input = f"{examples['instruction'][i]} {examples['input'][i]}"
                
                # Create the formatted text using Llama format
                text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that answers questions about Vietnamese law.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{examples['output'][i]}<|eot_id|>"""
                
                texts.append(text)
            
            return {"text": texts}
        
        # Apply formatting to each dataset
        formatted_datasets = {}
        for split_name, dataset in datasets.items():
            logger.info(f"📝 Formatting {split_name} split ({len(dataset)} examples)")
            
            # Apply formatting function
            formatted_dataset = dataset.map(
                formatting_prompts_func, 
                batched=True,
                remove_columns=dataset.column_names
            )
            
            formatted_datasets[split_name] = formatted_dataset
            logger.info(f"✅ Formatted {split_name} split - {len(formatted_dataset)} examples")
            
            # Log example of formatted data
            if len(formatted_dataset) > 0:
                logger.info(f"📄 Example from {split_name} split:")
                example_text = formatted_dataset[0]['text']
                preview = example_text[:300] + "..." if len(example_text) > 300 else example_text
                logger.info(f"   Text preview: {preview}")
        
        logger.info("✅ Chat template applied to all datasets")
        return formatted_datasets
    
    def setup_trainer(self, datasets: Dict[str, Dataset]):
        """Setup SFT trainer with Unsloth optimizations"""
        logger.info("🔧 Setting up SFT trainer with Unsloth optimizations")
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            run_name=self.config.run_name,
            
            per_device_train_batch_size=self.config.per_device_train_batch_size,  
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,    
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,  
            
            learning_rate=self.config.learning_rate,  
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_steps=self.config.warmup_steps,  
            
            # Training duration
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            
            # Optimization 
            optim=self.config.optim,  
            fp16=self.config.fp16,    
            bf16=self.config.bf16,    
            
            # Data loading 
            dataloader_num_workers=getattr(self.config, 'dataloader_num_workers', 8),
            dataloader_pin_memory=getattr(self.config, 'dataloader_pin_memory', True),
            remove_unused_columns=False,  # Keep all columns
            
            # Advanced H200 optimizations
            ddp_find_unused_parameters=False, 
            gradient_checkpointing=True,       
            
            # Logging and saving
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            eval_strategy=self.config.evaluation_strategy,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            
            # Reproducibility
            seed=self.config.seed,
            
            # Monitoring
            report_to=["wandb"] if os.getenv("WANDB_API_KEY") else [],
        )
        
        # Setup SFT trainer with Unsloth
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=datasets['train'],
            eval_dataset=datasets.get('val'),
            dataset_text_field=self.config.dataset_text_field,
            packing=self.config.packing,  # False for legal accuracy
            args=training_args,
            max_seq_length=self.config.max_seq_length,
        )
        
        logger.info("✅ SFT Trainer setup complete")
        logger.info(f"📊 Training dataset size: {len(datasets['train'])}")
        if 'val' in datasets:
            logger.info(f"📊 Validation dataset size: {len(datasets['val'])}")
        if 'test' in datasets:
            logger.info(f"📊 Test dataset size: {len(datasets['test'])}")
        
        # H200 MAXIMIZATION INFO
        eff_batch = self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps
        logger.info(f"🚀 H200 MAXIMIZED CONFIG:")
        logger.info(f"   🎯 Effective batch size: {eff_batch} ")
        logger.info(f"   📏 Max sequence length: {self.config.max_seq_length} ")
        logger.info(f"   🔧 LoRA rank: {self.config.lora_r} ")
        
    def train(self):
        """Run training with Unsloth optimizations"""
        if not self.trainer:
            raise ValueError("Trainer not initialized. Call setup_trainer first.")
        
        logger.info("🚀 Starting training with Unsloth optimizations...")
        
        # Initialize wandb if API key is available
        if os.getenv("WANDB_API_KEY"):
            wandb.init(
                project="vietnamese-legal-llama-unsloth",
                name=self.config.run_name,
                config=self.config.__dict__
            )
            logger.info("📊 Weights & Biases monitoring initialized")
        
        # Print GPU memory before training
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"🔋 GPU Memory before training: {gpu_allocated:.1f}GB / {gpu_memory:.1f}GB")
        
        # Train with Unsloth speed optimizations
        trainer_stats = self.trainer.train()
        
        # Print GPU memory after training
        if torch.cuda.is_available():
            gpu_allocated_after = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"🔋 GPU Memory after training: {gpu_allocated_after:.1f}GB / {gpu_memory:.1f}GB")
        
        logger.info(f"✅ Training completed! Stats: {trainer_stats}")
        
        # Save model automatically
        self.save_model()
        
        return trainer_stats
    
    def save_model(self, save_method: str = "merged_16bit"):
        """Save the fine-tuned model and upload to Digital Ocean Spaces"""
        logger.info(f"Saving model using method: {save_method}")
        
        save_dir = self.output_dir / "final_model"
        save_dir.mkdir(exist_ok=True)
        
        if save_method == "merged_16bit":
            # Save merged model in 16bit
            self.model.save_pretrained_merged(
                str(save_dir), 
                self.tokenizer, 
                save_method="merged_16bit"
            )
        elif save_method == "lora":
            # Save only LoRA adapters
            self.model.save_pretrained(str(save_dir))
            self.tokenizer.save_pretrained(str(save_dir))
        elif save_method == "gguf":
            # Save in GGUF format for inference
            self.model.save_pretrained_gguf(
                str(save_dir), 
                self.tokenizer,
                quantization_method="q4_k_m"
            )
        
        logger.info(f"Model saved to {save_dir}")
        
        # Upload to Digital Ocean Spaces
        try:
            logger.info("Uploading model to Digital Ocean Spaces...")
            spaces_manager = DOSpacesManager()
            
            # Create unique model name with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"vietnamese-legal-llama-{timestamp}"
            
            # Upload model directory
            if spaces_manager.upload_directory(str(save_dir), f"models/{model_name}"):
                logger.info(f"✅ Model uploaded to DO Spaces: models/{model_name}")
                
                # Save model info
                model_info = {
                    "model_name": model_name,
                    "timestamp": timestamp,
                    "config": self.config.__dict__,
                    "spaces_path": f"models/{model_name}",
                    "save_method": save_method
                }
                
                # Upload model info
                info_path = self.output_dir / "model_info.json"
                with open(info_path, 'w') as f:
                    json.dump(model_info, f, indent=2, default=str)
                
                spaces_manager.upload_file(
                    str(info_path), 
                    f"models/{model_name}/model_info.json"
                )
                
                logger.info(f"Model info saved: {model_name}")
                
            else:
                logger.error("Failed to upload model to DO Spaces")
                
        except Exception as e:
            logger.error(f"Error uploading to DO Spaces: {e}")
            logger.info("Model saved locally only")
    
    def evaluate(self, eval_dataset=None, dataset_name="validation") -> Dict[str, float]:
        """Run evaluation on validation or test set"""
        if not self.trainer:
            raise ValueError("Trainer not initialized")
        
        logger.info(f"📊 Running {dataset_name} evaluation...")
        
        if eval_dataset is not None:
            # Temporarily replace eval dataset
            original_eval_dataset = self.trainer.eval_dataset
            self.trainer.eval_dataset = eval_dataset
            eval_results = self.trainer.evaluate()
            self.trainer.eval_dataset = original_eval_dataset
        else:
            eval_results = self.trainer.evaluate()
        
        # Save evaluation results
        eval_filename = f"{dataset_name}_eval_results.json"
        with open(self.output_dir / eval_filename, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Pretty print results
        logger.info(f"🎯 {dataset_name.title()} Results:")
        for key, value in eval_results.items():
            if 'loss' in key.lower():
                logger.info(f"   📈 {key}: {value:.4f}")
            elif 'runtime' in key.lower():
                logger.info(f"   ⏱️  {key}: {value:.2f} seconds")
            elif 'samples_per_second' in key.lower():
                logger.info(f"   🚀 {key}: {value:.2f}")
            else:
                logger.info(f"   📊 {key}: {value}")
        
        return eval_results
    
    def print_trainable_parameters(self):
        """Print trainable parameters info"""
        if self.model:
            trainable_params = 0
            all_param = 0
            for _, param in self.model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            
            print(f"Trainable params: {trainable_params:,} || "
                  f"All params: {all_param:,} || "
                  f"Trainable%: {100 * trainable_params / all_param:.4f}")

def main(gpu_type: str = "h200"):
    """Main training function with GPU-specific optimization"""
    import sys
    
    # Get GPU type from command line argument if provided
    if len(sys.argv) > 1:
        gpu_type = sys.argv[1].lower()
    
    # Import configs
    try:
        from configs import get_config, print_config_comparison
        config = get_config(gpu_type)
        
        # Print configuration comparison
        print_config_comparison()
        print(f"\n🎯 Selected GPU Configuration: {gpu_type.upper()}")
        
    except ImportError:
        # Fallback to H200 MAXIMIZED config if configs.py not available
        logger.warning("configs.py not found, using H200 MAXIMIZED configuration")
        config = FineTuneConfig(
            # Model settings - H200 MAXIMIZED
            model_name="unsloth/Llama-3.1-8B-Instruct",
            max_seq_length=8192,   # DOUBLED for H200 - longer legal documents
            load_in_4bit=False,    # H200 has enough VRAM for 16bit precision
            
            # LoRA settings - AGGRESSIVE for Vietnamese legal domain
            lora_r=128,      # DOUBLED rank for complex legal reasoning
            lora_alpha=256,  # 2 * lora_r for optimal performance
            lora_dropout=0.0,     # 0 for Unsloth optimization
            use_rslora=True,      # Enable for high rank stability
            
            # Training settings - H200 MAXIMIZED (141GB VRAM)
            per_device_train_batch_size=16,  # AGGRESSIVE batch size for H200
            gradient_accumulation_steps=8,   # Effective batch size = 16 * 8 = 128 (HUGE!)
            warmup_steps=100,      # More warmup for larger batch sizes
            num_train_epochs=3,    # Conservative for legal domain
            learning_rate=3e-4,    # INCREASED for larger effective batch size
            weight_decay=0.01,
            lr_scheduler_type="cosine",  # Better scheduler for longer training
            
            # Data loading optimization
            dataloader_num_workers=8,
            dataloader_pin_memory=True,
            
            # Output
            output_dir="/home/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/llm_finetuning_serving/finetune/outputs",
            run_name=f"vietnamese-legal-llama-h200-MAXIMIZED-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    
    # Update output directory (outputs/ nằm trong finetune/)
    config.output_dir = str((Path(__file__).parent / "outputs").resolve())

    # Data directory (splits/ nằm trong data_processing/)
    data_dir = str((Path(__file__).parent.parent / "data_processing" / "splits").resolve())

    # Initialize trainer
    logger.info("🚀 Initializing Vietnamese Legal LLM Fine-tuner with Unsloth")
    logger.info("� H200 MAXIMIZED Configuration (140GB VRAM AGGRESSIVE USAGE):")
    logger.info(f"   - Model: {config.model_name}")
    logger.info(f"   - Max sequence length: {config.max_seq_length} (DOUBLED from 4K to 8K)")
    logger.info(f"   - Precision: {'16-bit' if not config.load_in_4bit else '4-bit'} (H200 native bf16)")
    logger.info(f"   - LoRA rank: {config.lora_r}, alpha: {config.lora_alpha} (HIGH-RANK for legal complexity)")
    logger.info(f"   - Batch size: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps} (MASSIVE!)")
    logger.info(f"   - Learning rate: {config.learning_rate} (optimized for large batch)")
    logger.info(f"   - LR scheduler: {config.lr_scheduler_type}")
    logger.info(f"   - Epochs: {config.num_train_epochs}")
    logger.info(f"   - Optimizer: {config.optim} (full precision for H200)")
    logger.info(f"   - Expected VRAM: ~80-120GB (maximizing 140GB capacity)")
    logger.info(f"   - Data workers: {getattr(config, 'dataloader_num_workers', 8)} parallel")

    finetuner = LlamaFineTuner(config, data_dir)
    
    try:
        # Load model with Unsloth optimizations
        finetuner.load_model()
        finetuner.print_trainable_parameters()
        
        # Load datasets from Digital Ocean Spaces
        datasets = finetuner.load_datasets()
        
        # Apply chat template to format data properly
        formatted_datasets = finetuner.apply_chat_template_to_datasets(datasets)
        
        # Setup trainer with optimized settings
        finetuner.setup_trainer(formatted_datasets)
        
        # Train with Unsloth speed optimizations
        stats = finetuner.train()
        
        # Evaluate final model on validation set
        if 'val' in formatted_datasets:
            logger.info("📊 Running final validation evaluation...")
            eval_results = finetuner.evaluate()
        else:
            eval_results = {}
            
        # Evaluate on test set if available
        test_results = finetuner.evaluate_test_set(datasets)
        
        print("\\n" + "="*80)
        print("🎉 VIETNAMESE LEGAL LLM FINETUNE HOÀN THÀNH! (H200 MAXIMIZED)")
        print("="*80)
        print(f"🦥 Trained with Unsloth + H200 MAXIMUM utilization")
        print(f"🖥️  GPU: H200 (140GB VRAM) - AGGRESSIVE 16-bit precision")
        print(f"📏 Context length: {config.max_seq_length} tokens (DOUBLED to 8K)")
        print(f"🎯 LoRA rank: {config.lora_r} (HIGH-RANK {config.lora_r} for legal complexity)")
        print(f"📦 Effective batch: {config.per_device_train_batch_size * config.gradient_accumulation_steps} (MASSIVE for fast training)")
        print(f"📁 Model saved to: {config.output_dir}/final_model")
        print(f"⏱️  Training time: {stats.metrics.get('train_runtime', 0):.2f} seconds")
        print(f"🚀 Training speed: {stats.metrics.get('train_samples_per_second', 0):.2f} samples/sec")
        
        # Show training and validation losses
        print(f"\n� LOSS SUMMARY:")
        if hasattr(stats, 'log_history') and stats.log_history:
            final_train_loss = None
            for log_entry in reversed(stats.log_history):
                if 'train_loss' in log_entry:
                    final_train_loss = log_entry['train_loss']
                    break
            if final_train_loss is not None:
                print(f"   🏋️ Final Training Loss: {final_train_loss:.4f}")
        
        if eval_results:
            print(f"   📝 Final Validation Loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
        
        if test_results:
            print(f"   🧪 Final Test Loss: {test_results.get('eval_loss', 'N/A'):.4f}")
        
        print(f"\n💾 Model uploaded to Digital Ocean Spaces")
        print(f"🔥 H200 VRAM utilization: MAXIMIZED (~80-120GB/140GB)")
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()