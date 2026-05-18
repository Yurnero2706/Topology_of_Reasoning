import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, concatenate_datasets, DatasetDict
import transformers
import trl

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)

def train():
    os.environ["WANDB_DISABLED"] = "true"
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {}
    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "use_cache": False}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    else:
        # Load in bf16 when bf16 training is requested so the full model
        # fits in VRAM before the sharding layer (FSDP/DeepSpeed) takes over.
        # Without this, a 14B model defaults to float32 (~58 GB) and OOMs
        # immediately on a 48 GB GPU before any sharding can help.
        _dtype = "bfloat16" if getattr(args, "bf16", False) else None
        _kw = {"torch_dtype": _dtype, "use_cache": False} if _dtype else {}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **_kw)

    dataset = load_dataset(config.train_file_path)

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"

    # If the dataset was loaded raw (e.g. simplescaling/s1K instead of
    # simplescaling/s1K_tokenized) it won't have a 'text' column.
    # Format it using the model's chat template so SFTTrainer can proceed.
    first_split = next(iter(dataset))
    if "text" not in dataset[first_split].features:
        logging.info("'text' column not found — applying chat template to format dataset.")

        def _format(example):
            if "messages" in example:
                messages = example["messages"]
            elif "question" in example and "solution" in example:
                messages = [
                    {"role": "user",      "content": example["question"]},
                    {"role": "assistant", "content": example["solution"]},
                ]
            elif "question" in example and "answer" in example:
                messages = [
                    {"role": "user",      "content": example["question"]},
                    {"role": "assistant", "content": example["answer"]},
                ]
            else:
                raise ValueError(
                    f"Cannot build 'text' field: dataset columns are {list(example.keys())}"
                )
            return {"text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )}

        dataset = dataset.map(_format, desc="Formatting dataset")

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    trainer = trl.SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()