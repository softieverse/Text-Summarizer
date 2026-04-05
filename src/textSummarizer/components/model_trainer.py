from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from textSummarizer.entity import ModelTrainerConfig
import torch
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        # Step 1 — load dataset from disk
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # Step 2 — split train into 90% train and 10% eval
        split = dataset_samsum_pt["train"].train_test_split(test_size=0.1)
        train_dataset = split["train"]
        eval_dataset = split["test"]

        # Step 3 — define training arguments
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=1,
            warmup_steps=500,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=500,
            save_steps=1e6,
            gradient_accumulation_steps=16,
            fp16=True,                          # half precision — cuts memory ~50%
            optim="adafactor",                  # much lighter optimizer than Adam
            gradient_checkpointing=True,        # trades compute for memory
)

        # Step 4 — initialize Trainer
        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            processing_class=tokenizer,   # <-- was 'tokenizer'
            data_collator=seq2seq_data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
)
        # Step 5 — train
        trainer.train()

        # Step 6 — save model and tokenizer
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))