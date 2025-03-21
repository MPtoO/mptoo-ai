"""
Script de fine-tuning LoRA pour MPTOO
Copyright 2025 Mohammed Amine Taybi
Licensed under the Apache License, Version 2.0
"""

import os
import argparse
from typing import Dict, Any, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import wandb
from trl import SFTTrainer


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tuning avec LoRA pour LLM")
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Modèle de base à fine-tuner",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mptoo/domain-data",
        help="Nom du dataset Hugging Face",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Répertoire de sortie pour le modèle",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="Rank de LoRA",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="Alpha de LoRA",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Nombre d'époques d'entraînement",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Taux d'apprentissage",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Taille des lots d'entraînement",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Utiliser Weights & Biases pour le tracking",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mptoo-fine-tuning",
        help="Nom du projet W&B",
    )
    
    return parser.parse_args()


def setup_lora_config(args) -> LoraConfig:
    """Configure LoRA pour le fine-tuning"""
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def prepare_model_and_tokenizer(args):
    """Prépare le modèle et le tokenizer pour le fine-tuning"""
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Charger le modèle en quantized 8-bit
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Préparer le modèle pour la quantization
    model = prepare_model_for_kbit_training(model)
    
    # Appliquer LoRA
    lora_config = setup_lora_config(args)
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer


def prepare_dataset(args, tokenizer):
    """Prépare le dataset pour le fine-tuning"""
    # Charger le dataset
    dataset = load_dataset(args.dataset_name)
    
    # Fonction de prétraitement
    def preprocess_function(examples):
        # Format attendu: "instruction: <instruction>\ncontext: <context>\nresponse: <response>"
        # Adaptez selon votre format de données
        texts = []
        for instruction, context, response in zip(
            examples["instruction"], examples["context"], examples["response"]
        ):
            text = f"instruction: {instruction}\ncontext: {context}\nresponse: {response}"
            texts.append(text)
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
    
    # Appliquer le prétraitement
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    return tokenized_dataset


def setup_training_args(args) -> TrainingArguments:
    """Configure les arguments d'entraînement"""
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        fp16=True,
        report_to="wandb" if args.use_wandb else "none",
    )
    
    return training_args


def train_model(args):
    """Exécute le fine-tuning du modèle"""
    # Configuration W&B si activé
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=f"lora-{args.base_model.split('/')[-1]}")
    
    # Préparer modèle et tokenizer
    model, tokenizer = prepare_model_and_tokenizer(args)
    print(f"Modèle {args.base_model} chargé avec LoRA")
    
    # Préparer le dataset
    tokenized_dataset = prepare_dataset(args, tokenizer)
    print(f"Dataset {args.dataset_name} préparé")
    
    # Configuration de l'entraînement
    training_args = setup_training_args(args)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation", None),
        data_collator=data_collator,
    )
    
    # Entraînement
    print("Démarrage de l'entraînement...")
    trainer.train()
    
    # Sauvegarde du modèle
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"Modèle fine-tuné sauvegardé dans {final_model_path}")
    
    # Terminer W&B si utilisé
    if args.use_wandb:
        wandb.finish()
    
    return final_model_path


def main():
    """Fonction principale"""
    args = parse_args()
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Lancer le fine-tuning
    output_path = train_model(args)
    
    print("Fine-tuning terminé avec succès!")
    print(f"Modèle sauvegardé dans: {output_path}")


if __name__ == "__main__":
    main() 