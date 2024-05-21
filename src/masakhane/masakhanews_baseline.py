import os
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoTokenizer
)
import argparse
from tqdm import tqdm
import json
from csv import writer
import sys
import torch
import random
import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

print(torch.cuda.is_available())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(preds, labels, average='macro')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def main(args):
    set_seed(42)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=7)
    tokenizer = AutoTokenizer.from_pretrained(args.model)


    train_langs = ["amh","eng", "fra", "hau", "swa", "orm", "som"]
    test_langs = ["ibo", "lin", "lug", "pcm", "run", "sna", "tir", "xho", "yor"]

    dataset_train, dataset_valid, dataset_test = [], [], {}
    columns_to_remove = ["text", "headline_text", "url"]

    for lang in train_langs:
        dataset = load_dataset('masakhane/masakhanews', lang)
        dataset_train.append(dataset['train'].remove_columns(columns_to_remove))
        dataset_valid.append(dataset['validation'].remove_columns(columns_to_remove))

    for lang in train_langs+test_langs:
        dataset = load_dataset('masakhane/masakhanews', lang)
        dataset_test[lang] = dataset['test'].remove_columns(columns_to_remove)

    dset_dict = DatasetDict({
        'train': concatenate_datasets(dataset_train),
        'valid': concatenate_datasets(dataset_valid)
    })
    dset_test_dict = DatasetDict(dataset_test)


    def encode_batch(batch):
        return tokenizer(batch["headline"], max_length=80, truncation=True, padding="max_length")

    dset_dict = dset_dict.map(encode_batch, batched=True).rename_column("label", "labels").with_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dset_test_dict = dset_test_dict.map(encode_batch, batched=True).rename_column("label", "labels").with_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"/mnt/beegfs/farid/lingualchemy/train-logs/training_output-{args.model}-masakhanews",
        learning_rate=5e-5,
        num_train_epochs=30,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        dataloader_num_workers=32,
        logging_steps=100,
        save_total_limit=2,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to='tensorboard',
        seed=42,
        metric_for_best_model='f1'
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dset_dict['train'],
        eval_dataset=dset_dict['valid'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(3)]
    )

    # Training
    trainer.train()
    # trainer.save_model(args.output_dir)

    # Evaluation on test datasets
    test_metrics = {}
    for name, test_dset in dset_test_dict.items():
        test_metrics[name] = trainer.evaluate(test_dset)
        print(f"Evaluation for {name}: {test_metrics[name]}")

        
    # Save results to CSV and JSON
    with open(f"./results/{args.exp_name}.csv", 'w') as f:
        writer_object = writer(f)
        header = ["language"] + list(test_metrics[next(iter(test_metrics))].keys())
        writer_object.writerow(header)

        for language, metrics in test_metrics.items():
            row = [language] + list(metrics.values())
            writer_object.writerow(row)
    with open(f"./results/{args.exp_name}.json", "w") as outfile:
        json.dump(test_metrics, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, help="Experiment Name", default="baseline-masakhanews")

    args, _ = parser.parse_known_args()

    default_language_model = "xlm-roberta-base"
    default_model_save_path = '/home/alham.fikri/farid/lingualchemy/masakhanews/{}.pt'.format(args.exp_name)
    default_uriel_path = '/home/alham.fikri/farid/adapters-lid/cache/lang_vec_masakhanews.pt'
    default_result_csv_path = "/home/alham.fikri/farid/adapters-lid/results/result-{}.csv".format(args.exp_name)
    default_training_output_dir = "/mnt/beegfs/farid/lingualchemy/masakhanews/training_output-{}".format(args.exp_name)


    parser.add_argument('--model', type=str, default=default_language_model)
    parser.add_argument('--output-dir', type=str, default="/mnt/beegfs/farid/lingualchemy/masakhanews/")
    parser.add_argument("--uriel_path", type=str, help="URIEL Vector path", default=default_uriel_path)
    parser.add_argument("--model_save_path", type=str, help="Model save path", default=default_model_save_path)
    parser.add_argument("--result_csv_path", type=str, help="Result save path", default=default_result_csv_path)
    parser.add_argument("--training_output_dir", type=str, help="Training output directory", default=default_training_output_dir)
    
    
    main(parser.parse_args())
