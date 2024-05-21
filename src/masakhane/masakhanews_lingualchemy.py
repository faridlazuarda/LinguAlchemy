import os
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoTokenizer,
    AutoConfig
)
from torch.utils.data import DataLoader
import pandas as pd

import argparse
from tqdm import tqdm
import json
from csv import writer
import sys
import torch
import random
import numpy as np
from utils.modelling import FusionBertForSequenceClassification, FusionXLMRForSequenceClassification, LinguAlchemyTrainer, EarlyStoppingEpochCallback

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def evaluate_model(model, dset_test_dict):
    model.eval()
    all_preds, all_labels = [], []
    results = {}
    for lang, dataset in dset_test_dict.items():
        test_loader = DataLoader(dataset, batch_size=64)
        for batch in tqdm(test_loader, desc=f"Testing lang {lang}"):
            input_ids, attention_mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
            with torch.no_grad():
                logits = model(input_ids, attention_mask)[0][0]
                # print(logits)
                # print(logits)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        results[lang] = compute_language_metrics(all_preds, all_labels)
    return results

def compute_language_metrics(all_preds, all_labels):
    metrics = {}
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')
    accuracy = accuracy_score(all_labels, all_preds)
    metrics['precision_macro'], metrics['precision_micro'] = precision_macro, precision_micro
    metrics['recall_macro'], metrics['recall_micro'] = recall_macro, recall_micro
    metrics['f1_macro'], metrics['f1_micro'], metrics['accuracy'] = f1_macro, f1_micro, accuracy
    return metrics

def save_results(results, result_csv_path):
    df_result_full = pd.DataFrame(results).T
    df_result_full.to_csv(result_csv_path)
    print(results)

def main(args):
    set_seed(42)


    config = AutoConfig.from_pretrained(args.model, num_labels=7)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    uriel_data = torch.load(args.uriel_path)
    uriel_vector = torch.stack([torch.tensor(uriel_data[lang]) for lang in sorted(uriel_data.keys())])
    lang_to_index = {lang: idx for idx, lang in enumerate(sorted(uriel_data.keys()))}

    if args.model == "bert-base-multilingual-cased":
        model = FusionBertForSequenceClassification(config, uriel_vector).to(DEVICE)
    elif "xlm" in args.model:
        model = FusionXLMRForSequenceClassification(config, uriel_vector).to(DEVICE)


    train_langs = ["amh","eng", "fra", "hau", "swa", "orm", "som"]
    test_langs = ["ibo", "lin", "lug", "pcm", "run", "sna", "tir", "xho", "yor"]

    dataset_train, dataset_valid, dataset_test = [], [], {}
    columns_to_remove = ["text", "headline_text", "url"]

    for lang in train_langs:
        dataset = load_dataset('masakhane/masakhanews', lang)
        dataset_train.append(dataset['train'].remove_columns(columns_to_remove).map(lambda example: {'lang': lang}, batched=False))
        dataset_valid.append(dataset['validation'].remove_columns(columns_to_remove).map(lambda example: {'lang': lang}, batched=False))

    for lang in train_langs+test_langs:
        dataset = load_dataset('masakhane/masakhanews', lang)
        dataset_test[lang] = dataset['test'].remove_columns(columns_to_remove).map(lambda example: {'lang': lang}, batched=False)

    dset_dict = DatasetDict({
        'train': concatenate_datasets(dataset_train),
        'valid': concatenate_datasets(dataset_valid)
    })
    dset_test_dict = DatasetDict(dataset_test)



    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        # print(batch['lang'])
        lang_label = batch['lang'][0]  # Use the first locale in the list as the language label
        encoding = tokenizer(batch["headline"], max_length=80, truncation=True, padding="max_length", return_tensors="pt")
        
        # Language labels as indices
        lang_index = lang_to_index[lang_label]

        # URIEL vectors
        uriel_vec = uriel_vector[lang_index]
        
        encoding['uriel_labels'] = uriel_vec.repeat(len(batch['headline']), 1)
        return encoding

    dset_dict = dset_dict.map(encode_batch, batched=True).rename_column("label", "labels").with_format(type="torch", columns=["input_ids", "attention_mask", "labels", "uriel_labels"])
    dset_test_dict = dset_test_dict.map(encode_batch, batched=True).rename_column("label", "labels").with_format(type="torch", columns=["input_ids", "attention_mask", "labels", "uriel_labels"])


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
        save_strategy="epoch",
        report_to='tensorboard',
        seed=42,
    )

    # Trainer
    trainer = LinguAlchemyTrainer(
        model=model,
        config=config,
        args=training_args,
        train_dataset=dset_dict['train'],
        compute_metrics=compute_language_metrics,
        callbacks=[EarlyStoppingEpochCallback(3)],
        lang_vec=uriel_vector,
        uriel_factor=args.uriel_factor
    )

    # Training
    trainer.train()
    trainer.save_model(args.output_dir)

    # Evaluation on test datasets
    # test_metrics = {}
    # for name, test_dset in dset_test_dict.items():
    #     print(test_dset)
    #     test_metrics[name] = trainer.evaluate(test_dset)
    #     print(f"Evaluation for {name}: {test_metrics[name]}")

    results = evaluate_model(model, dset_test_dict)
    save_results(results, f"./results/{args.exp_name}.csv")

    # # Save results to CSV and JSON
    # with open(f"./results/{args.exp_name}.csv", 'w') as f:
    #     writer_object = writer(f)
    #     header = ["language"] + list(test_metrics[next(iter(test_metrics))].keys())
    #     writer_object.writerow(header)

    #     for language, metrics in test_metrics.items():
    #         row = [language] + list(metrics.values())
    #         writer_object.writerow(row)
    # with open(f"./results/{args.exp_name}.json", "w") as outfile:
    #     json.dump(test_metrics, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, help="Experiment Name", default="LinguAlchemy-masakhanews")

    args, _ = parser.parse_known_args()

    default_language_model = "xlm-roberta-base"
    default_model_save_path = '/home/alham.fikri/farid/lingualchemy/masakhanews/{}.pt'.format(args.exp_name)
    default_uriel_path = '/home/alham.fikri/farid/adapters-lid/cache/lang_vec_masakhanews.pt'
    default_result_csv_path = "/home/alham.fikri/farid/adapters-lid/result/result-{}.csv".format(args.exp_name)
    default_training_output_dir = "/mnt/beegfs/farid/lingualchemy/masakhanews/training_output-{}".format(args.exp_name)


    parser.add_argument('--model', type=str, default=default_language_model)
    parser.add_argument('--output-dir', type=str, default="/mnt/beegfs/farid/lingualchemy/masakhanews/")
    parser.add_argument("--uriel_path", type=str, help="URIEL Vector path", default=default_uriel_path)
    parser.add_argument("--model_save_path", type=str, help="Model save path", default=default_model_save_path)
    parser.add_argument("--result_csv_path", type=str, help="Result save path", default=default_result_csv_path)
    parser.add_argument("--training_output_dir", type=str, help="Training output directory", default=default_training_output_dir)
    parser.add_argument("--uriel_factor", type=str, help="Uriel factor", default=10)


    main(parser.parse_args())

