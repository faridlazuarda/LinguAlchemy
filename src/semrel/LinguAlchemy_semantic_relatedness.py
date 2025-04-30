from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments, AutoConfig, XLMRobertaModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import csv
from io import StringIO
import requests
import re
from transformers import AutoTokenizer, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

uriel_data = torch.load('/home/alham.fikri/farid/adapters-lid/cache/lang_vec_semeval.pt')
uriel_vector = torch.stack([torch.tensor(uriel_data[lang]) for lang in sorted(uriel_data.keys())])



def download_data(url):
    data = csv.reader(StringIO(requests.get(url).text))
    next(data) # skip header
    delimiters = "[\n\t]"

    return [(re.split(delimiters, row[1])[0], re.split(delimiters, row[1])[1], float(row[2])) for row in data]

class SemanticSimilarityDataset(Dataset):
    def __init__(self, encodings, labels, lang_label, lang_uriel):
        self.encodings = encodings
        self.labels = labels
        self.lang_label = lang_label
        self.uriel_labels = uriel_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['lang_label'] = torch.tensor(self.lang_label[idx])
        item['uriel_labels'] = torch.tensor(self.uriel_labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Load tokenizer
# tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')


# Dummy data - replace with your dataset
sentences = []
for idx, lang in enumerate(["amh", "arq", "ary", "eng", "esp", "hau", "kin", "mar", "tel"]):
  tmp_sentences = download_data(f"https://raw.githubusercontent.com/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024/main/Track%20A/{lang}/{lang}_train.csv")
  # print(tmp_sentences)
  t_sentences = [tuple(x) + (idx,) + (uriel_vector[idx],) for x in tmp_sentences]

  print("load", lang, t_sentences[:2])
  sentences += t_sentences

print("Total data size", len(sentences))
# Preprocess data
encodings = tokenizer([s[0] for s in sentences], [s[1] for s in sentences], padding=True, truncation=True, return_tensors="pt")
labels = [s[2] for s in sentences]
lang_label = [s[3] for s in sentences]
uriel_labels = [s[4] for s in sentences]
dataset = SemanticSimilarityDataset(encodings, labels, lang_label, uriel_labels)

import torch
import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification

class BertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks with BERT."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to BERT's [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class FusionBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, lang_vec):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.lang_vec = lang_vec

        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = BertClassificationHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        language_labels=None,
        uriel_labels=None,
        labels=None
    ):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        logits = self.classifier(sequence_output)

        return ((logits.to(device), pooled_output.to(device)),)



class XLMRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class FusionXLMRForSequenceClassification(XLMRobertaForSequenceClassification):
    def __init__(self, config, lang_vec):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.lang_vec = lang_vec

        self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-large")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = XLMRobertaClassificationHead(config)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        language_labels=None,
        uriel_labels=None,
        labels=None
    ):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # print(outputs)

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)

        logits = self.classifier(sequence_output)
        # logits = torch.sigmoid(self.classifier(pooled_output))

        return ((logits.to(device), pooled_output.to(device)),)
    

# Load model
# model = XLMRForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=1).to('cuda')
# config = AutoConfig.from_pretrained('xlm-roberta-large', num_labels=1)
# model = FusionXLMRForSequenceClassification(config, uriel_vector)


config = AutoConfig.from_pretrained('bert-base-multilingual-cased', num_labels=1)
model = FusionBertForSequenceClassification(config, uriel_vector)

model = model.to(device)



# special loss

class CustomTrainer(Trainer):
    def __init__(self, lang_vec, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lang_vec = lang_vec.to(self.model.device)  # Ensure lang_vec is on the same device as the model
        self.lang_projection = nn.Linear(config.hidden_size, self.lang_vec.size(1)).to(self.model.device)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)[0]
        logits = outputs[0].squeeze()
        pooled_output = outputs[1].squeeze()

        uriel_labels = inputs.pop("uriel_labels")

        # Custom loss computation
        loss_fct = torch.nn.MSELoss()
        loss_semantic = loss_fct(logits, labels.float())
        loss_uriel = 0



        factor=1
        uriel_labels = uriel_labels.to(self.lang_vec.device).long()
        correct_uriel_vectors = self.lang_vec[uriel_labels].squeeze(1).float().to(device)

        projected_pooled_output = self.lang_projection(pooled_output).float().to(device)

        projected_pooled_output = projected_pooled_output.to(device)
        correct_uriel_vectors = correct_uriel_vectors.to(device)

        loss_uriel = F.mse_loss(projected_pooled_output.unsqueeze(-1), correct_uriel_vectors)*factor


        loss = loss_semantic + loss_uriel

        return (loss, outputs) if return_outputs else loss

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=5e-5,
    logging_dir='./logs',
    logging_steps=100,
    seed=42
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    lang_vec=uriel_vector
)

# Train the model
trainer.train()

model.save_pretrained('semeval-lingualchemy-15apr-mbert.pt')
