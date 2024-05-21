import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertModel, XLMRobertaForSequenceClassification, XLMRobertaModel, Trainer, TrainerCallback

import torch.nn.functional as F
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

# Assuming 'device' is defined somewhere in your code, typically like this:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(preds, labels[0], average='macro')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Model Definition
class FusionBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, lang_vec):
        super().__init__(config)
        self.lang_vec = lang_vec
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.lang_projection = nn.Linear(config.hidden_size, lang_vec.size(1))
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, language_labels=None, uriel_labels=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs[1])
        logits = self.classifier(pooled_output)
        lang_logits = self.lang_projection(pooled_output)
        return ((logits.to(device), lang_logits.to(device), pooled_output.to(device)),)

# Uncomment for XLM-Roberta Model
class FusionXLMRForSequenceClassification(XLMRobertaForSequenceClassification):
    def __init__(self, config, lang_vec):
        super().__init__(config)
        self.lang_vec = lang_vec
        self.bert = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.lang_projection = nn.Linear(config.hidden_size, lang_vec.size(1))
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, language_labels=None, uriel_labels=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs[1])
        logits = self.classifier(pooled_output)
        lang_logits = self.lang_projection(pooled_output)
        return ((logits.to(device), lang_logits.to(device), pooled_output.to(device)),)


class CustomTrainer(Trainer):
    def __init__(self, config, scale, lang_vec, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lang_vec = lang_vec.to(self.model.device)
        self.scale = scale
        self.lang_projection = nn.Linear(config.hidden_size, self.lang_vec.size(1)).to(self.model.device)

    def _prepare_inputs(self, inputs):
        if "language_labels" in self.train_dataset.column_names:
            inputs["language_labels"] = inputs["language_labels"].to(self.model.device)
        if "uriel_labels" in self.train_dataset.column_names:
            inputs["uriel_labels"] = inputs["uriel_labels"].to(self.model.device)
        return super()._prepare_inputs(inputs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        uriel_labels = inputs.pop("uriel_labels")

        logits, _, pooled_output = model(**inputs)[0]
        loss_fct = nn.CrossEntropyLoss()
        labels = labels.to(logits.device).long()
        loss_cls = loss_fct(logits.view(-1, model.module.num_labels), labels.view(-1))

        if uriel_labels is not None:        
            uriel_labels = uriel_labels.to(self.lang_vec.device).long()
            correct_uriel_vectors = self.lang_vec[uriel_labels].squeeze(1).float().to(device)
            projected_pooled_output = self.lang_projection(pooled_output).float().to(device)
            loss_uriel = F.mse_loss(projected_pooled_output.unsqueeze(-1), correct_uriel_vectors) * self.scale
        else:
            loss_uriel = 0

        total_loss = loss_cls + loss_uriel
        if return_outputs:
            return (total_loss, logits)
        else:
            return total_loss

class EarlyStoppingEpochCallback(TrainerCallback):
    def __init__(self, early_stopping_patience):
        self.early_stopping_patience = early_stopping_patience
        self.best_loss = None
        self.patience_counter = 0

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.log_history:
            current_loss = state.log_history[-1].get('eval_loss')
            if current_loss is not None:
                if self.best_loss is None or current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        print(f"No improvement in evaluation loss for {self.early_stopping_patience} epochs. Stopping training.")
                        control.should_training_stop = True