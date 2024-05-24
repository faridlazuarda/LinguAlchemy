import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertModel, XLMRobertaForSequenceClassification, XLMRobertaModel, Trainer, TrainerCallback

import torch.nn.functional as F
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

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


class CustomTrainerDynamiclearn(Trainer):
    def __init__(self, lang_vec, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lang_vec = lang_vec.to(self.model.device)  # Ensure lang_vec is on the same device as the model
        self.lang_projection = nn.Linear(config.hidden_size, self.lang_vec.size(1)).to(self.model.device)
        
        self.scaling_factors = nn.Parameter(torch.ones(2, device=model.device), requires_grad=True)
        self.scale_optimizer = torch.optim.Adam([self.scaling_factors], lr=0.01)

    def _prepare_inputs(self, inputs):
        # Add additional labels to the inputs
        if "language_labels" in self.train_dataset.column_names:
            inputs["language_labels"] = inputs["language_labels"].to(self.model.device)
        if "uriel_labels" in self.train_dataset.column_names:
            inputs["uriel_labels"] = inputs["uriel_labels"].to(self.model.device)
        return super()._prepare_inputs(inputs)
    
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # factor = 10
        labels = inputs.pop("labels")
        language_labels = inputs.pop("language_labels")
        uriel_labels = inputs.pop("uriel_labels")

        logits, _, pooled_output = model(**inputs)[0]
        loss_fct = nn.CrossEntropyLoss()

        labels = labels.to(logits.device).long()

        loss_cls_raw = loss_fct(logits.view(-1, model.num_labels), labels.view(-1)) if labels is not None else 0
        

        # if language_labels is not None:
        #     language_labels = language_labels.to(lang_logits.device).long()
        #     loss_lid_raw = loss_fct(lang_logits, language_labels)
        # else:
        #     loss_lid = 0

        if uriel_labels is not None:        
            uriel_labels = uriel_labels.to(self.lang_vec.device).long()
            correct_uriel_vectors = self.lang_vec[uriel_labels].squeeze(1).float().to(device)

            projected_pooled_output = self.lang_projection(pooled_output).float().to(device)

            projected_pooled_output = projected_pooled_output.to(device)
            correct_uriel_vectors = correct_uriel_vectors.to(device)
            
            loss_uriel_raw = F.mse_loss(projected_pooled_output.unsqueeze(-1), correct_uriel_vectors)
        else:
            loss_uriel = 0
        
        scaling_factors_clone = self.scaling_factors.clone()

        # Apply scaling factors to each loss component
        loss_cls = loss_cls_raw * scaling_factors_clone[0]
        # loss_lid = loss_lid_raw * scaling_factors_clone[1]
        loss_uriel = loss_uriel_raw * scaling_factors_clone[1]

        total_loss = loss_cls + loss_uriel

        # self.step += 1  # Increment step
        
        # Update scaling factors if in training mode
        is_training_mode = labels is not None
        if is_training_mode:
            self.scale_optimizer.zero_grad()
            mini_loss = torch.abs(1.0 - self.scaling_factors.sum())
            mini_loss.backward(retain_graph=True)
            self.scale_optimizer.step()
        
        # print("loss_cls, scaling ", loss_cls, self.scaling_factors[0])
        # print("loss_lid, scaling ", loss_lid, self.scaling_factors[1])
        # print("loss_uriel, scaling ", loss_uriel, self.scaling_factors[2])
        
        if return_outputs:
            return (total_loss, logits)
        else:
            return total_loss



class CustomTrainerDynamicscale(Trainer):
    def __init__(self, lang_vec, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lang_vec = lang_vec.to(self.model.device)  # Ensure lang_vec is on the same device as the model
        self.lang_projection = nn.Linear(config.hidden_size, self.lang_vec.size(1)).to(self.model.device)
        
        self.scaling_factors = None
        self.initial_losses_set = False
        self.ema_alpha = 0.1  # Smoothing factor for EMA, can be adjusted
        self.update_frequency = 100  # Update the scaling factors every 100 steps
        self.step = 0

    def _prepare_inputs(self, inputs):
        # Add additional labels to the inputs
        if "language_labels" in self.train_dataset.column_names:
            inputs["language_labels"] = inputs["language_labels"].to(self.model.device)
        if "uriel_labels" in self.train_dataset.column_names:
            inputs["uriel_labels"] = inputs["uriel_labels"].to(self.model.device)
        return super()._prepare_inputs(inputs)
    
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # factor = 10
        labels = inputs.pop("labels")
        # language_labels = inputs.pop("language_labels")
        uriel_labels = inputs.pop("uriel_labels")

        logits, lang_logits, pooled_output = model(**inputs)[0]
        loss_fct = nn.CrossEntropyLoss()

        labels = labels.to(logits.device).long()

        loss_cls_raw = loss_fct(logits.view(-1, model.module.num_labels), labels.view(-1))
        

        # if language_labels is not None:
        #     language_labels = language_labels.to(lang_logits.device).long()
        #     loss_lid_raw = loss_fct(lang_logits, language_labels)
        # else:
        #     loss_lid = 0

        if uriel_labels is not None:        
            uriel_labels = uriel_labels.to(self.lang_vec.device).long()
            correct_uriel_vectors = self.lang_vec[uriel_labels].squeeze(1).float().to(device)

            projected_pooled_output = self.lang_projection(pooled_output).float().to(device)

            projected_pooled_output = projected_pooled_output.to(device)
            correct_uriel_vectors = correct_uriel_vectors.to(device)
            
            loss_uriel_raw = F.mse_loss(projected_pooled_output.unsqueeze(-1), correct_uriel_vectors)
        else:
            loss_uriel = 0
        
        # Dynamically set initial scaling factors if not already set
        if self.scaling_factors is None:
            initial_losses = torch.tensor([
                loss_cls_raw.item(), 
                # loss_lid_raw.item(), 
                loss_uriel_raw.item()
            ], device=logits.device)
            self.scaling_factors = initial_losses.mean() / initial_losses
            self.initial_losses_set = True
        else:
            current_losses = torch.tensor([
                loss_cls_raw.item(), 
                # loss_lid_raw.item(), 
                loss_uriel_raw.item()
            ], device=logits.device)
            if self.step % self.update_frequency == 0:
                # Update the scaling factors using EMA
                new_factors = current_losses.mean() / current_losses
                # More conservative adjustment for URIEL factor
                new_factors[1] = max(new_factors[1], self.scaling_factors[1] * 1.1)  # Slight increase for URIEL factor
                self.scaling_factors = self.ema_alpha * new_factors + (1 - self.ema_alpha) * self.scaling_factors


        # Apply scaling factors to each loss component
        loss_cls = loss_cls_raw * self.scaling_factors[0]
        # loss_lid = loss_lid_raw * self.scaling_factors[1]
        loss_uriel = loss_uriel_raw * self.scaling_factors[1]

        total_loss = loss_cls + loss_uriel

        self.step += 1  # Increment step
        
        # print("loss_cls, scaling ", loss_cls, self.scaling_factors[0])
        # print("loss_lid, scaling ", loss_lid, self.scaling_factors[1])
        # print("loss_uriel, scaling ", loss_uriel, self.scaling_factors[2])
        
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