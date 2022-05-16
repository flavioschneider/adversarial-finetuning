import wandb
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse
from torchmetrics import MetricCollection, Accuracy, F1Score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Adafactor
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from smart_pytorch import SMARTLoss

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=int, default=0)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--smart-loss-weight", type=float, default=1.0)
parser.add_argument("--pretrained-model", type=str, default='roberta-base')
parser.add_argument("--sequence-length", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

wandb.login()
pl.seed_everything(args.seed)

class MCTACODataset(Dataset):

    def __init__(self, split: str, tokenizer, sequence_length: int):
        self.dataset = load_dataset("mc_taco")[split]
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.dataset)

    def truncate_pair(self, tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def __getitem__(self, idx): 
        item = self.dataset[idx] 
        tokenize = self.tokenizer.tokenize
        sequence = tokenize(item['sentence'] + " " + item['question'])
        answer = tokenize(item['answer']) 
        label = item['label']
        # Truncate excess tokens 
        if answer: 
            self.truncate_pair(sequence, answer, self.sequence_length - 3)
        else: 
            if len(sequence) > self.sequence_length - 2:
                sequence = sequence[0:(self.sequence_length - 2)]
        # Compute tokens, ids, mask 
        tokens = ['<s>'] + sequence + ['</s></s>'] + answer + ['</s>']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # Pad with 0 
        while len(input_ids) < self.sequence_length:
            input_ids.append(0)
            input_mask.append(0)
        return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(label)
    

class MCTACODatamodule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        batch_size: int,
        sequence_length: int 
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.dataset_train = None
        self.dataset_valid = None

    def setup(self, stage = None):
        self.dataset_train = MCTACODataset(
            split='validation', 
            tokenizer=self.tokenizer, 
            sequence_length=self.sequence_length
        )
        self.dataset_valid = MCTACODataset(
            split='test', 
            tokenizer=self.tokenizer, 
            sequence_length=self.sequence_length
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset_valid,
            batch_size=self.batch_size,
            shuffle=False,
        )


"""
def kl_loss(s_p, s):
    # s_p: perturbed state, s: initial state 
    s_p = F.log_softmax(s_p, dim=1) # (b, n)
    s = F.log_softmax(s, dim=1) # (b, n)
    l0 = F.kl_div(s_p, s, reduction = 'sum', log_target=True)
    l1 = F.kl_div(s, s_p, reduction = 'sum', log_target=True)
    return l0 + l1
"""

def kl_loss(input, target, reduction='sum'):
    return F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target.detach(), dim=-1),
        reduction=reduction,
    ) + F.kl_div(
        F.log_softmax(target, dim=-1),
        F.softmax(input.detach(), dim=-1),
        reduction=reduction,
    )

class SMARTClassificationModel(nn.Module):
    # b: batch_size, s: sequence_length, d: hidden_size , n: num_labels

    def __init__(self, model, weight):
        super().__init__()
        self.model = model 
        self.weight = weight

    def forward(self, input_ids, attention_mask, labels):
        # input_ids: (b, s), attention_mask: (b, s), labels: (b,)

        embed = self.model.roberta.embeddings(input_ids) # (b, s, d)

        def eval(embed):
            outputs = self.model.roberta(inputs_embeds=embed, attention_mask=attention_mask) # (b, s, d)
            pooled = outputs[0] # (b, d)
            logits = self.model.classifier(pooled) # (b, n)
            return logits 

        smart_loss_fn = SMARTLoss(eval_fn = eval, loss_fn = kl_loss)
        state = eval(embed)
        loss = F.cross_entropy(state.view(-1, 2), labels.view(-1))
        smart_loss = torch.tensor(0)
        if embed.requires_grad:
            smart_loss = smart_loss_fn(embed, state)
            loss += self.weight * smart_loss
       
        return state, loss


class TextClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module
    ):
        super().__init__()
        self.model = model
        metrics = MetricCollection([ Accuracy(), F1Score() ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.valid_metrics = metrics.clone(prefix='val_')

    def configure_optimizers(self):
        optimizer = Adafactor(self.model.parameters(), warmup_init=True)
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids, attention_masks, labels = batch
        # Compute output 
        outputs, loss = self.model(input_ids = input_ids, attention_mask = attention_masks, labels = labels)
        labels_pred = torch.argmax(outputs, dim=1)
        # Compute metrics
        metrics = self.train_metrics(labels, labels_pred)
        # Log loss and metrics
        self.log("train_loss", loss, on_step=True)
        self.log_dict(metrics, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_masks, labels = batch
        # Compute output 
        outputs, loss = self.model(input_ids = input_ids, attention_mask = attention_masks, labels = labels)
        labels_pred = torch.argmax(outputs, dim=1)
        # Compute metrics
        metrics = self.valid_metrics(labels, labels_pred)
        # Log loss and metrics
        self.log("valid_loss", loss, on_step=True)
        self.log_dict(metrics, on_step=True, on_epoch=True)
        return loss

tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
architecture = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model)    

datamodule = MCTACODatamodule(tokenizer, batch_size = args.batch_size, sequence_length = args.sequence_length) 
datamodule.setup()      

smart_architecture = SMARTClassificationModel(architecture, weight=args.smart_loss_weight)
model = TextClassificationModel(smart_architecture)
# input_ids, input_mask, labels = next(iter(datamodule.train_dataloader()))   
# output, loss = smart_architecture(input_ids, input_mask, labels)

# Wandb Logger
logger = pl.loggers.wandb.WandbLogger(project = 'cs4nlp', entity='nextmachina')
# Callbacks 
cb_progress_bar = pl.callbacks.RichProgressBar()
cb_model_summary = pl.callbacks.RichModelSummary()
# Train 
trainer = pl.Trainer(logger=logger, callbacks=[cb_progress_bar, cb_model_summary], max_epochs=args.epochs, gpus=args.gpus)
trainer.logger.log_hyperparams(args)
trainer.fit(model=model, datamodule=datamodule)
wandb.finish() 