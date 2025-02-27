
import torch 
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np


def llm_classifier(df):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_texts, val_texts, train_labels, val_labels = train_test_split(df['title'], df['label'], test_size=0.2)

    def compute_metrics(p):
        predictions, labels = p
        preds = np.argmax(predictions, axis=1)  # Convert logits to class predictions
        accuracy = accuracy_score(labels, preds)  # Calculate accuracy
        return {'accuracy': accuracy}  # Return as a dictionary


    def tokenize_function(examples):
        return tokenizer(examples, padding="max_length", truncation=True, max_length=512)

    train_data = {'title': train_texts, 'label': train_labels}
    val_data = {'title': val_texts, 'label': val_labels}

    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))

    train_dataset = train_dataset.map(lambda x: tokenize_function(x['title']), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x['title']), batched=True)

    train_dataset = train_dataset.remove_columns(["title"])
    val_dataset = val_dataset.remove_columns(["title"])

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()

    model.save_pretrained("finetuned_bert_model")
    tokenizer.save_pretrained("finetuned_bert_model")