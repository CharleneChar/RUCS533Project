import argparse
import pandas as pd
import numpy as np
import random
import os
from transformers import Trainer, pipeline, set_seed, BartTokenizer, AutoTokenizer, BartForConditionalGeneration, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
nltk.download('punkt')
import csv
from datasets import load_dataset
import evaluate
from textblob import TextBlob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='bart', choices=['bart'])
    parser.add_argument('-s', '--setting', default='unconstrained', choices=['unconstrained', 'controlled', 'predict'])
    parser.add_argument('--train', default='data/wholetrain.csv')
    parser.add_argument('--dev', default='data/wholedev.csv')
    parser.add_argument('--test', default='data/wholetest.csv')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args

#BART
def run_bart_unconstrained():
    # Define tokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    def preprocess_function(examples):
        inputs, targets = examples["original_text"], examples["reframed_text"]
        model_inputs = tokenizer(text=inputs, max_length=1024, truncation=True)
        labels = tokenizer(text_target=targets, max_length=1024, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    # Prepcrocess datasets
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True) # can instead use tokenized_train_datasets.shuffle(seed=42).select(range(1000)) to get a smaller set
    tokenized_dev_dataset = dev_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
    # Define model to train
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)  
    batch_size = 6
    # Define training parameters
    args = Seq2SeqTrainingArguments(
        "test-summarization",
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True,
        optim="adamw_torch",
        ddp_find_unused_parameters=False
    ) 
    # Define metrics
    metric1 = evaluate.load("rouge")
    metric2 = evaluate.load("sacrebleu")
    metric3 = evaluate.load("bertscore") 
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Expect a newline after each sentence when using rouge
        decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        # Add rouge
        result1 = metric1.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
        result = {key: value for key, value in result1.items()}
        
        # Add average generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        # Add bleu
        decoded_labels_expanded = [[x] for x in decoded_labels]
        result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)
        result['sacrebleu'] = round(result2["score"], 1)
        
        return {k: round(v, 4) for k, v in result.items()}
    # Start training and validating
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_dataset["train"],
        eval_dataset=tokenized_dev_dataset["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model("output/reframer")

    # Load trained model
    model = AutoModelForSeq2SeqLM.from_pretrained("output/reframer")
    tokenizer = AutoTokenizer.from_pretrained("output/reframer")
    reframer = pipeline('summarization', model=model, tokenizer=tokenizer)
    # Start testing
    test = pd.read_csv(test_path)
    texts, truths = test['original_text'].to_list(), test['reframed_text'].to_list()
    reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]
    predictions = []

    with open(os.path.join(path,'bart_unconstrained.txt'), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)
            predictions.append(str(item))
    
    # Compute rouge (on test dataset)
    result1 = metric1.compute(predictions=predictions, references=truths)
    result = {key: value for key, value in result1.items()}
 
    # Compute bleu (on test dataset)
    result2 = metric2.compute(predictions=predictions, references=truths)
    result['sacrebleu'] = round(result2["score"], 1)

    # Compute bert score (on test dataset)
    result3 = metric3.compute(predictions=predictions, references=truths, lang="en")
    result['bertscore'] = sum(result3["f1"]) / len(result3["f1"])       
       
    # Compute delta textblob (on test dataset)
    total_textblob = 0
    for original, reframed in zip(texts, predictions):
        total_textblob += (TextBlob(reframed).sentiment.polarity - TextBlob(original).sentiment.polarity)
    result['delta_textblob'] = round(total_textblob / len(texts), 4)

    # Compute average length (on test dataset)
    prediction_lens = [len(pred.split()) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    # Display every evaluation metric (for test dataset)
    for k, v in result.items():
        print(f'{k}: {round(v, 4)}')


def run_bart_controlled():
    # Define tokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    def preprocess_function(examples):
        inputs, targets = examples["original_with_label"], examples["reframed_text"]
        model_inputs = tokenizer(text=inputs, max_length=1024, truncation=True)
        labels = tokenizer(text_target=targets, max_length=1024, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    # Preprocess datasets
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_dev_dataset = dev_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
    # Define model to train
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    batch_size = 6
    # Define training parameters
    args = Seq2SeqTrainingArguments(
        "test-summarization",
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True,
        optim="adamw_torch",
        ddp_find_unused_parameters=False
    )
    # Define metrics
    metric1 = evaluate.load("rouge")
    metric2 = evaluate.load("sacrebleu")
    metric3 = evaluate.load("bertscore") 
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Expect a newline after each sentence when using rouge
        decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        # Add rouge
        result1 = metric1.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
        result = {key: value for key, value in result1.items()}
        
        # Add average generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        # Add bleu
        decoded_labels_expanded = [[x] for x in decoded_labels]
        result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)
        result['sacrebleu'] = round(result2["score"], 1)
        
        return {k: round(v, 4) for k, v in result.items()}
    # Start training and validating
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_dataset["train"],
        eval_dataset=tokenized_dev_dataset["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model("output/reframer")
    
    # Load trained model
    model = AutoModelForSeq2SeqLM.from_pretrained("output/reframer")
    tokenizer = AutoTokenizer.from_pretrained("output/reframer")
    reframer = pipeline('summarization', model=model, tokenizer=tokenizer)
    # Start testing
    test = pd.read_csv(test_path)
    texts, truths = test['original_with_label'].to_list(), test['reframed_text'].to_list()
    reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]
    predictions = []

    with open(os.path.join(path,'bart_controlled.txt'), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)
            predictions.append(str(item))
    
    # Compute rouge (on test dataset)
    result1 = metric1.compute(predictions=predictions, references=truths)
    result = {key: value for key, value in result1.items()}
 
    # Compute bleu (on test dataset)
    result2 = metric2.compute(predictions=predictions, references=truths)
    result['sacrebleu'] = round(result2["score"], 1)

    # Compute bert score (on test dataset)
    result3 = metric3.compute(predictions=predictions, references=truths, lang="en")
    result['bertscore'] = sum(result3["f1"]) / len(result3["f1"])       
       
    # Compute delta textblob (on test dataset)
    total_textblob = 0
    for original, reframed in zip(texts, predictions):
        total_textblob += (TextBlob(reframed).sentiment.polarity - TextBlob(original).sentiment.polarity)
    result['delta_textblob'] = round(total_textblob / len(texts), 4)

    # Compute average length (on test dataset)
    prediction_lens = [len(pred.split()) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    # Display every evaluation metric (for test dataset)
    for k, v in result.items():
        print(f'{k}: {round(v, 4)}')


def run_bart_predict():
    # Define tokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    def preprocess_function(examples):
        inputs, targets = examples["original_text"], examples["strategy_reframe"]
        model_inputs = tokenizer(text=inputs, max_length=1024, truncation=True)
        labels = tokenizer(text_target=targets)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    # Preprocess datasets
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_dev_dataset = dev_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
    # Define model to train
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    batch_size = 6
    # Define training parameters
    args = Seq2SeqTrainingArguments(
        "test-summarization",
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True,
        optim="adamw_torch",
        ddp_find_unused_parameters=False
    )
    # Define metrics   
    metric1 = evaluate.load("rouge")
    metric2 = evaluate.load("sacrebleu")
    metric3 = evaluate.load("bertscore") 
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Expect a newline after each sentence when using rouge
        decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        # Add rouge
        result1 = metric1.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
        result = {key: value for key, value in result1.items()}
        
        # Add average generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        # Add bleu
        decoded_labels_expanded = [[x] for x in decoded_labels]
        result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)
        result['sacrebleu'] = round(result2["score"], 1)
        
        return {k: round(v, 4) for k, v in result.items()}
    # Start training and validating
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_dataset["train"],
        eval_dataset=tokenized_dev_dataset["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model("output/reframer")

    # Load trained model
    model = AutoModelForSeq2SeqLM.from_pretrained("output/reframer")
    tokenizer = AutoTokenizer.from_pretrained("output/reframer")
    reframer = pipeline('summarization', model=model, tokenizer=tokenizer)
    # Start testing
    test = pd.read_csv(test_path)
    texts, truths = test['original_text'].to_list(), test['strategy_reframe'].to_list()
    reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]
    predictions = []

    with open(os.path.join(path,'bart_predict.txt'), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)
            predictions.append(str(item))

    # Compute rouge (on test dataset)
    result1 = metric1.compute(predictions=predictions, references=truths)
    result = {key: value for key, value in result1.items()}
 
    # Compute bleu (on test dataset)
    result2 = metric2.compute(predictions=predictions, references=truths)
    result['sacrebleu'] = round(result2["score"], 1)

    # Compute bert score (on test dataset)
    result3 = metric3.compute(predictions=predictions, references=truths, lang="en")
    result['bertscore'] = sum(result3["f1"]) / len(result3["f1"])       
       
    # Compute delta textblob (on test dataset)
    total_textblob = 0
    for original, reframed in zip(texts, predictions):
        total_textblob += (TextBlob(reframed).sentiment.polarity - TextBlob(original).sentiment.polarity)
    result['delta_textblob'] = round(total_textblob / len(texts), 4)

    # Compute average length (on test dataset)
    prediction_lens = [len(pred.split()) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    # Display every evaluation metric (for test dataset)
    for k, v in result.items():
        print(f'{k}: {round(v, 4)}')


def main():
    # Run models
    if args.model =='bart' and args.setting =='unconstrained':
        run_bart_unconstrained()
    elif args.model =='bart' and args.setting =='controlled':
        run_bart_controlled()
    elif args.model =='bart' and args.setting =='predict':
        run_bart_predict()


if __name__=='__main__':
    args = parse_args()
    model = args.model

    if model != 'bart':
        raise Exception("Sorry, this model is currently not included.")

    # Load datasets
    train_path = args.train
    train_dataset = load_dataset('csv', data_files = train_path)
    dev_path = args.dev
    dev_dataset = load_dataset('csv', data_files = train_path)
    test_path = args.test
    test_dataset = load_dataset('csv', data_files = test_path)

    # Set up path for storing prediction result from model
    path = args.output_dir
    main()
