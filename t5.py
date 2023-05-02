import argparse
import pandas as pd
import numpy as np
import random
from sentence_transformers import SentenceTransformer, util
import os
from transformers import Trainer, pipeline, set_seed, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
nltk.download('punkt')
import csv
from datasets import load_dataset, load_metric


def preprocess_function(examples, tokenizer, prefix, type, type_label):
        inputs = [prefix + doc for doc in examples[type]]
        model_inputs = tokenizer(inputs) # max_length=max_input_length, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples[type_label]) #, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def compute_metric_with_extra(tokenizer, metric, metric2, metric3):
    tokenizer = tokenizer
    metric = metric
    metric2 = metric2
    metric3 = metric3
    def compute_metric(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        decoded_labels_expanded = [[x] for x in decoded_labels]
        result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)

        # print(result2)
        result['sacrebleu'] = round(result2["score"], 1)
        

        result3 = metric3.compute(predictions=decoded_preds, references=decoded_labels_expanded,  lang="en")

        result['bertscore'] = sum(result3["f1"])/len(result3["f1"])

        return {k: round(v, 6) for k, v in result.items()}
    return compute_metric


def run_t5_unconstrained(): 
    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")
    metric3 = load_metric('bertscore')
    model_checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    #print(tokenizer("Hello, this one sentence!")) #a test
    prefix = "summarize: "

    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "prefix": prefix, "type": "original_text", "type_label":"reframed_text"})
   
    tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "prefix": prefix, "type": "original_text", "type_label": "reframed_text"})
  
    tokenized_test_datasets = test_dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "prefix": prefix, "type": "original_text", "type_label": "reframed_text"})
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    batch_size = 6
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
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    compute_metric = compute_metric_with_extra(tokenizer, metric, metric2, metric3)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_test_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metric
    )
    trainer.train()
    trainer.evaluate()
    # save model
    trainer.save_model("output/reframer") #TODO

    #prediction
    model = AutoModelForSeq2SeqLM.from_pretrained("output/reframer")
    tokenizer = AutoTokenizer.from_pretrained("output/reframer")
    reframer = pipeline('summarization', model=model, tokenizer=tokenizer)

    test = pd.read_csv(test_path)
    texts = test['original_text'].to_list()
    reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]

    with open(os.path.join(path,'t5_unconstrained.txt'), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)



def run_t5_controlled(): 
    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")
    metric3 = load_metric('bertscore')
    model_checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    prefix = "summarize: "

    
    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "prefix": prefix, "type": "original_with_label", "type_label":"reframed_text"})
   
    tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "prefix": prefix, "type": "original_with_label", "type_label":"reframed_text"})
  
    tokenized_test_datasets = test_dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "prefix": prefix, "type": "original_with_label", "type_label":"reframed_text"})
    

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    batch_size = 6
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
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    compute_metric = compute_metric_with_extra(tokenizer, metric, metric2, metric3)

    
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_test_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metric
    )
    trainer.train()
    trainer.evaluate()
    # save model
    trainer.save_model("output/reframer") #TODO

    #prediction
    model = AutoModelForSeq2SeqLM.from_pretrained("output/reframer")
    tokenizer = AutoTokenizer.from_pretrained("output/reframer")
    reframer = pipeline('summarization', model=model, tokenizer=tokenizer)

    test = pd.read_csv(test_path)
    texts = test['original_with_label'].to_list()
    reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]

    with open(os.path.join(path,'t5_controlled.txt'), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)


def run_t5_predict(): 
    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")
    metric3 = load_metric('bertscore')
    model_checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    #print(tokenizer("Hello, this one sentence!")) #a test
    prefix = "summarize: "

    tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "prefix": prefix, "type": "original_text", "type_label":"strategy_reframe"})
   
    tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "prefix": prefix, "type": "original_text", "type_label": "strategy_reframe"})
  
    tokenized_test_datasets = test_dataset.map(preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer, "prefix": prefix, "type": "original_text", "type_label": "strategy_reframe"})
    

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    batch_size = 6
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
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    compute_metric = compute_metric_with_extra(tokenizer, metric, metric2, metric3)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train_datasets["train"],
        eval_dataset=tokenized_test_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metric
    )

    trainer.train()
    trainer.evaluate()
    # save model
    trainer.save_model("output/reframer") #TODO

    #prediction
    model = AutoModelForSeq2SeqLM.from_pretrained("output/reframer")
    tokenizer = AutoTokenizer.from_pretrained("output/reframer")
    reframer = pipeline('summarization', model=model, tokenizer=tokenizer)

    test = pd.read_csv(test_path)
    texts = test['original_text'].to_list()
    reframed_phrases = [reframer(phrase)[0]['summary_text'] for phrase in texts]

    with open(os.path.join(path,'t5_predict.txt'), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)




def main():
    #run models
    if args['setting']=='unconstrained':
        run_t5_unconstrained()
    elif  args['setting']=='controlled':
        run_t5_controlled()
    elif args['setting']=='predict':
        run_t5_predict()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--setting', default='unconstrained', choices=['unconstrained', 'controlled', 'predict'])
    parser.add_argument('--train', default='data/wholetrain.csv') #default is for bart/t5; data format will be different for GPT
    parser.add_argument('--dev', default='data/wholedev.csv')
    parser.add_argument('--test', default='data/wholetest.csv')
    parser.add_argument('--output_dir', type=str, default='output/')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    train_path = args['train']
    train_dataset = load_dataset('csv', data_files=train_path)
    dev_path = args['dev']
    dev_dataset = load_dataset('csv', data_files=train_path)
    test_path = args['test']
    test_dataset = load_dataset('csv', data_files=test_path)


    path = args['output_dir']
    main()