from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments, AutoModelForCausalLM, pipeline, AutoTokenizer
import csv
import os
import re
import pandas as pd
import evaluate
from textblob import TextBlob
import argparse
from tqdm import tqdm

def run_gpt():
    # Use the gpt tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
    # Collect dataset
    training_dataset = TextDataset(tokenizer=tokenizer, file_path=train_path, block_size=50)
    testing_dataset = TextDataset(tokenizer=tokenizer, file_path=test_path, block_size=50)
    # Create Data colator for the batches
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Import the GPT model
    gpt_model = AutoModelForCausalLM.from_pretrained("openai-gpt")

    # Define the training arguments 
    training_arguments = TrainingArguments(
        output_dir='./gpt',
        overwrite_output_dir=True,
        num_train_epochs = 5,
        per_device_train_batch_size = 6,
        per_device_eval_batch_size = 6,
        eval_steps = 200,
        save_steps = 400,
        warmup_steps = 300,
        prediction_loss_only = True,
        learning_rate = 3e-5,
        optim = 'adamw_torch'
    )

    # Define the trainer 
    trainer = Trainer(
        model = gpt_model,
        args = training_arguments,
        data_collator = data_collator,
        train_dataset = training_dataset,
        eval_dataset = testing_dataset
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model("text_reframe_gpt")

    # Import the trained model
    gpt_model = AutoModelForCausalLM.from_pretrained('text_reframe_gpt')
    # Create a pipeline for tect generation
    reframed_pipeline = pipeline('text-generation', model = gpt_model, tokenizer='openai-gpt')

    # Test the model 
    reframed_text_test = []
    with open(test_path, newline='') as data:
        annotations = csv.DictReader(data, delimiter=',', quotechar='"')
        annotations_list = list(annotations)
        print("Generating Text")
        for i in tqdm(range(len(annotations_list))):
            prefix = annotations_list[i]['original_text'] + "\nreframed:"
            gen_text = reframed_pipeline(prefix, max_length=len(prefix.split()) + 30)[0]['generated_text']
            temp_reframe = re.findall(r'reframed:(.*)', gen_text)
            temp_reframe = re.sub('<(.*)?>?', '', temp_reframe[0])
            reframed_text_test.append(temp_reframe)
    
    # Write the results into the output
    with open(os.path.join(output_path,'gpt.txt'), 'w') as f:
        for item in reframed_text_test:
            f.write("%s\n" % item)

    # Read the testing data
    testing_data = pd.read_csv(test_path)
    original_testing_text = testing_data['original_text'].to_list()

    # Evaluation 
    # ROUGE
    rouge = evaluate.load('rouge')
    rouge_score = rouge.compute(predictions=reframed_text_test, references=testing_data['original_text'].to_list())
    print(rouge_score)

    # BLEU score
    bleu = evaluate.load('bleu')
    bleu_score = bleu.compute(predictions=reframed_text_test, references=testing_data['original_text'].to_list())
    print(bleu_score)

    # BERTScore
    bscore = evaluate.load("bertscore")
    bscore_results = bscore.compute(predictions=reframed_text_test, references=testing_data['original_text'].to_list(), lang="en")
    print("BScore:", sum(bscore_results['f1'])/len(bscore_results['f1']))

    # Change in TB and Avg Len
    total_change_tb = 0
    total_len = 0
    for i in range(len(reframed_text_test)):
        total_change_tb += TextBlob(reframed_text_test[i]).sentiment.polarity - TextBlob(testing_data['original_text'].to_list()[i]).sentiment.polarity 
        total_len += len(reframed_text_test[i].split())
    print("\u0394TB:",total_change_tb/len(reframed_text_test))
    print("Avg Len:", total_len/len(reframed_text_test))
    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='gpt', choices=['gpt'])
    parser.add_argument('-s', '--setting', default='unconstrained', choices=['unconstrained'])
    parser.add_argument('--train', default='data/wholetrain_gpt.txt')
    parser.add_argument('--dev', default='data/wholedev.csv')
    parser.add_argument('--test', default='data/wholetest.csv')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    if args.model != 'gpt':
        raise Exception("Sorry, this model is currently not included.")

    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    output_path = args.output_dir
    run_gpt()
    