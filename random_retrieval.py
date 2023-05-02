import pandas as pd
import evaluate
from textblob import TextBlob
import argparse


def run_random():
    training_data = pd.read_csv(train_path)
    testing_data = pd.read_csv(test_path)

    # In this model, we randomly extract reframed sentences from the
    # Training data and map it with the testing data
    randomly_retrieved_data = training_data.sample(n=len(testing_data), random_state=random_seed)

    # Rouge Score
    rouge = evaluate.load('rouge')
    rouge_score = rouge.compute(predictions=randomly_retrieved_data['reframed_text'].to_list(), references=testing_data['original_text'].to_list())
    print(rouge_score)

    # BLEU score
    bleu = evaluate.load('bleu')
    bleu_score = bleu.compute(predictions=randomly_retrieved_data['reframed_text'].to_list(), references=testing_data['original_text'].to_list())
    print(bleu_score)

    # BERTScore
    bscore = evaluate.load("bertscore")
    bscore_results = bscore.compute(predictions=randomly_retrieved_data['reframed_text'].to_list(), references=testing_data['original_text'].to_list(), lang="en")
    print("BScore:", sum(bscore_results['f1'])/len(bscore_results['f1']))

    # Change in TB and Avg Len
    total_change_tb = 0
    total_len = 0
    for i in range(len(randomly_retrieved_data['reframed_text'])):
        total_change_tb += TextBlob(randomly_retrieved_data['reframed_text'].to_list()[i]).sentiment.polarity - TextBlob(testing_data['original_text'].to_list()[i]).sentiment.polarity 
        total_len += len(randomly_retrieved_data['reframed_text'].to_list()[i].split())
    print("\u0394TB:",total_change_tb/len(randomly_retrieved_data['reframed_text']))
    print("Avg Len:", total_len/len(randomly_retrieved_data['reframed_text']))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='random', choices=['random'])
    parser.add_argument('--train', default='data/wholetrain.csv')
    parser.add_argument('--dev', default='data/wholedev.csv')
    parser.add_argument('--test', default='data/wholetest.csv')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    if args.model != 'random':
        raise Exception("Sorry, this model is currently not included.")

    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    output_path = args.output_dir
    random_seed = 7
    run_random()