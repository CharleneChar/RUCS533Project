import pandas as pd
import evaluate
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
import torch
import os
import argparse
from tqdm import tqdm


def run_sbert():
    training_data = pd.read_csv(train_path)
    testing_data = pd.read_csv(test_path)

    # This model uses cosine similarity to retrieve rephrased text from the training data based
    # on the original text
    original_training_text = training_data['original_text'].to_list()
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Get the training embeddings using the sbert model
    train_embeddings = []
    print("Retrieving training embeddings")
    for i in tqdm(range(len(original_training_text))):
        train_embeddings.append(sbert_model.encode([original_training_text[i]], convert_to_tensor=True).cuda())

    # Get the testing and training data
    output_reframed_text = []
    original_testing_text = testing_data['original_text'].to_list()
    reframed_train_text = training_data['reframed_text'].to_list()

    # Use cosine similairty to select the reframed text
    print("Retrieving sentences")
    for original in tqdm(original_testing_text):
        original_text_embeddings = sbert_model.encode([original], convert_to_tensor=True).cuda()
        max_value, max_index = 0, 0
        for i in range(len(train_embeddings)):
            val = util.pytorch_cos_sim(original_text_embeddings, train_embeddings[i])
            if(val > max_value):
                max_value = val
                max_index = i
        output_reframed_text.append(reframed_train_text[max_index])
    
    # Write reframed text
    with open(os.path.join('output/','sbert.txt'), 'w') as f:
        for item in output_reframed_text:
            f.write("%s\n" % item)

    # Rouge Score
    rouge = evaluate.load('rouge')
    rouge_score = rouge.compute(predictions=output_reframed_text, references=testing_data['original_text'].to_list())
    print(rouge_score)

    # BLEU score
    bleu = evaluate.load('bleu')
    bleu_score = bleu.compute(predictions=output_reframed_text, references=testing_data['original_text'].to_list())
    print(bleu_score)

    # BERTScore
    bscore = evaluate.load("bertscore")
    bscore_results = bscore.compute(predictions=output_reframed_text, references=testing_data['original_text'].to_list(), lang="en")
    print("BScore:", sum(bscore_results['f1'])/len(bscore_results['f1']))

    # Change in TB and Avg Len
    total_change_tb = 0
    total_len = 0
    for i in range(len(output_reframed_text)):
        total_change_tb += TextBlob(output_reframed_text[i]).sentiment.polarity - TextBlob(testing_data['original_text'].to_list()[i]).sentiment.polarity 
        total_len += len(output_reframed_text[i].split())
    print("\u0394TB:",total_change_tb/len(output_reframed_text))
    print("Avg Len:", total_len/len(output_reframed_text))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='sbert', choices=['sbert'])
    parser.add_argument('--train', default='data/wholetrain.csv')
    parser.add_argument('--dev', default='data/wholedev.csv')
    parser.add_argument('--test', default='data/wholetest.csv')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    if args.model != 'sbert':
        raise Exception("Sorry, this model is currently not included.")

    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    output_path = args.output_dir
    run_sbert()