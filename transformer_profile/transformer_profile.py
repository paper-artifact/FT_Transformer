from gpt2 import gpt2_profile
from bert_base import bert_base_profile
from bert_large import bert_large_profile
from t5_small import t5_small_profile
import argparse
import random
import nltk
from nltk.corpus import brown

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer Profiler')
    parser.add_argument('--model', default='gpt2', choices=['gpt2', 'bert-base', 'bert-large', 't5-small'], help="choose the model for profiling")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=10)

    return parser.parse_args()

def generate_random_sentence(batch_size=1, seq_len=10, corpus='brown'):
    try:
        nltk.data.find(f'corpora/{corpus}')
    except LookupError:
        nltk.download(corpus)
    
    print("### generating input tokens ###")
    sentences = []
    punctuation = ['.', ',', '!', '?', ';', ':']  

    for _ in range(batch_size):
        words_list = [random.choice(brown.words()) for _ in range(seq_len)]

        num_punctuations = max(1, seq_len // 8) 
        insert_positions = random.sample(range(1, len(words_list)), num_punctuations)
        for pos in sorted(insert_positions, reverse=True):  
            words_list.insert(pos, random.choice(punctuation))

        words_list[0] = words_list[0].capitalize()
        words_list.append(random.choice(['.', '!', '?']))
        sentence = ' '.join(words_list)
        sentences.append(sentence)
    
    print("### generation completed ###")

    return sentences

if __name__ == "__main__":
    args = parse_args()
    model = args.model
    batch_size = args.batch_size
    seq_len = args.seq_len

    test_input = generate_random_sentence(batch_size, seq_len)
  
    if model == 'gpt2':
        gpt2_profile(test_input)
    elif model == 'bert-base':
        bert_base_profile(test_input)
    elif model == 'bert-large':
        bert_large_profile(test_input)
    elif model == 't5-small':
        t5_small_profile(test_input)