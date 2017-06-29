import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    """
    Penn treeback has no periods in it, i.e. no sentence demarcation, just lines: we assume that lines are different sentences
    Wikipedia has periods as well as <eos>, but unsure what <eos> means (tokenized by wiki2text in torchtext).
    To keep clairity end of sentence here is donoted by <Reos>
    """
        
    def __init__(self, path, to_sentence = False):
        self.dictionary = Dictionary()
        if (to_sentence):
            self.train = self.tokenize_sentence(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize_sentence(os.path.join(path, 'valid.txt'))
            self.test = self.tokenize_sentence(os.path.join(path, 'test.txt'))            
        else:
            self.train = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
            self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file into stream of words"""
        # Penn treeback has no periods in it, i.e. no sentence demarcation, just lines.
        # Wikipedia already has eos, and sentences
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                print(line)
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
                        
        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
            
                
                                
    def tokenize_sentence(self, path):
        """Tokenizes a text file into sentences."""
        print(path)
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                sens = line.split(" .")
                for sentence in sens:
                    words = sentence.split() + ['<Reos>']
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)
                        
        # Tokenize file content
        with open(path, 'r') as f:
            ids = []
            for line in f:
                sens = line.split(" .")
                for sentence in sens:
                    words = sentence.split() + ['<Reos>']
                    sent_ids = torch.LongTensor(
                        [self.dictionary.word2idx[word] for word in words])
                    ids.append(sent_ids)            
        return ids