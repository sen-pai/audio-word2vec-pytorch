import numpy as np
import torch

from torch.utils import data
from random import choice, randrange
from itertools import zip_longest


def batch(iterable, n=1):
    args = [iter(iterable)] * n
    return zip_longest(*args)


def pad_tensor(vec, pad, value=0, dim=0):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = pad - vec.shape[0]

    if len(vec.shape) == 2:
        zeros = torch.ones((pad_size, vec.shape[-1])) * value
    elif len(vec.shape) == 1:
        zeros = torch.ones((pad_size,)) * value
    else:
        raise NotImplementedError
    return torch.cat([torch.Tensor(vec), zeros], dim=dim)


def pad_collate(batch, values=(0, 0), dim=0):
    """
    args:
        batch - list of (tensor, label)

    reutrn:
        xs - a tensor of all examples in 'batch' after padding
        ys - a LongTensor of all labels in batch
        ws - a tensor of sequence lengths
    """

    sequence_lengths = torch.Tensor([int(x[0].shape[dim]) for x in batch])
    sequence_lengths, xids = sequence_lengths.sort(descending=True)
    target_lengths = torch.Tensor([int(x[1].shape[dim]) for x in batch])
    target_lengths, yids = target_lengths.sort(descending=True)
    # find longest sequence
    src_max_len = max(map(lambda x: x[0].shape[dim], batch))
    tgt_max_len = max(map(lambda x: x[1].shape[dim], batch))
    # pad according to max_len
    batch = [(pad_tensor(x, pad=src_max_len, dim=dim), pad_tensor(y, pad=tgt_max_len, dim=dim)) for (x, y) in batch]

    # stack all
    xs = torch.stack([x[0] for x in batch], dim=0)
    ys = torch.stack([x[1] for x in batch]).int()
    xs = xs[xids]
    ys = ys[yids]
    return xs, ys, sequence_lengths.int(), target_lengths.int()

import numpy as np
import torch

from torch.utils import data


#use to check reconstructing autoencoders 
#output of decoder should be same as input to encoder

class Toy_Numbers(data.Dataset):

    def __init__(self, max_len, train = True):
        self.basic_lenght = 3
        self.max_len = max_len - 3
        self.eos = -1

        if train:
            self.set = [self.create_samples() for _ in range(3000)]
        else:
            self.set = [self.create_samples() for _ in range(300)]

    def __len__(self):
        return len(self.set)

    def __getitem__(self, item):
        # print("this is item")
        # print(self.set[item])
        return self.set[item]

    def create_samples(self):

        #create a random lenght sample
        sample_lenght = np.random.randint(self.max_len) + self.basic_lenght

        #sample would end at 100 atleast
        start_number = np.random.randint(100 - self.max_len -1) + 2
        end_number = start_number + sample_lenght
        sample = np.arange(start_number, end_number, dtype = np.int64)

        #append eos to know when to stop
        sample = np.append(sample, self.eos)

        sample = sample.reshape(-1,1)
        #return sample twice as it is input and output
        # print("this is sample", sample)
        return sample , sample



class ToyDataset(data.Dataset):
    """
    https://talbaumel.github.io/blog/attention/
    """
    def __init__(self, min_length=5, max_length=20, type='train'):
        self.SOS = "<s>"  # all strings will end with the End Of String token
        self.EOS = "</s>"  # all strings will end with the End Of String token
        self.characters = list("abcd")
        self.int2char = list(self.characters)
        self.char2int = {c: i+3 for i, c in enumerate(self.characters)}
        print(self.char2int)
        self.VOCAB_SIZE = len(self.characters)
        self.min_length = min_length
        self.max_length = max_length
        if type == 'train':
            self.set = [self._sample() for _ in range(3000)]
        else:
            self.set = [self._sample() for _ in range(300)]

    def __len__(self):
        return len(self.set)

    def __getitem__(self, item):
        return self.set[item]

    def _sample(self):
        random_length = randrange(self.min_length, self.max_length)  # Pick a random length
        random_char_list = [choice(self.characters[:-1]) for _ in range(random_length)]  # Pick random chars
        random_string = ''.join(random_char_list)
        print("This is random_string", random_string)
        a = np.array([self.char2int.get(x) for x in random_string])
        print("This is a", a)
        b = np.array([self.char2int.get(x) for x in random_string[::-1]] + [2]) # Return the random string and its reverse
        x = np.zeros((random_length, self.VOCAB_SIZE))

        x[np.arange(random_length), a-3] = 1
        print("This is x", x)
        print("This is b", b)
        fffff = self.int2char(b)
        print("This is ffff", fffff)

        return x, b


