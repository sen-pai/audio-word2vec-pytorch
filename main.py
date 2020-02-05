import argparse
import torch
import json
import os

from training import train, evaluate
from models.seq2seq import Seq2Seq
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from utils.data_generator import ToyDataset, pad_collate, Toy_Numbers


def run():
    USE_CUDA = torch.cuda.is_available()
    FLAGS.config = 'example_seq2seq.json'
    config_path = os.path.join("experiments", FLAGS.config)
    print(FLAGS.config)

    if not os.path.exists(config_path):
        raise FileNotFoundError

    with open(config_path, "r") as f:
        config = json.load(f)

    config["gpu"] = torch.cuda.is_available()

    writer = SummaryWriter('experiments/finally')

#     dataset = ToyDataset(5, 15)
#     eval_dataset = ToyDataset(5, 15, type='eval')
    dataset = Toy_Numbers(10)
    eval_dataset = Toy_Numbers(10, train = False)
    BATCHSIZE = 32
    train_loader = data.DataLoader(dataset, batch_size=BATCHSIZE, shuffle=False, collate_fn=pad_collate, drop_last=True)
    eval_loader = data.DataLoader(eval_dataset, batch_size=BATCHSIZE, shuffle=False, collate_fn=pad_collate,
                                  drop_last=True)
    config["batch_size"] = BATCHSIZE

    # Models
    model = Seq2Seq(config)
    model = model.float()

    # dataiter = iter(train_loader)
    # sample_input= dataiter.next()

    # writer.add_graph(model, sample_input)
    # writer.close()

    if USE_CUDA:
        model = model.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", .001))

    print("=" * 60)
    print(model)
    print("=" * 60)
    for k, v in sorted(config.items(), key=lambda i: i[0]):
        print(" (" + k + ") : " + str(v))
    print()
    print("=" * 60)

    print("\nInitializing weights...")
    for name, param in model.named_parameters():
        if 'bias' in name:
            torch.nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            torch.nn.init.xavier_normal_(param)

    for epoch in range(FLAGS.epochs):
        run_state = (epoch, FLAGS.epochs, FLAGS.train_size)

        # Train needs to return model and optimizer, otherwise the model keeps restarting from zero at every epoch
        model, optimizer= train(model, optimizer, train_loader, run_state, writer)
        # print("losses", l_list)
        # for i in l_list:
        #     # print(i)
        #     writer.add_scalar('Loss/train',i)
        evaluate(model, eval_loader, writer)


        # TODO implement save models function


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train_size', default=28000, type=int)
    parser.add_argument('--eval_size', default=2600, type=int)
    FLAGS, _ = parser.parse_known_args()
    run()



