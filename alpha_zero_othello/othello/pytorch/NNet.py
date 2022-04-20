import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet

import torch
import torch.optim as optim
from .OthelloNNet import OthelloNNet as onnet


args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 20,
    'batch_size': 256,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


def loss_pi(targets, outputs):
    return -torch.sum(targets * outputs) / targets.size()[0]


def loss_v(targets, outputs):
    return torch.nn.functional.mse_loss(torch.squeeze(outputs, 1), targets)


class ExamplesDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, sample_ids):
        boards, pis, vs = list(zip(*[self.examples[i] for i in sample_ids]))
        boards = torch.from_numpy(np.array(boards).astype(np.float32))
        target_pis = torch.from_numpy(np.array(pis).astype(np.float32))
        target_vs = torch.from_numpy(np.array(vs).astype(np.float32))
        return boards, target_pis, target_vs

    def __len__(self):
        return len(self.examples)


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = torch.jit.script(onnet(game, args))
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if args.cuda:
            self.nnet = self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        self.nnet.train()
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            ds = ExamplesDataset(examples)
            sampler = torch.utils.data.RandomSampler(ds, True)
            sampler = torch.utils.data.BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
            dl = torch.utils.data.DataLoader(ds, batch_size=None, sampler=sampler, pin_memory=True)
            t = tqdm(dl, desc='Training Net', disable=False)

            with torch.jit.fuser('fuser2'):
                for boards, target_pis, target_vs in t:
                    if args.cuda:
                        boards, target_pis, target_vs = boards.contiguous().cuda(non_blocking=True), target_pis.contiguous().cuda(non_blocking=True), target_vs.contiguous().cuda(non_blocking=True)

                    # compute output
                    out_pi, out_v = self.nnet(boards)
                    l_pi = loss_pi(target_pis, out_pi)
                    l_v = loss_v(target_vs, out_v)
                    total_loss = l_pi + l_v
                                            
                    pi_losses.update(l_pi.item(), boards.size(0))
                    v_losses.update(l_v.item(), boards.size(0))
                    t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                    # compute gradient and do SGD step
                    total_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(True)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.from_numpy(board.astype(np.float32))
        if args.cuda: board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.inference_mode():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
