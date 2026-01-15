import torch
import torch.utils.data
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from math import ceil


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def log_msg(message, log_file):
    with open(log_file, 'a') as f:
        print(message, file=f)


def get_default_train_val_test_loader(args):
    dsid = args.dataset
    # get dataset
    data_train = torch.load(f'/DET-DyAGCN/data/{dsid}/X_train.pt').squeeze(dim=1)
    data_val = torch.load(f'/DET-DyAGCN/data/{dsid}/X_valid.pt').squeeze(dim=1)
    label_train = torch.load(f'/DET-DyAGCN/data/{dsid}/y_train.pt')
    label_val = torch.load(f'/DET-DyAGCN/data/{dsid}/y_valid.pt')
    # fill the tail to allow for integer segmentation
    if data_train.size(-1) % args.slots:
        pad_size = (args.slots - data_train.size(-1) % args.slots) / 2
        data_train = F.pad(data_train, (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
        data_val = F.pad(data_val, (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
        # the subsequences after slicing are of odd length.
        if (data_train.size(-1) // args.slots) % 2:
            pad_size = args.slots
            data_train = F.pad(data_train, (0, pad_size), mode='constant', value=0.0)
            data_val = F.pad(data_val, (0, pad_size), mode='constant', value=0.0)
    # init [num_variables, seq_length, num_classes]
    if args.batch_size >= data_train.size(0):
        real_batch_size = data_train.size(0)
        drop_last = False
    else:
        real_batch_size = args.batch_size
        drop_last = True
    num_nodes = data_val.size(-2)
    seq_length = data_val.size(-1)
    num_classes = len(torch.bincount(label_val.type(torch.int)))
    # convert to TensorDataset
    train_dataset = TensorDataset(data_train, label_train)
    val_dataset = TensorDataset(data_val, label_val)
    # data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=drop_last)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             drop_last=drop_last)
    return train_loader, val_loader, num_nodes, seq_length, num_classes, real_batch_size