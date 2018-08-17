import torch
import torch.nn as nn
import torch.optim as optim

def getScheduler(optimizer, args):
    '''Gets the lr schedular '''
    if args.lr_scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.stepsize, gamma=args.stepfactor, last_epoch=-1)
    elif args.lr_scheduler == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steplist, gamma=args.stepfactor)

    return scheduler

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def weights_init(model, opt):
    '''Add your favourite weight initializations.'''
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.affine == True:
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data = nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            # TODO: Check bias
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
