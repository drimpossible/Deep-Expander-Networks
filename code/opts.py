import argparse

dset_choices = ['cifar10','cifar100','tuberlin','imagenet12']
gtype_choices = ['full','groups','shuffle','expander','expandershuffle','expanderacc']
reporttype_choices = ['acc']
criterion_choices = ['crossentropy']
optim_choices = ['sgd','adam']
model_def_choices = ['vgg16_bn','vgg16expander_bn','alexnet','alexnetexpander','densenet_cifar','densenetgrouped_cifar','densenetexpander_cifar','densenetexpander2_cifar','cifarresnext','densenet121','densenetexpander121','resnet34','resnetexpander34','densenet161','densenetexpander161','resnet50','resnetexpander50','densenet201','densenetexpander201','resnet101','resnetexpander101','densenet169','densenetexpander169','resnet152','resnetexpander152','vgg16cifar_bn','vgg16cifar_bnexpander','mobilenet','mobilenet_cifar']

def myargparser():
    parser = argparse.ArgumentParser(description='PyTorch Core Training')

    #data stuff
    parser.add_argument('--dataset', choices=dset_choices, help='chosen dataset'+'Options:'+str(dset_choices))
    parser.add_argument('--data_dir', required=True, help='Dataset directory')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (Default: 4)')
    parser.add_argument('--weight_init', action='store_true', help='Turns on weight inits')
    #other default stuff
    parser.add_argument('--epochs', required=True, type=int,help='number of total epochs to run')
    parser.add_argument('--expandConfig',  type=int, help='Configuration')
    parser.add_argument('--batch-size', required=True, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--nclasses', type=int, help='number of classes', default=0)
    parser.add_argument('--tenCrop', action='store_true', help='ten-crop testing')
    parser.add_argument('--printfreq', default=200, type=int, help='print frequency (default: 10)')
    parser.add_argument('--learningratescheduler', default='decayschedular', help='print frequency (default: 10)')
    parser.add_argument('--inpsize', default=224,  type=int, help='Input Image Size')
    parser.add_argument('--widthmult', default=1.0,  type=float, help='Multiply width by')
    parser.add_argument('--grouptype', default="full",  type=str, choices=gtype_choices, help='Input Image Size')
    parser.add_argument('--sp', default=1,  type=int, help='Input Image Size')
    parser.add_argument('--exp', default=1,  type=int, help='Input Image Size')


    #optimizer/criterion stuff
    parser.add_argument('--decayinterval', type=int, help='decays by a power of decay_var in these epochs')
    parser.add_argument('--decaylevel', type=float, help='decays by a power of decaylevel')
    parser.add_argument('--criterion', default="crossentropy", choices=criterion_choices, type=str, help='Criterion. Options:'+str(criterion_choices))
    parser.add_argument('--optimType', required=True, choices=optim_choices, type=str, help='Optimizers. Options:'+str(optim_choices))

    parser.add_argument('--maxlr', required=True, type=float, help='initial learning rate')
    parser.add_argument('--lr', type=float, help='initial learning rate')
    parser.add_argument('--minlr', required=True, type=float, help='initial learning rate')

    parser.add_argument('--nesterov', action='store_true', help='nesterov momentum')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (Default: 0.9)')
    parser.add_argument('--weightDecay', default=0, type=float, help='weight decay (Default: 1e-4)')

    #extra model stuff
    parser.add_argument('--model_def', required=True, choices=model_def_choices, help='Architectures to be loaded. Options:'+str(model_def_choices))
    parser.add_argument('--name', required=True, type=str, help='name of experiment')
    #default
    parser.add_argument('--cachemode', default=True, help='if cachemode')
    parser.add_argument('--cuda', default=True, help='If cuda is available')
    parser.add_argument('--manualSeed', type=int, default=123, help='fixed seed for experiments')
    parser.add_argument('--ngpus', type=int, required=True, help='no. of gpus')
    parser.add_argument('--logdir', type=str, default='../logs', help='log directory')
    parser.add_argument('--tensorboard',help='Log progress to TensorBoard', default=False)
    parser.add_argument('--testOnly',  action='store_true', help='run on validation set only')
    parser.add_argument('--acc_type', default = "class")

    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--pretrained_file', default='')


    #Densenet
    parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
    parser.add_argument('--expandSize', default=2, type=int,
                    help='factor to compress by')
    parser.add_argument('--growth', default=48, type=int,
                    help='number of new channels per layer (default: 12)')
    parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
    parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
    parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
    parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    parser.set_defaults(bottleneck=True)
    parser.set_defaults(augment=True)

    parser.add_argument('--from_modelzoo', action='store_true')
    #model stuff
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--store', default='', type=str, metavar='PATH',
                        help='path to storing checkpoints (default: none)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')


    return parser
