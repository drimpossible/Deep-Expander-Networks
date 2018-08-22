# Deep-Expander-Networks

This repository contains the code for our ECCV '18 paper:

[Deep Expander Networks: Efficient Deep Networks from Graph Theory](https://arxiv.org/pdf/1711.08757.pdf)

[Ameya Prabhu](http://researchweb.iiit.ac.in/~ameya.prabhu)\*, [Girish Varma](https://github.com/geevi)\* and [Anoop Namboodiri](https://faculty.iiit.ac.in/~anoop/)  (\* Authors contributed equally).

### Citation
If you find our work useful in your research, please consider citing:

	@inproceedings{prabhu2018expander,
	  title={Deep Expander Networks: Efficient Deep Networks from Graph Theory},
	  author={Prabhu, Ameya and Varma, Girish and Namboodiri, Anoop},
	  booktitle={ECCV},
	  year={2018}}

## Introduction

Efficient CNN designs like ResNets and DenseNet were proposed to improve accuracy vs efficiency trade-offs. They essentially increased the connectivity, allowing efficient information flow across layers. Inspired by these techniques, we propose to model connections between filters of a CNN using graphs which are simultaneously sparse and well connected. Sparsity results in efficiency while well connectedness can preserve the expressive power of the CNNs. 

We use a well-studied class of graphs from theoretical computer science that satisfies these properties known  as  Expander  graphs.  Expander  graphs  are  used  to  model  connections between filters in CNNs to design networks called X-Nets. 

This repository contains the implementation used for the results in our [paper](https://arxiv.org/pdf/1711.08757.pdf).
 

## Installation and Dependencies

- Anaconda
- Pytorch 0.3.1
- Torchvision

Install PyTorch in a new anaconda environment by the command:
```
conda install pytorch=0.3.1 torchvision -c soumith
```

## Usage

Here is an example to train a X-VGG network on CIFAR10:

```
bash clean.sh;python main.py --dataset='cifar10' --ngpus 1 --data_dir='../data' --workers=2 --epochs=300 --batch-size=128 --nclasses=10 --learningratescheduler='decayschedular' --decayinterval=30 --decaylevel=2 --optimType='sgd' --nesterov --maxlr=0.05 --minlr=0.0005 --weightDecay=5e-4 --model_def='vgg16cifar_bnexpander' --name='example_run' | tee "../logs/example_run_vgg_expander.txt"
```

Another example to train a X-DenseNet-BC with depth 40, growth rate 48 and having an expander graph compress all connections by a factor of 2 (expandSize=2) on CIFAR-100:

```
bash clean.sh; python main.py --dataset='cifar100' --ngpus 1 --data_dir='../data' --workers=2 --epochs=300 --batch-size=64 --nclasses=10 --learningratescheduler='cifarschedular' --optimType='sgd' --nesterov --maxlr=0.1 --minlr=0.0005 --weightDecay=1e-4 --model_def='densenet_cifar' --name='densenetexpander_40_48_2' --expandSize=2 --layers 40 --growth 48 --reduce 0.5 | tee '../logs/densenetexpander_40_48_2.txt'
``` 
An example on Imagenet:

We use the Pytorch dataloader format for the ImageNet dataset. Preprocessing instructions can be found [here](https://github.com/pytorch/examples/tree/master/imagenet).

```
bash clean.sh; python main.py --dataset='imagenet12' --ngpus=1 --data_dir='<PATH TO IMAGENET FOLDER>' --nclasses=1000 --workers=8 --epochs=90 --batch-size=128 --learningratescheduler='imagenetschedular' --decayinterval=30 --decaylevel=10 --optimType='sgd' --verbose --maxlr=0.1 --nesterov --minlr=0.00001 --weightDecay=1e-4 --model_def='resnetexpander34' --expandSize=2 --name='imagenet_resnetexpander34_expandsize2' | tee "../logs/imagenet_resnetexpander34_expandsize2.txt"
```

## TBA

4. Replication Results on CIFAR

5. Replication Results on ImageNet

## Contact

Please do get in touch with us by email for any questions, comments, suggestions you have!
ameya dot pandurang dot prabhu at gmail dot com  
girish dot varma at iiit dot ac dot in

Formatting borrowed from the [DenseNet repository](https://raw.githubusercontent.com/liuzhuang13/DenseNet)
