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
	  year={2018}
	}

## Contents
1. Introduction
2. Installation and Dependencies

## TBA Shortly

2. Usage (Refer run_cifar.sh. Complete  instructions to be updated)
3. Replication Results on CIFAR
4. Replication Results on ImageNet

## Introduction
Efficient CNN designs like ResNets and DenseNet were proposed to improve accuracy vs efficiency trade-offs. They essentially increased the connectivity, allowing efficient information flow across layers. Inspired by these techniques, we propose to model connections between filters of a CNN using graphs which are simultaneously sparse and well connected. Sparsity results in efficiency while well connectedness can preserve the expressive power of the CNNs. 

We use a well-studied class of graphs from theoretical computer science that satisfies these properties known  as  Expander  graphs.  Expander  graphs  are  used  to  model  connections between filters in CNNs to design networks called X-Nets. 

This repository contains the implementation used for the results in our [paper](https://arxiv.org/pdf/1711.08757.pdf).
 
## Installation and Dependencies

- Pytorch 0.3
