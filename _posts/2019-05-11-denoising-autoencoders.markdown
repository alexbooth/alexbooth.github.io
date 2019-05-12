---
layout: post
title:  "Denoising Autoencoders"
date:   2019-05-11 11:41:31 -0700
categories: jekyll update
author: Alexander Booth
---

<p align="center"><img src="{{ '/assets/images/DAE/autoencoder.png' | relative_url }}" width="54%"></p>

<!--Autoencoders take a high dimensional input which is passed through a bottleneck in an attempt to obtain a meaningful latent representation of the data.
The latent representation is then passed into the decoder, which tries to reconstruct the original input.
A useful encoder has input $$x \in \mathbb{R}^d $$ and output $$ z_{latent} \in \mathbb{R}^k$$ where $$d \gg k$$. -->

The denoising autoencoder (DAE) is a natural extension of the traditional autoencoder. 
In fact, an autoencoder can be made into a DAE with a simple modification of the loss function.

Where $$f_{\theta}(x)$$ is the autoencoder parameterized by $$\theta$$.
The usual autoencoder loss function is

$$\boldsymbol{\mathcal{L}}(x)=\frac{1}{n}\sum_{i=1}^{n}(x-f_{\theta}(x))^2$$

while the DAE loss is

$$\boldsymbol{\mathcal{L}}(x, \hat{x})=\frac{1}{n}\sum_{i=1}^{n}(x-f_{\theta}(\hat{x}))^2\text{, where } C(x)=\hat{x}$$

In the case of the DAE the input is assumed to be corrupted by some stochastic process $$C(\cdot)$$. 
While $$C$$ is arbitrary, this post will consider Gaussian noise:

$$C(x) = x + \epsilon \text{, where } \epsilon \sim \mathcal{N}(0, \, I)$$

<p align="center"><img src="{{ '/assets/images/DAE/corrupt.gif' | relative_url }}"></p>

Aside from the obvious denoising capabilities of the DAE, training using the DAE loss offers various other benefits. 
Random noise forces the encoder to learn a robust mapping from input to the latent space.
The DAE loss acts as a regularizer since even an overparameterized model cannot simply learn a 1-to-1 mapping of input to output.

# Demo
{% include dae-demo-1.html %}
Try it out! Traverse the latent space of a DAE with the sliders and see what the decoder outputs if the encoder had output these two variables.
The inputs and outputs to this DAE were 32x32 so these results are pretty good for a 512x reduction in dimensionality.

# Source code 
* [DAE in Tensorflow 2.0](https://github.com/alexbooth/DAE-Tensorflow-2.0)  
* [This blog](https://github.com/alexbooth/alexbooth.github.io)


# References
[1] MNIST dataset <http://yann.lecun.com/exdb/mnist/>  
[2] Pascal Vincent, et al. [Extracting and Composing Robust Features with Denoising Autoencoders](http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf) ICML, 2008.  
[3] “14.5 Denoising Autoencoders.” Deep Learning, by Ian Goodfellow, Yoshua Bengio, and Aaron Courville, The MIT Press, 2017, pp. 501–502.
