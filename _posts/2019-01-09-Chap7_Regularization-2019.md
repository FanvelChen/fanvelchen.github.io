---
layout:     post
title:      Chap7 Regularization
date:       2019-01-09
author:     Fanvel Chen
header-img: img/posts/chap6_img.jpg
catalog:    true
mathjax:    true
tags:
    - deep learning
---

# Chap7 Regularization for Deep Learning

In practice, an overly complex model family does not necessarily include the target function or the true data-generating process, or even a close approximation of either.

## 7.1 Parameter Norm Penalties

we penalizes only the weights of the aﬃne transformation at each layer and leaves the biases unregularized.

### 7.1.1 $L^2$ Parameter Regularization

$$
\Omega (\boldsymbol{\theta}) = \frac{1}{2} \left| \left| \boldsymbol{w} \right| \right|_{2}^{2}
$$

the update is

$$
\boldsymbol{w} \gets (1-\epsilon \alpha)\boldsymbol{w} - \epsilon \nabla_{\boldsymbol{w}} J(\boldsymbol{w};\boldsymbol{X},\boldsymbol{y})
$$

It shrink the weight vector by a constant factor on each step.

### 7.1.2 $L^1$ Parameter Regularization

$$
\Omega (\boldsymbol{\theta}) = \frac{1}{2} \left| \left| \boldsymbol{w} \right| \right|_{1} = \sum_i \left| w_i  \right|
$$

more sparse than $L^2$

## 7.2 Norm Penalties as Constrained Optimization

constrain $\Omega (\boldsymbol{\theta}) < k$:

$$
\mathcal{L}(\boldsymbol{\theta},\alpha; \boldsymbol{X},\boldsymbol{y}) = J(\boldsymbol{\theta}; \boldsymbol{X},\boldsymbol{y}) + \alpha (\Omega (\boldsymbol{\theta}) - k)
$$

solution given by:

$$
\boldsymbol{\theta^{*}} = \arg \min_{\boldsymbol{\theta}} \max_{\alpha, \alpha \geq 0} \mathcal{L}(\boldsymbol{\theta},\alpha)
$$

Sometimes we may wish to use explicit constraints rather than penalties.

we can modify algorithms such as stochastic gradientdescent to take a step downhill on $J(\boldsymbol{\theta})$ and then project $\boldsymbol{\theta}$ back to the nearestpoint that satisﬁes $\Omega (\boldsymbol{\theta})< k$.

reason:

1. penalties can cause nonconvex optimizationprocedures to get stuck in local minima corresponding to small $\boldsymbol{\theta}$
2. explicit constraints with reprojection can be useful because they impose some stability on the optimization procedure.

Another strategy : constraining the norm of each column of the weight matrixof a neural net layer, rather than constraining the Frobenius norm of the entireweight matrix.

## 7.3 Regularization and Under-Constrained Problems

We can thus interpret the pseudoinverse as stabilizing underdetermined problems using regularization.

## 7.4 Dataset Augmentation

Injecting noise in the input to a neural network can also be seen as a form of data augmentation.

## 7.5 Noise Robustness

noise can be added to the input as a dataset augmentation technique or to the weights interpreted as a stochastic implementation of Bayesian inference over weights.

For small $\eta$, the minimization of $J$ with added weight noise (with covariance $\eta \boldsymbol{I}$) is equivalent to minimization of $J$ with an additional regularization term. This form of regularization encourages the parameters to go to regions of parameter space where small perturbations of the weights have a relatively small inﬂuence on the output.

### 7.5.1 Injecting Noise at the Output Targets

**label smoothing** replacing the hard 0 and 1 classiﬁcation targets with targets of $\frac{\epsilon}{k-1}$ and $1-\epsilon$, respectively.

## 7.6 Semi-Supervised Learning

## 7.7 Multitask Learning

## 7.8 Early Stopping

the number of training steps is just another hyperparameter.

After the initial training with early stopping has completed, we can use the validation set to retrain.

why regularizer? early stopping has the eﬀect of restricting the optimization procedure to a relatively small volume of parameter space in the neighborhood of the initial parameter value

## 7.9 Parameter Tying and Parameter Sharing

for similar task, we believe the model parameters should be close:

$$
\Omega (\boldsymbol{w}^{(A)},\boldsymbol{w}^{(B)}) = \left| \left| \boldsymbol{w}^{(A)} - \boldsymbol{w}^{(B)}  \right| \right|_{2}^{2}
$$

**parameter sharing**

### 7.9.1 Convolutional Neural Networks

## 7.10 Sparse Representations

$\Omega (\boldsymbol{h})$

## 7.11 Bagging and Other Ensemble Methods

if we have $k$ regression models, with error $\epsilon_i$ on each example, and draw from $\mathbb{E} [\epsilon_i^2] = v, \mathbb{E} [\epsilon_i \epsilon_j] = c$

$$
\begin{align}
\begin{split}
\mathbb{E} \left[ \left( \frac{1}{k} \sum_i \epsilon_i  \right)^2 \right]  &= \frac{1}{k^2} \mathbb{E} \left[  sum_i \left( \epsilon_i^2 + \sum_{j \neq i} \epsilon_i \epsilon_j \right) \right] \\
&= \frac{1}{k} v + \frac{k-1}{k}c
\end{split}
\end{align}
$$

if all models are correlated $c = v$, which have no help. but if all models are perfectly uncorrelated $c = 0$, the expected squared error is only $\frac{1}{k} v$

## 7.12 Dropout

when inference, the weights should be multiplied the probability( 0.5 in practice).

## 7.13 Adversarial Training

One of the primary causes of theseadversarial examples is excessive linearity. Adversarial training discourages this highly sensitive locally linear behavior by encouraging the network to be locally constant in the neighborhood of the training data. This can be seen as a way of explicitly introducing a local constancy prior into supervised neural nets.

## 7.14 Tangent Distance, Tangent Prop and ManifoldTangent Classiﬁer

