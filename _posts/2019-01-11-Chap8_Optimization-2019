---
layout:     post
title:      Chap8 Optimization
date:       2019-01-11
author:     Fanvel Chen
header-img: img/posts/chap6_img.jpg
catalog:    true
mathjax:    true
tags:
    - deep learning
---

# Chap8 Optimization for Training DeepModels

## Chanllenges in Neural Network Optimization

### Ill-Conditioning

When the $\frac{1}{2} \epsilon^2 \boldsymbol{g^THg}$ greater than $\epsilon \boldsymbol{g^Tg}$ in the second-order Taylor series expansion of the cost function, ill-conditioning becomes a problem.

### Local Minima

most local minima have a low cost function value, and that it is not important to ﬁnd a true global minimum rather than to ﬁnd a point in parameter space that has low but not minimal cost.

### Plateaus, Saddle Points and Other Flat Regions

The expected ratio of the number of saddle points to local minima growsexponentially with $n$.

### Cliﬀs and Exploding Gradients

Gradient clipping

### Long-Term Dependencies

a long-term dependency path of a weight would bring vanishing and exploding gradient problem.

## Basic Algorithms

### Stochastic Gradient Descent

### Momentum

$$
\boldsymbol{v} \gets \alpha \boldsymbol{v} - \epsilon \boldsymbol{g},
$$

$$
\boldsymbol{\theta} \gets \boldsymbol{\theta} + \boldsymbol{v}
$$

![momentum](https://ws2.sinaimg.cn/large/006tNc79ly1fz1jdkput7j30az0a7wfk.jpg)

### Nesterov Momentum

$$
\boldsymbol{g} = \nabla_{\theta} \left[ \frac{1}{m} \sum_{i=1}^m L(\boldsymbol{f}(\boldsymbol{x^{(i)}} ; \boldsymbol{\theta} + \alpha \boldsymbol{v} ) , \boldsymbol{y^{(i)}})  \right]
$$

## Parameter Initialization Strategies

The initialparameters need to “break symmetry” between diﬀerent units.

## Algorithms with Adaptive Learning Rates

### AdaGrad

**AdaGrad Algorithm**

$\boldsymbol{r} = 0$

**while** stopping criterion not met **do**

&emsp; Sample $m$ examples minibatch

&emsp; $\boldsymbol{g} \gets \frac{1}{m} \nabla_{\boldsymbol{\theta}} \sum_i L(f(\boldsymbol{x^{(i)}} ; \boldsymbol{\theta}) , \boldsymbol{y^{(i)}} )$

&emsp; $\boldsymbol{r} \gets \boldsymbol{r} + \boldsymbol{g} \odot \boldsymbol{g}$

&emsp; $\Delta \boldsymbol{\theta} \gets - \frac{\epsilon}{\delta+\sqrt{\boldsymbol{r} }  }  \odot \boldsymbol{g}$

&emsp; $\boldsymbol{\theta} \gets \boldsymbol{g} + \Delta \boldsymbol{\theta}$

**end while**

### RMSProp

RMSProp uses an exponentially decaying average to discard history from the extreme past so that it can converge rapidly after ﬁnding a convex bowl.

**RMSProp Algorithm**

$\boldsymbol{r} = 0$

**while** stopping criterion not met **do**

&emsp; Sample $m$ examples minibatch

&emsp; $\boldsymbol{g} \gets \frac{1}{m} \nabla_{\boldsymbol{\theta}} \sum_i L(f(\boldsymbol{x^{(i)}} ; \boldsymbol{\theta}) , \boldsymbol{y^{(i)}} )$

&emsp; $\boldsymbol{r} \gets \rho \boldsymbol{r} + (1-\rho)\boldsymbol{g} \odot \boldsymbol{g}$

&emsp; $\Delta \boldsymbol{\theta} \gets - \frac{\epsilon}{\sqrt{\delta+\boldsymbol{r} }  }  \odot \boldsymbol{g}$

&emsp; $\boldsymbol{\theta} \gets \boldsymbol{g} + \Delta \boldsymbol{\theta}$

**end while**

### Adam

**RMSProp Algorithm**

$\boldsymbol{s} = 0,\boldsymbol{r} = 0, t = 0$

**while** stopping criterion not met **do**

&emsp;Sample $m$ examples minibatch

&emsp; $\boldsymbol{g} \gets \frac{1}{m} \nabla_{\boldsymbol{\theta}} \sum_i L(f(\boldsymbol{x^{(i)}} ; \boldsymbol{\theta}) , \boldsymbol{y^{(i)}} )$

&emsp;$t \gets t+1$

&emsp;$\boldsymbol{s} \gets \rho_1 \boldsymbol{s} + (1-\rho_1)\boldsymbol{g}$

&emsp;$\boldsymbol{r} \gets \rho_2 \boldsymbol{r} + (1-\rho_2)\boldsymbol{g} \odot \boldsymbol{g}$

&emsp;$\hat{\boldsymbol{s}} \gets \frac{\boldsymbol{s}}{1-\rho_1^t}$

&emsp;$\hat{\boldsymbol{r}} \gets \frac{\boldsymbol{r}}{1-\rho_2^t}​$

&emsp; $\boldsymbol{\theta} \gets \boldsymbol{g} + \Delta \boldsymbol{\theta}$

**end while**

## Approximate Second-Order Methods

### Newton’s Method

$$
J(\boldsymbol{\theta}) \simeq J(\boldsymbol{\theta_0}) + (\boldsymbol{\theta} - \boldsymbol{\theta_0})^T \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta_0}) + \frac{1}{2}(\boldsymbol{\theta} - \boldsymbol{\theta_0})^T \boldsymbol{H} (\boldsymbol{\theta} - \boldsymbol{\theta_0} )
$$

the Newton parameter update rule:

$$
\boldsymbol{\theta^*} = \boldsymbol{\theta_0} - \boldsymbol{H^{-1}} \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_0})
$$

If the eigenvalues of the Hessian are not all positive, for example, near a saddle point, then Newton's method can actually cause updates to move in the wrong direction. We can use the regularization version:

$$
\boldsymbol{\theta^*} = \boldsymbol{\theta_0} - [H(f(\boldsymbol{\theta_0}))+ \alpha \boldsymbol{I}]^{-1} \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_0})
$$

but the nagetive eigenvalues should close to 0.

And the computational burden is significant.

### Conjugate Gradients

$$
\boldsymbol{d_t} = \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) + \beta_t \boldsymbol{d_{t-1}}
$$

computing $\beta_t$

Fletcher-Reeves:

$$
\beta_t = \frac{\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_t})^T \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_t}) }{\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta_{t-1}})^T \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta_{t-1}}) }
$$

Polak-Ribière

$$
\beta_t = \frac{(\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_t}) - \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_{t-1} }))^T \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_t})}{\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_{t-1}  })^T \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_{t-1} })}
$$

### BFGS

approximate $\boldsymbol{H^{-1}}$ with $\boldsymbol{M_t}$

## Optimization Strategies and Meta-Algorithms

### Batch Normalization

At training time

$$
\mu = \frac{1}{m} \sum_i \boldsymbol{H_{i,:}}
$$

and

$$
\sigma = \sqrt{\delta + \frac{1}{m} \sum_i (\boldsymbol{H} - \boldsymbol{\mu})_i^2  }
$$

at test time, $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ may be replaced by running averages that were collected during training time.

To maintain the expressive power of the network, it is common to replace the batch of hidden unit activations $\boldsymbol{H}$ with $\gamma \boldsymbol{H}' + \beta$

### Coordinate Descent

minimize $f(\boldsymbol{x})$ with respect to a single $x_i$, respectively.

### Polyak Averaging

$$
\hat{\boldsymbol{\theta}}^{(t)}  = \alpha \hat{\boldsymbol{\theta}}^{(t-1)} + (1 - \alpha) \boldsymbol{\theta}^{(t)}
$$

### Supervised Pretraining

greedy supervised pretraining
