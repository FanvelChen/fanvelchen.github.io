---
layout:     post
title:      Chap6 Deep Feedforward Networks
date:       2019-01-06
author:     Fanvel Chen
header-img: img/posts/chap6_img.jpg
catalog:    true
mathjax:    true
tags:
    - deep learning
---

# Chap 6 Deep Feedforward Networks

To extend linear models to represent nonlinear functions of $\boldsymbol{x}$ , we can apply the linear model not to $\boldsymbol{x}$ itself but to a tranformed input $\phi(\boldsymbol{x})$ 

How to choose the mapping $\phi$

1. a generic $\phi$ .   Based only on the principle of local smoothness and do not encode enough prior information to solve advanced problems.
2. manually engineer $\phi$ .  Dominant approach before deep learning.
3. to learn $\phi$ .  Model $y = f(\boldsymbol{x};\boldsymbol{\theta},\boldsymbol{w})=\phi(\boldsymbol{x};\boldsymbol{\theta})^T\boldsymbol{w}$

## 6.1 Example: Learning XOR

target function : $y = f^*(\boldsymbol{x})$ we want to learn.

model provides a function : $y = f(\boldsymbol{x};\boldsymbol{\theta})$

treat this problem as a **regression problem**. use a **mean squared error** loss function.

$$
J(\boldsymbol{\theta}) = \frac{1}{4}\sum_{x \in \mathbb{X}} (f^*(\boldsymbol{x})-f(\boldsymbol{x};\boldsymbol{\theta}))
$$

$\mathbb{X}=\lbrace  [0,0]^T ,  [0,1]^T ,  [1,0]^T ,  [1,1]^T  \rbrace$ as train data and test data( just learn XOR and forgot generalization )

hidden units $\boldsymbol{h}=f^{(1)}(\boldsymbol{x};\boldsymbol{W},\boldsymbol{c})$ , and output $y = f^{(2)}(\boldsymbol{x};\boldsymbol{w},b)$

and the complete model is $f(\boldsymbol{x};\boldsymbol{W},\boldsymbol{c},\boldsymbol{w},b) = f^{(2)}(f^{(1)}(x))$

Then choose $f^{(1)}$ , it shouldn't be linear. We choose **activation function** after affine transformation $\boldsymbol{h} = g(\boldsymbol{W}^T\boldsymbol{x}+\boldsymbol{c})$ , and respectively $\boldsymbol{h_i} = g(\boldsymbol{x}^T \boldsymbol{W_{:,i}}+\boldsymbol{c_i})$

the default recommendation is to use *rectified linear unit* or **ReLU** :

$$
g(z)=\max \left\{ 0,z\right\}
$$

so our complete network is

$$
f(\boldsymbol{x};\boldsymbol{W},\boldsymbol{c},\boldsymbol{w},b) = \boldsymbol{w}^T max\left\{ 0,\boldsymbol{W}^T\boldsymbol{x}+\boldsymbol{c} \right\} +b
$$

there is a solution:

$$
\boldsymbol{W} = 
\begin{bmatrix} 
1 & 1 \\ 
1 & 1 
\end{bmatrix},
$$

$$
\boldsymbol{c} = 
\begin{bmatrix} 
0 \\ 
-1  
\end{bmatrix},
$$

$$
\boldsymbol{w} = 
\begin{bmatrix} 
1 & -2\\  
\end{bmatrix}, b = 0
$$

so for
$$
\boldsymbol{X} = 
\begin{bmatrix} 
0 & 0 & 1 & 1 \\ 
0 & 1 & 0 & 1 
\end{bmatrix}
$$

$$
\begin{align}
\begin{split}
\boldsymbol{W}^T \boldsymbol{X} + \boldsymbol{c} &= 
\begin{bmatrix} 
1 & 1 \\ 
1 & 1 
\end{bmatrix}^T
\begin{bmatrix} 
0 & 0 & 1 & 1 \\ 
0 & 1 & 0 & 1 
\end{bmatrix} +
\begin{bmatrix} 
0 \\ 
-1  
\end{bmatrix}  \\
& = 
\begin{bmatrix} 
0 & 1 & 1 & 2 \\ 
-1 & 0 & 0 & 1 
\end{bmatrix}

\end{split}
\end{align}
$$

$$
\begin{align}
\begin{split}
\boldsymbol{w}^T max\left\{ 0,\boldsymbol{W}^T\boldsymbol{x}+\boldsymbol{c} \right\} + b &=
\begin{bmatrix} 
1 \\ 
-2  
\end{bmatrix}
\begin{bmatrix} 
0 & 1 & 1 & 2 \\ 
0 & 0 & 0 & 1  
\end{bmatrix} \\
&= \begin{bmatrix} 
0 & 1 & 1 & 0 
\end{bmatrix}

\end{split}
\end{align}
$$

## 6.2 Gradient-Based Learning

### 6.2.1 Cost Function

#### 6.2.1.1 Learning Conditional distribution with Maximum Likelihood

Most networks are trained using **maximum likelihood** . The cost function is simply the **negative log-likelihood** , or equivalently the **cross-entropy** between the training data and the model distribution

$$
J(\boldsymbol{\theta}) = - \mathbb{E}_{\mathsf{x,y} \sim \hat{p}_{data}} \log p_{model}(\boldsymbol{y}\, |\, \boldsymbol{x} )
$$

#### 6.2.1.2 Learning Conditional Statistic

We often want to learn just one conditional statistic of $\boldsymbol{y}$ given $\boldsymbol{x}$ , instead of the full probability distribution.

We can view the const function as being a **functional**  rather than just a function, so we can thus think of learning as choosing a function rather than merely choosing a set of parameters.

**calculus of variations** can be used to derive the following two results:

1

$$
f^* = \arg \min_f \mathbb{E}_{\mathsf{x,y} \sim p_{data} }||\boldsymbol{y} -f(\boldsymbol{x})||^2
$$

so
$$
f^*(\boldsymbol{x}) = \mathbb{E}_{\mathsf{y} \sim p_{data}(\boldsymbol{y}\,|\,\boldsymbol{x})}[\boldsymbol{y}]
$$
which means **mean squared error** cost function gives a function that predicts the **mean** of $\boldsymbol{y}$ for each value of $\boldsymbol{x}$.

2

$$
f^* = \arg \min_f \mathbb{E}_{\mathsf{x,y} \sim p_{data} }||\boldsymbol{y} -f(\boldsymbol{x})||_1
$$

so

$$
f^*(\boldsymbol{x}) = Median_{\mathsf{y} \sim p_{data}(\boldsymbol{y}\,|\,\boldsymbol{x})}[\boldsymbol{y}]
$$

which means **mean absolute error** cost function yields a function that predicts the **median** value of $\boldsymbol{y}$ for each value of $\boldsymbol{x}$.

### 6.2.2 Output Units

#### 6.2.2.1 Linear Units for Gaussian Output Distributions

Given features $\boldsymbol{h}$ , the output units produces a vector $\hat{\boldsymbol{y}} = \boldsymbol{W}^T \boldsymbol{h} + \boldsymbol{b}$ .

The Linear output layers are often used to produce the mean of a conditional Gaussian distribution:

$$
p(\boldsymbol{y} \, | \, \boldsymbol{x}) = \mathcal{N}(\boldsymbol{y} ; \hat{\boldsymbol{y}},\boldsymbol{I})
$$

**Maximizing the log-likelihood** is then equivalent to **minimizing the mean squared error.**

#### 6.2.2.2 Sigmoid Units for Bernoulli Output Distributions

Classiﬁcation problems with two classes can be cast in this form.

The maximum likelihood approach is to deﬁne a Bernoulli distribution over $y$ conditioned on $\boldsymbol{x}$ .

if we supposed to use a linear unit and threshold its value to obtain a valid probability: 

$$
P(y=1 \,|\, \boldsymbol{x}) = \max \left\{ 0, \min\left\{ 1, \boldsymbol{w}^T \boldsymbol{h} + b \right\}  \right\}
$$

but the gradient would be 0. so we use sigmoid:

$$
P(y=1\, |\,\boldsymbol{x}) = \sigma(\boldsymbol{w}^T \boldsymbol{h} + b) \\
\sigma(z) = \frac{1}{1+\exp(-z)}
$$

![](https://ws2.sinaimg.cn/large/006tNc79ly1fyw4yzgl4zj30va0numyt.jpg)

If we begin with the assumption that the **unnormalized log probabilities** $\log \tilde{P}(y)$ are linear in $y$ and $z=\boldsymbol{w}^T \boldsymbol{h} + b$

$$
\begin{align}
\begin{split}
\log \tilde{P}(y) &= yz \\
\tilde{P}(y) &= \exp(yz) \\
P(y) &= \frac{\exp(yz)}{\exp(0z) +\exp(1z)}  \\
P(y) &= \frac{\exp(yz)}{1 +\exp(z)} \\
P(y) &= \sigma((2y-1)z)
\end{split}
\end{align}
$$

The $z$ variable deﬁning such a distribution over binary variables is called **logit**
If we use **maximum likelihood** as our cost function:

$$
\begin{align}
\begin{split}
J(\theta) &= - \log P(y_{gt} \,|\, \boldsymbol{x}) \\
&= - \log \sigma((2y_{gt}-1)z) \\
&= \log (1+\exp((1-2y_{gt})z)) \\
&= \zeta((1-2y_{gt})z)
\end{split}
\end{align} \\
$$

and $\zeta(x) = \log (1 + exp(x))$ is called **softplus function** 

![](https://ws1.sinaimg.cn/large/006tNc79ly1fyw6h34fm8j30uo0n63zm.jpg)

so it saturates only when $(1-2y_{gt})z$  is **very negative**, which occurs when the model already has the right answer: ( $z$ is very positive and $y_{gt}=1$ ) or ( $z$ is very negative and $y_{gt}=0$ ) . When the model is totally wrong: ( $z$ is very positive but $y_{gt} = 0$ ) or ($z$ is very negative and $y_{gt}=1$) , the $(1-2y_{gt})z$ is very positive ( close to $\left\| z \right\|$) and the softplus function asymptotes toward $\left\| z \right\|$ , with derivative asymptotes to $sign(z)$. 

If we use other loss function, MSE e.g., the loss function saturates anytime $\sigma(z)$ saturates. The sigmoid function saturates to $0$ when $z$ becomes very negative and saturates to $1$ when $z$ becomes very positive, no matter the model has the correct answer or the incorrect answer.

For this reason, **maximum likelihood** is almost always the **preferred** approach to training **sigmoid output units**.

**Notes** : In software implementations, to avoid numerical problems, we should write the negative log-likelihood $-\log P(y_{gt} \, \| \, \boldsymbol{x})$ as a function of $z$ : $\zeta((1-2y_{gt})z)$, rather than as a function of $\hat{y}=\sigma(z)$ : $- \log \sigma((2y_{gt}-1)z)$ . Because if the sigmoid function **underflow**s to $0$, then taking the logarithm of $\hat{y}$ would get **negative infinity**.

for example, if the $z$ is wrong( initialization e.g.), maybe $z = -0.1x+0.5$ ,but for positive $x$, the $y_{gt} = 1$ . The part of $\log$ as a function of $\hat{y}$ is  $\sigma((2y_{gt}-1)z)$, and the counterpart of $z$ is $1+\exp((1-2y_{gt})z)$.

![](https://ws3.sinaimg.cn/large/006tNc79ly1fywt1aw95cj30lz0bnaar.jpg)

we can see the former would be close to infinity.

#### 6.2.2.3 Softmax Units for Multinoulli Output Distributions
To represent a probability distribution over a discrete variable with $n$ possible values.
First, a linear layer predicts unnormalized log probabilities:

$$
\boldsymbol{z} = \boldsymbol{W}^T \boldsymbol{h} + \boldsymbol{b}
$$

where $z_i = \log \hat{P}(y = i \, \| \, \boldsymbol{x})$ , and we exponentiate and normalize $z$:

$$
softmax(\boldsymbol{z})_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

Using maximum log-likelihood, we wish to maximize $\log P(y = i;\boldsymbol{z}) = \log softmax(\boldsymbol{z})_i$

$$
\log softmax(\boldsymbol{z})_i = z_i - \log \sum_j \exp(z_j)
$$

Observe that:

$$
softmax(\boldsymbol{z}) = softmax(\boldsymbol{z}+c)
$$

so we can derive a numerically stable variant of the softmax:

$$
softmax(\boldsymbol{z}) = softmax(\boldsymbol{z}- \max_i z_i)
$$

if the ground truth $i$ of $z$ : $z_i = \max_i z_i$ and $z_i$ is much greater than all the others, $softmax(z)_i$ saturates to 1. if $z_i$ is not maximal and the maximal value is much greater, the $softmax(z)_i$ saturates to 0.

#### 6.2.2.4 Other Output Types

## 6.3 Hidden Units

### 6.3.1 Rectiﬁed Linear Units and Their Generalizations

Rectiﬁed linear units use the activation function $g(z)=\max \lbrace 0,z \rbrace$

### 6.3.2 Logistic Sigmoid and Hyperbolic Tangent

The widespread saturation of sigmoidal units can make gradient-based learning very diﬃcult. For this reason, their use as hidden units in feedforward networks is now discouraged.

### 6.3.3 Other Hidden Units

One possibility is to not have an activation at all. If every layer of the neural network consists of only linear transformations, then the network as a whole will be linear. However, we can use it to save parameters.
Considering a $n$-dimension input and $p$-dimension output layer $\boldsymbol{h}=g(\boldsymbol{W}^T \boldsymbol{x}+\boldsymbol{b}) = g(\boldsymbol{V}^T\boldsymbol{U}^T \boldsymbol{x}+\boldsymbol{b})$ , $\boldsymbol{W} \in \mathbb{R}^{n \times p}, \boldsymbol{U} \in \mathbb{R}^{n \times q}, \boldsymbol{V} \in \mathbb{R}^{q \times p}$
if $q(n+p) < np$ ,it can save parameters. It comes at the cost of constraining the linear transformation to be low rank, but these low-rank relationships are often suﬃcient.
others:
1. radial basis function, RBF: $h_i = \exp(-\frac{1}{\sigma_i^2} \left\| \left\| \boldsymbol{W}_{:,i} - \boldsymbol{x} \right\| \right\|^2)$
2. softplus $\zeta(z) = \log(1+e^z)$ , a smoother version of ReLU.
3. hard tans: $g(z) = \max(-1, \min(1,z))$

## 6.4 Architecture Design

### 6.4.1 Universal Approximation Properties and Depth

**universal approximation theorem** states that a feedforward network with a linear ouput layer and at leat one hidden layer with any "squashing" activation function can approximate any **Borel measurable function** from one finite-dimensional space to another with any desired nonzero amount of error, provided that the network is given enough hidden units.

we choose deeper model rather than wider model.

### 6.4.2 Other Architectural Considerations

In general, the layers need not be connected in a chain. 

Another key consideration of architecture design is exactly how to connect a pair of layers to each other. Every input unit is connected to every outputunit. or a sparse connections.

## 6.5 Back-Propagation and Other DiﬀerentiationAlgorithms

Actually, back-propagationrefers only to the method for computing the gradient.

### 6.5.1 Computational Graphs

### 6.5.2 Chain Rule of Calculus

$y = g(x)$ and $z = f(g(x)) = f(y)$

$$
\frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx}
$$

and we can generalize this beyong the scalar case. $\boldsymbol{x} \in \mathbb{R}^m$ , $\boldsymbol{y} \in \mathbb{R}^n$.

$$
\frac{\partial z}{\partial x_i} = \sum_j \frac{\partial z}{\partial y_j} \frac{\partial y_j}{\partial x_i}
$$

or in vector notation

$$
\nabla_{\boldsymbol{x}}z = \left( \frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}} \right)^T \nabla_{\boldsymbol{y}}z
$$

or even in tensor 

$$
\nabla_{\mathsf{X}}z = \sum_j (\nabla_{\mathsf{X}} \mathsf{Y}_j) \frac{\partial z}{\partial \mathsf{Y}_j}
$$

### 6.5.3 Recursively Applying the Chain Rule to Obtain Backprop
