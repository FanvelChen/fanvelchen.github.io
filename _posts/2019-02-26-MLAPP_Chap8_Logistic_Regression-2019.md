---
layout:     post
title:      MLAPP Chap8 Logistic Regression
date:       2019-02-26
author:     Fanvel Chen
header-img: img/posts/chap6_img.jpg
catalog:    true
mathjax:    true
tags:
    - machine learning
    - MLAPP
---

# Logistic regression

## Model specification

判别式二分类模型

$$
p(y \,|\, \boldsymbol{\mathbf{x}} , \boldsymbol{\mathbf{w}}) = Ber(y \,|\, sigm(\boldsymbol{\mathbf{w}}^T\boldsymbol{\mathbf{x}}))
$$

## Model fitting

负对数似然为

$$
\begin{align}
\begin{split}
NLL(\boldsymbol{\mathbf{w}}) &= -\sum_{i=1}^N \log [\mu_i^{\mathbb{I}(y_i = 1)} \times (1-\mu_i)^{\mathbb{I}(y_i=0)}] \\
&= -\sum_{i=1}^N [y_i \log \mu_i + (1-y_i) \log(1-\mu_i)] \\
\mu_i = \frac{1}{1+\exp(-\boldsymbol{\mathbf{w}}^T\boldsymbol{\mathbf{x}}_i)}
\end{split}
\end{align}
$$

这也被称为**cross entropy**。

梯度和 Hessian 为

$$
\begin{align}
\begin{split}
\boldsymbol{\mathbf{g}} &= \frac{d}{d\boldsymbol{\mathbf{w}}} f(\boldsymbol{\mathbf{w}}) = \sum_i (\mu_i - y_i) \boldsymbol{\mathbf{x}}_i = \boldsymbol{\mathbf{X}}^T(\boldsymbol{\mathbf{\mu}}-\boldsymbol{\mathbf{y}}) \\
\boldsymbol{\mathbf{H}} &= \frac{d}{d\boldsymbol{\mathbf{w}}}\boldsymbol{\mathbf{g}}(\boldsymbol{\mathbf{w}})^T =\sum_i (\nabla_{\boldsymbol{\mathbf{w}}}\mu_i)\boldsymbol{\mathbf{x}}_i^T = \sum_i \mu_i (1-\mu_i) \boldsymbol{\mathbf{x}}_i \boldsymbol{\mathbf{x}}_i^T \\
&= \boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{SX}}
\end{split}
\end{align}
$$

其中 $\\boldsymbol{\mathbf{S}} \triangleq diag(\mu_i(1-\mu_i))$ 。$H$ 是正定的，所以 NLL 是凸的，有唯一全局最小值。

### Steepest descent

基础版本

$$
\boldsymbol{\mathbf{\theta_{k+1}}} = \boldsymbol{\mathbf{\theta}}_k - \eta_k \boldsymbol{\mathbf{g}}_k
$$

缺点：学习率太小则太慢，太大则不收敛

一个更稳定的方法，可以保证收敛到一个局部最小值。根据泰勒公式

$$
f(\boldsymbol{\mathbf{\theta}}+ \eta \boldsymbol{\mathbf{d}}) \approx f(\boldsymbol{\mathbf{\theta}}) + \eta \boldsymbol{\mathbf{g}}^T \boldsymbol{\mathbf{d}}
$$

其中 $\boldsymbol{\mathbf{d}}$ 是下降方向的单位方向向量。选择 $\eta$ 为 最小化 $\phi(\eta) = f(\boldsymbol{\mathbf{\theta}}+\eta \boldsymbol{\mathbf{d}}_k )$，这被称为 **线性搜索**

线性搜索会是zig-zag的，因为 $\eta_k = \arg \min_{\eta > 0} \phi (\eta)$，所以 $\phi'(\eta) = 0$ , 即 $\boldsymbol{\mathbf{d}}^T\boldsymbol{\mathbf{g}} = 0$，所以垂直。

一个启发式的减少zig-zag的方法是加入 **动量 momentum** 项：

$$
\boldsymbol{\mathbf{\theta}}_{k+1} = \boldsymbol{\mathbf{\theta}}_k - \eta_k \boldsymbol{\mathbf{g}}_k + \mu_k (\boldsymbol{\mathbf{\theta}}_k - \boldsymbol{\mathbf{\theta}}_{k-1})
$$

### Newton's method

$$
\boldsymbol{\mathbf{\theta_{k+1}}} = \boldsymbol{\mathbf{\theta}}_k - \eta_k \boldsymbol{\mathbf{g}}_k^{-1}\boldsymbol{\mathbf{g}}_k
$$

#### Iteratively reweighted least squares (IRLS)

用牛顿法求二分类logistic回归的MLE

$$
\begin{align}
\begin{split}
\boldsymbol{\mathbf{w}}_{k+1} &=  \boldsymbol{\mathbf{w}}_k - \boldsymbol{\mathbf{H}}^{-1}\boldsymbol{\mathbf{g}}_k \\
&= \boldsymbol{\mathbf{w}}_k + (\boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{S}}_k \boldsymbol{\mathbf{X}})^{-1} \boldsymbol{\mathbf{X}}^T (\boldsymbol{\mathbf{y}} - \boldsymbol{\mathbf{\mu}}_k) \\
&= (\boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{S}}_k \boldsymbol{\mathbf{X}})^{-1} [(\boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{S}}_k \boldsymbol{\mathbf{X}})\boldsymbol{\mathbf{w}}_k+\boldsymbol{\mathbf{X}}^T (\boldsymbol{\mathbf{y}} - \boldsymbol{\mathbf{\mu}}_k)] \\
&= (\boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{S}}_k \boldsymbol{\mathbf{X}})^{-1} \boldsymbol{\mathbf{X}}^T [\boldsymbol{\mathbf{S}}_k\boldsymbol{\mathbf{Xw}}_k + \boldsymbol{\mathbf{y}} - \boldsymbol{\mathbf{\mu}}_k] \\
&= (\boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{S}}_k \boldsymbol{\mathbf{X}})^{-1} \boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{S}}_k \boldsymbol{\mathbf{z}}_k
\end{split}
\end{align}
$$

其中定义 **working response** 为

$$
\boldsymbol{\mathbf{z}}_k \triangle \boldsymbol{\mathbf{Xw}}_k + \boldsymbol{\mathbf{S}}_k^{-1}(\boldsymbol{\mathbf{y}} - \boldsymbol{\mathbf{\mu}}_k)
$$

考虑到 $\boldsymbol{\mathbf{S}}_k$ 是个对角矩阵，所以

$$
z_{ki} = \boldsymbol{\mathbf{w}}_k^T\boldsymbol{\mathbf{x}}_i + \frac{y_i - \mu_{ki}}{\mu_{ki}(1-\mu_{ki})}
$$

<img src="https://ws2.sinaimg.cn/large/006tKfTcly1g0hte6oogwj30i309jmy7.jpg"/>

### 8.3.5 Quasi-Newton (variable metric) methods

计算 Hessian 太昂贵，仿牛顿方法求Hessian的近似，例如 BFGS 法通过计算 $B_k \approx H_k$

$$
\begin{align}
\begin{split}
B_{k+1} &= B_k + \frac{\boldsymbol{\mathbf{y}}_k\boldsymbol{\mathbf{y}}_k^T}{\boldsymbol{\mathbf{y}}_k^T \boldsymbol{\mathbf{s}}_k} - \frac{(\boldsymbol{\mathbf{B}}_k\boldsymbol{\mathbf{s}}_k)(\boldsymbol{\mathbf{B}}_k\boldsymbol{\mathbf{s}}_k)^T }{\boldsymbol{\mathbf{s}}_k^T\boldsymbol{\mathbf{B}}_k\boldsymbol{\mathbf{s}}_k } \\
\boldsymbol{\mathbf{s}}_k &= \boldsymbol{\mathbf{\theta}}_k - \boldsymbol{\mathbf{\theta}}_{k-1} \\
\boldsymbol{\mathbf{y}}_k &= \boldsymbol{\mathbf{g}}_k - \boldsymbol{\mathbf{g}}_{k-1}
\end{split}
\end{align}
$$

设置 $\boldsymbol{\mathbf{B}}_0 = \boldsymbol{\mathbf{I}}$

逆矩阵 $\boldsymbol{\mathbf{C}}_k \approx \boldsymbol{\mathbf{H}}_k^{-1}$

$$
\boldsymbol{\mathbf{C}}_{k+1} = (\boldsymbol{\mathbf{I}}-\frac{\boldsymbol{\mathbf{s}}_k\boldsymbol{\mathbf{y}}_k^T}{\boldsymbol{\mathbf{y}}_k^T\boldsymbol{\mathbf{s}}_k})\boldsymbol{\mathbf{C}}_k(\boldsymbol{\mathbf{I}}-\frac{\boldsymbol{\mathbf{s}}_k\boldsymbol{\mathbf{y}}_k^T}{\boldsymbol{\mathbf{y}}_k^T\boldsymbol{\mathbf{s}}_k}) +\frac{\boldsymbol{\mathbf{s}}_k\boldsymbol{\mathbf{s}}_k^T}{\boldsymbol{\mathbf{y}}_k^T\boldsymbol{\mathbf{s}}_k}
$$

### $\ell_2$ regularization

如果数据是线性可分的，则所有参数同样比例的增大都是解，通过梯度下降会使得参数趋向无穷。所以需要正则化。

### Multi-class logistic regression

现对后验分布做一个高斯近似

$$
p(y = c \,|\, \boldsymbol{\mathbf{x}}, \boldsymbol{\mathbf{W}}) = \frac{\exp(\boldsymbol{\mathbf{w}}_c^T\boldsymbol{\mathbf{x}})}{\sum_{c'=1}^C \exp(\boldsymbol{\mathbf{w}}_{c'}^T\boldsymbol{\mathbf{x}})}
$$

## 8.4 Bayesian logistic regression

因为没有合适的共轭先验，所以无法确切直接建模 logistic 回归的后验。

### 8.4.1 Laplace approximation

$$
p(\boldsymbol{\mathbf{\theta}} \,|\, \mathcal{D}) = \frac{1}{Z} e^{-E(\boldsymbol{\mathbf{\theta}})}
$$

$E(\boldsymbol{\mathbf{E}})$ 叫做 **能量函数 energy function**，$E(\boldsymbol{\mathbf{\theta}}) = - \log p(\boldsymbol{\mathbf{\theta}}, \mathcal{D})$，以及 $Z = p(\mathcal{D})$。对能量函数在众数$\boldsymbol{\mathbf{\theta}}^*$做泰勒展开

$$
E(\boldsymbol{\mathbf{\theta}}) \approx E(\boldsymbol{\mathbf{\theta}}^*) + (\boldsymbol{\mathbf{\theta}} - \boldsymbol{\mathbf{\theta}}^*)^T\boldsymbol{\mathbf{g}} + \frac{1}{2}(\boldsymbol{\mathbf{\theta}}-\boldsymbol{\mathbf{\theta}}^*)^T \boldsymbol{\mathbf{H}}(\boldsymbol{\mathbf{\theta}} - \boldsymbol{\mathbf{\theta}}^*)
$$

众数时，梯度为0，所以

$$
\begin{align}
\begin{split}
\hat{p}(\boldsymbol{\mathbf{\theta}} \,|\, \mathcal{D}) &\approx \frac{1}{Z} e^{-E(\boldsymbol{\mathbf{\theta}}^*)} \exp [-\frac{1}{2}(\boldsymbol{\mathbf{\theta}} - \boldsymbol{\mathbf{\theta}}^*)^T \boldsymbol{\mathbf{H}} (\boldsymbol{\mathbf{\theta}} - \boldsymbol{\mathbf{\theta}}^*)] \\
&= \mathcal{N} (\boldsymbol{\mathbf{\theta}} \,|\, \boldsymbol{\mathbf{\theta}}^*, \boldsymbol{\mathbf{H}}^{-1})  \\
Z &= p(\mathcal{D}) \approx \int \hat{p}(\boldsymbol{\mathbf{\theta}} \,|\, \mathcal{D})d\boldsymbol{\mathbf{\theta}} = e^{-E(\boldsymbol{\mathbf{\theta}}^*)}(2\pi)^{D/2} |\boldsymbol{\mathbf{H}}|^{-\frac{1}{2}}
\end{split}
\end{align}
$$

### Derivation of the BIC

对上面的近似重写对数边缘似然

$$
\log p(\mathcal{D}) \approx \log p(\mathcal{D} \,|\, \boldsymbol{\mathbf{\theta}}^*) + \log p(\boldsymbol{\mathbf{\theta}}^*) - \frac{1}{2} \log |\boldsymbol{\mathbf{H}}|
$$

如果我们认为先验是uniform的，则其正比于1，可以去掉第二项，以及用 MLE $\hat{\boldsymbol{\mathbf{\theta}}}$ 取代 $\boldsymbol{\mathbf{\theta}}^*$

现在近似第三项。

$$
\boldsymbol{\mathbf{H}} = \sum_{i=1}^N \boldsymbol{\mathbf{H}}_i = \nabla \nabla \log p(\mathcal{D}_i \,|\, \boldsymbol{\mathbf{\theta}})
$$

用一个固定的 $\hat{\boldsymbol{\mathbf{H}}}$ 来近似 $\boldsymbol{\mathbf{H}}_i$

$$
\log |\boldsymbol{\mathbf{H}}| = \log |N\hat{\boldsymbol{\mathbf{H}}}| = \log(N^d |\hat{\boldsymbol{\mathbf{H}}}|) = D\log N + \log |\hat{\boldsymbol{\mathbf{H}}}|
$$

所以 BIC  score 为

$$
\log p(\mathcal{D}) = \log p(\mathcal{D} \,|\, \hat{\boldsymbol{\mathbf{\theta}}}) - \frac{D}{2} \log N
$$

### Gaussian approximation for logistic regression

高斯先验

$$
p(\boldsymbol{\mathbf{w}}) = \mathcal{N}(\boldsymbol{\mathbf{w}} \,|\, \boldsymbol{\mathbf{0}},\boldsymbol{\mathbf{V}}_0)
$$

近似的后验为

$$
p(\boldsymbol{\mathbf{w}} \,|\, \mathcal{D}) \approx \mathcal{N}(\boldsymbol{\mathbf{w}} \,|\, \hat{\boldsymbol{\mathbf{w}}}, \boldsymbol{\mathbf{H}}^{-1})
$$

$$
\begin{align}
\begin{split}
\hat{\boldsymbol{\mathbf{w}}} &= \arg \min_{\boldsymbol{\mathbf{w}}}E(\boldsymbol{\mathbf{w}}) \\
E(\boldsymbol{\mathbf{w}}) &= -(\log p(\mathcal{D} \,|\, \boldsymbol{\mathbf{w}}) + \log p(\boldsymbol{\mathbf{w}})) \\
\boldsymbol{\mathbf{H}} &=  \nabla^2 E(\boldsymbol{\mathbf{w}}) |_{\hat{\boldsymbol{\mathbf{w}}}}
\end{split}
\end{align}
$$

### Approximating the posterior predictive

形式上，预测应为

$$
p(y \,|\, \boldsymbol{\mathbf{w}} , \mathcal{D}) = \int p(y \,|\, \boldsymbol{\mathbf{x}}, \boldsymbol{\mathbf{w}}) p(\boldsymbol{\mathbf{w}} \,|\, \mathcal{D}) d\boldsymbol{\mathbf{w}}
$$

然而这个积分难以计算。

最简单的近似是 plug-in，但这会低估不确定性

#### Monte Carlo approximation

$$
p(y = 1 \,|\, \boldsymbol{\mathbf{x}}, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^S sigm((\boldsymbol{\mathbf{w}}^s)^T \boldsymbol{\mathbf{x}})
$$

#### Probit approximation (moderated output)

利用 **probit** 函数和 sigmoid 函数的相似

$$
\Phi(a) \triangleq \int_{-\infty}^a \mathcal{N}(x \,|\, 0,1)dx
$$

### Residual analysis (outlier detection)

在交叉验证的后验预测概率很小的样例。

## Online learning and stochastic optimization

### Online learning and regret minimization

呈现一个样本$\boldsymbol{\mathbf{z}}_k$，学习器回应一个参数估计$\boldsymbol{\mathbf{\theta}}_k$

$$
regret_k \triangleq \frac{1}{k} \sum_{t=1}^{k} f(\boldsymbol{\mathbf{\theta}}_t,\boldsymbol{\mathbf{z}}_t) - \min_{\boldsymbol{\mathbf{\theta}}^* \in \Theta} \frac{1}{k} \sum_{t=1}^k f(\boldsymbol{\mathbf{\theta}}^*,\boldsymbol{\mathbf{z}}_t)
$$

**online gradient descent**

$$
\boldsymbol{\mathbf{\theta}}_{k+1} = proj_{\Theta}(\boldsymbol{\mathbf{\theta}}_k - \eta_k \boldsymbol{\mathbf{g}}_k)
$$

其中

$$
proj_{\mathcal{V}}(\boldsymbol{\mathbf{v}}) = \arg \min_{\boldsymbol{\mathbf{w}} \in \boldsymbol{\mathbf{V}}} ||\boldsymbol{\mathbf{w}} - \boldsymbol{\mathbf{v}}||_2
$$

### Stochastic optimization and risk minimization

优化存在随机变量函数叫做 **stochastic optimization**。

每一步采用上面的算法叫做 **stochastic gradient descent**

#### Setting the step size

保证 SGD 收敛的充分条件， **Robbins-Monro** 条件

$$
\sum_{k=1}^{\infty} \eta_k = \infty, \sum_{k=1}^\infty \eta_k^2 < \infty
$$

#### Per-parameter step sizes

SGD 的一个缺点是它对所有的参数采用一样的 step size。用一种叫 **adagrad** 的方法

$$
\theta_i(k+1) = \theta_i(k) - \eta \frac{g_i(k)}{\tau_0 + \sqrt{s_i(k)}} \\
s_i(k) = s_i(k-1) +g_i(k)^2
$$

#### SGD compared to batch learning

如果数据流不是无限的，我们也可以通过随机采样来模拟

<img src="https://ws4.sinaimg.cn/large/006tKfTcly1g0jsozl2gdj30dr08a0tk.jpg" />

过完整个数据集叫做以一个 **epoch**。可以采取有 B 个样本的 **mini-batch**，如果$B=1$ ，就是标准 SGD ，如果 $B=N$ ，则为标准 steepest descent 。

### A Bayesian view

迭代地使用 Bayes rule

$$
p(\boldsymbol{\mathbf{\theta}} \,|\, \mathcal{D}_{1:k}) \propto p(\mathcal{D} \,|\, \boldsymbol{\mathbf{\theta}}) p(\boldsymbol{\mathbf{\theta}} \,|\, \mathcal{D}_{1:k-1})
$$
