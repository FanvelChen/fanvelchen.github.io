---
layout:     post
title:      MLAPP Chap7 Linear Regression
date:       2019-02-23
author:     Fanvel Chen
header-img: img/posts/chap6_img.jpg
catalog:    true
mathjax:    true
tags:
    - machine learning
    - MLAPP
---

[TOC]

# MLAPP 第7章 线性回归（Linear Regression）

## Model specification

线性形式：

$$
p(y \,|\, \boldsymbol{\mathbf{x},\theta}) = \mathcal{N}(y \,|\, \boldsymbol{\mathbf{w}}^T \boldsymbol{\mathbf{x}}, \sigma^2)
$$

或者通过 **基函数扩展 basis function expansion** 将 $\boldsymbol{\mathbf{x}}$ 替换为非线性的 $\phi (\boldsymbol{\mathbf{x}})$ ,例如 $\phi (x) = [1,x,x^2,\cdots,x^d]$

## MLE (least squares)

$$
\hat{\boldsymbol{\theta}} \triangleq \arg \max_{\boldsymbol{\theta}} \log p(\mathcal{D} \,|\, \boldsymbol{\theta})
$$

假设训练数据是iid的，则**对数似然 log likelihood**为：

$$
\ell(\boldsymbol{\theta}) \triangleq \log p(\mathcal{D} \,|\, \boldsymbol{\theta}) = \sum_{i=1}^{N} \log p(y_i \,|\, \boldsymbol{\mathbf{x}}_i ,\boldsymbol{\theta})
$$

或者可以用最小化 **负对数似然 negative log likelihood (NLL)**

$$
{\rm NLL}(\boldsymbol{\theta}) \triangleq - \sum_{i=1}^{N} \log p(y_{i} \,|\, \boldsymbol{\mathbf{x}}_{i} ,\boldsymbol{\theta})
$$

因为很多优化软件提供的是最小化算法。

$$
\begin{align}
\begin{split}
\ell({\boldsymbol{\theta}}) &= \sum_{i=1}^N \log [ (\frac{1}{2 \pi \sigma^2})^\frac{1}{2} \exp (- \frac{1}{2\sigma^2}(y_{i} - \boldsymbol{\mathbf{w}}^T \boldsymbol{\mathbf{x}}_{i})^2)] \\
&= -\frac{1}{2\sigma^2} RSS(\boldsymbol{\mathbf{w}}) - \frac{N}{2} \log(2 \pi \sigma^2)
\end{split}
\end{align}
$$

其中RSS为 **residual sum of squares** ：

$$
{\rm RSS}(\boldsymbol{\mathbf{w}}) \triangleq \sum_{i=1}^N (y_i - \boldsymbol{\mathbf{w}}^T \boldsymbol{\mathbf{x}}_{i}))^2
$$

RSS/N 称为 **mean squared error (MSE) ** ，所以 MLE 等价于最小化  NLL 或者 RSS 或者 MSE ，即 **least squares**。

### Derivation of the MLE

重写 NLL：

$$
\begin{align}
\begin{split}
{\rm NLL}(\boldsymbol{\mathbf{w}}) &\propto \frac{1}{2}(\boldsymbol{\mathbf{y}} - \boldsymbol{\mathbf{Xw}})^T (\boldsymbol{\mathbf{y}} - \boldsymbol{\mathbf{Xw}}) \\
&= \frac{1}{2} \boldsymbol{\mathbf{w}}^T (\boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{X}})\boldsymbol{\mathbf{w}} - \boldsymbol{\mathbf{w}}^T (\boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{y}})
\end{split}
\end{align}
$$

其中

$$
\begin{align}
\begin{split}
\boldsymbol{\mathbf{X}}^T\boldsymbol{\mathbf{X}} &= \sum_{i=1}^N \boldsymbol{\mathbf{x}}_i\boldsymbol{\mathbf{x}}_i^T \\
&= \sum_{i=1}^N \left(
 \begin{matrix}
   x_{i,1}^2 & \cdots & x_{i,1}x_{i,D} \\
    & \ddots &  \\
   x_{i,D}x_{i,1} & \cdots & x_{i,D}^2
  \end{matrix}
  \right)
\end{split}
\end{align}
$$

以及

$$
\boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{y}} = \sum_{i=1}^{N} \boldsymbol{\mathbf{x}}_i y_i
$$

$\boldsymbol{\mathbf{X}} \in \mathbb{R}^{N \times D}$

所以梯度为

$$
\begin{align}
\begin{split}
g(\boldsymbol{\mathbf{w}}) &= \frac{\partial NLL}{\partial \boldsymbol{\mathbf{w}}} \\
&= \boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{X}}\boldsymbol{\mathbf{w}} - \boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{y}} \\
&= \sum_{i=1}^{N}  \boldsymbol{\mathbf{x}}_i (\boldsymbol{\mathbf{w}}^T \boldsymbol{\mathbf{x}}_i -y_i)
\end{split}
\end{align}
$$

使其为 0 ，则得到 **ordinary least squares (OLS) ** 解：

$$
\hat{\boldsymbol{\mathbf{w}}}_{OLS} = (\boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{X}})^{-1} \boldsymbol{\mathbf{X}}\boldsymbol{\mathbf{y}}
$$

### Geometric interpretation

假设 $N > D$，记 $\boldsymbol{\tilde{\mathbf{x}}}_j \in \mathbb{R}^N$ 为 $\boldsymbol{\mathbf{X}}$ 的第 $j$ 列，则可以把这个问题看作

$$
arg\min\limits_{\hat{\boldsymbol{\mathbf{y}}} \in {\rm span}(\lbrace \boldsymbol{\tilde{\mathbf{x}}}_1, \cdots, \boldsymbol{\tilde{\mathbf{x}}}_D  \rbrace)} \| \boldsymbol{\mathbf{y}} - \hat{\boldsymbol{\mathbf{y}}} \|_2
$$

所以存在一个权值向量 $\boldsymbol{\mathbf{w}} \in \mathbb{R}^D$ 使得

$$
\hat{\boldsymbol{\mathbf{y}}} = w_1\boldsymbol{\tilde{\mathbf{x}}}_1 + \cdots + w_D\boldsymbol{\tilde{\mathbf{x}}}_D = \boldsymbol{\mathbf{Xw}}
$$

所以 $\hat{\boldsymbol{\mathbf{y}}}$ 应是 $\boldsymbol{\mathbf{y}}$ 在 $\boldsymbol{\tilde{\mathbf{x}}}_j$ 张成空间中的投影，即

$$
\begin{align}
\begin{split}
&\boldsymbol{\tilde{\mathbf{x}}}_j^T(\boldsymbol{\mathbf{y}} - \hat{\boldsymbol{\mathbf{y}}}) = 0 \\
\Rightarrow & \boldsymbol{\mathbf{X}}^T(\boldsymbol{\mathbf{y}}-\boldsymbol{\mathbf{Xw}}) = \boldsymbol{\mathbf{0}} \\
\Rightarrow & \boldsymbol{\mathbf{w}} = (\boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{X}})^{-1} \boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{y}}
\end{split}
\end{align}
$$

额外的，注意到 $\hat{\boldsymbol{\mathbf{y}}} = \boldsymbol{\mathbf{Xw}} = \boldsymbol{\mathbf{X}}  (\boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{X}})^{-1} \boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{y}} = \boldsymbol{\mathbf{Py}}$ ，所以我们称

$$
\boldsymbol{\mathbf{P}} \triangleq \boldsymbol{\mathbf{X}}  (\boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{X}})^{-1} \boldsymbol{\mathbf{X}}^T 
$$

为 **hat matrix**

## Robust linear regression

如果训练数据中有**离群值 outliers** ，则会得到一个较差的拟合，因为平方惩罚偏离的幅度较大。

一种获得**鲁棒性**的方法是用一些具有**heavy tails**的分布来取代高斯分布，例如拉普拉斯分布：

$$
\begin{align}
\begin{split}
p(y \,|\, \boldsymbol{\mathbf{x}}, \boldsymbol{\mathbf{w}} , b) &= Lap(y \,|\, \boldsymbol{\mathbf{w}}^T \boldsymbol{\mathbf{x}},b) \\
& \propto exp(- \frac{1}{b} | y - \boldsymbol{\mathbf{w}}^T \boldsymbol{\mathbf{x}} |)
\end{split}
\end{align}
$$

$$
\ell(\boldsymbol{\mathbf{w}}) = \sum_i |r_i(\boldsymbol{\mathbf{w}}) |
$$

其中 $r_i = y_i - \boldsymbol{\mathbf{w}}^T\boldsymbol{\mathbf{x}}_i$

即通过L1代替L2来获得鲁棒性。然而这是非线性函数，较难优化，但我们可以通过一些**分离变量 split variable**技巧的线性约束使 NLL 转化为线性目标。

$$
r_i \triangleq r_i^+ - r_i^-
$$

则优化问题转化为

$$
\min\limits_{\boldsymbol{\mathbf{w}} ,\boldsymbol{\mathbf{r}}^+ , \boldsymbol{\mathbf{r}}^-} \sum_i (r_i^+ - r_i^-) \\
s.t. \;\;\; r_i^+ \geq 0 , r_i^- \geq 0, \boldsymbol{\mathbf{w}}^T\boldsymbol{\mathbf{x}}_i +r_i^+ - r_i^- = y_i
$$

成为一个线性规划问题，$3N$ 个限制条件以及 $D+2N$ 个未知量。

另一个方法是使用 **Huber Loss**：

$$
L_H(r,\delta) =\left\{
\begin{aligned}
&r^2/2 & {\rm if} \; |r| \leq \delta \\
&\delta |r| - \delta^2/2 & {\rm if} \; |r| > \delta 
\end{aligned}
\right.
$$

## Ridge regression

为了鼓励参数较小使得拟合更平滑，我们使用一个zero-mean的**高斯先验** （对比uniform先验）

$$
p(\boldsymbol{\mathbf{w}}) = \prod_j \mathcal{N}(w_j \,|\, 0,\tau^2)
$$

则 MAP 估计问题为

$$
\arg \max_{\boldsymbol{\mathbf{w}}} \sum_{i=1}^{N} \log \mathcal{N}(y_i \,|\, w_0 + \boldsymbol{\mathbf{w}}^T \boldsymbol{\mathbf{x}}_i, \sigma^2) + \sum_{j=1}^D \log \mathcal{N}(w_j \,|\, 0, \tau^2)
$$

即损失函数为

$$
J(\boldsymbol{\mathbf{w}}) = \frac{1}{N} \sum_{i=1}^N (y_i -(w_0 + \boldsymbol{\mathbf{w}}^T \boldsymbol{\mathbf{x}}_i))^2 + \lambda ||\boldsymbol{\mathbf{w}}||_2^2
$$

其中 $\lambda \triangleq \sigma^2 / \tau^2$

解为

$$
\hat{\boldsymbol{\mathbf{w}}}_{ridge} = (\lambda \boldsymbol{\mathbf{I}}_D + \boldsymbol{\mathbf{X}}^T\boldsymbol{\mathbf{X}})^{-1}\boldsymbol{\mathbf{X}}^T\boldsymbol{\mathbf{y}}
$$

### Numerivally stable computation

我们可以把先验当作对训练数据的增加。$p(\boldsymbol{\mathbf{w}}) = \mathcal{N}(\boldsymbol{\mathbf{0}}, \boldsymbol{\mathbf{\Lambda}}^{-1})$，其中$\boldsymbol{\mathbf{\Lambda}} = (1/\tau^2)\boldsymbol{\mathbf{I}}$。对原训练数据进行中心化从而不需要 $w_0$。

$$
\tilde{\boldsymbol{\mathbf{X}}} = \left(
 \begin{matrix}
   \boldsymbol{\mathbf{X}} / \sigma  \\
   \sqrt{\boldsymbol{\mathbf{\Lambda}}}
  \end{matrix}
\right) , 
\tilde{\boldsymbol{\mathbf{y}}} = \left(
 \begin{matrix}
   \boldsymbol{\mathbf{y}} / \sigma  \\
   \boldsymbol{\mathbf{0}}_{D \times 1}
  \end{matrix}
\right)
$$

其中 $\sqrt{\boldsymbol{\mathbf{\Lambda}}}$ 是 **Cholesky decomposition** of $\boldsymbol{\mathbf{\Lambda}}$ ，$\boldsymbol{\mathbf{\Lambda}} = \sqrt{\boldsymbol{\mathbf{\Lambda}}} \sqrt{\boldsymbol{\mathbf{\Lambda}}}^T$

现在 $\tilde{\boldsymbol{\mathbf{X}}} \in \mathbb{R}^{(N+D)\times D}$

所以类似 OLS ：

$$
\tilde{\boldsymbol{\mathbf{w}}}_{ridge} = (\tilde{\boldsymbol{\mathbf{X}}}^T \tilde{\boldsymbol{\mathbf{X}}})^{-1}\tilde{\boldsymbol{\mathbf{X}}}^T \tilde{\boldsymbol{\mathbf{y}}}
$$

现在对$\tilde{\boldsymbol{\mathbf{X}}}$ 做** QR decomposition**

$$
\tilde{\boldsymbol{\mathbf{X}}} = \boldsymbol{\mathbf{Q}}\boldsymbol{\mathbf{R}}
$$

其中 $\boldsymbol{\mathbf{Q}}$ 是正交矩阵，$\boldsymbol{\mathbf{R}}$ 是上三角矩阵。

所以

$$
(\tilde{\boldsymbol{\mathbf{X}}}^T \tilde{\boldsymbol{\mathbf{X}}})^{-1} = (\boldsymbol{\mathbf{R}}^T \boldsymbol{\mathbf{Q}}^T \boldsymbol{\mathbf{QR}})^{-1} = (\boldsymbol{\mathbf{R}}^T\boldsymbol{\mathbf{R}})^{-1} = \boldsymbol{\mathbf{R}}^{-1}\boldsymbol{\mathbf{R}}^{-T}
$$

因此

$$
\tilde{\boldsymbol{\mathbf{w}}}_{ridge} = \boldsymbol{\mathbf{R}}^{-1}\boldsymbol{\mathbf{R}}^{-T}  \boldsymbol{\mathbf{R}}^T \boldsymbol{\mathbf{Q}}^T \tilde{\boldsymbol{\mathbf{y}}} = \boldsymbol{\mathbf{R}}^{-1} \boldsymbol{\mathbf{Q}} \tilde{\boldsymbol{\mathbf{y}}}
$$

现在 $\boldsymbol{\mathbf{R}} $ 是一个容易求逆的矩阵。求QR分解的时间复杂度为$O(ND^2)$

如果$D \gg N$ ，应该用 SVD 分解。$\boldsymbol{\mathbf{X}} = \boldsymbol{\mathbf{USV}}^T$，记 $\boldsymbol{\mathbf{Z}} = \boldsymbol{\mathbf{UD}}$

$$
\hat{\boldsymbol{\mathbf{w}}}_{ridge} = \boldsymbol{\mathbf{V}}(\boldsymbol{\mathbf{Z}}^T\boldsymbol{\mathbf{Z}}+\lambda\boldsymbol{\mathbf{I}}_N)^{-1}\boldsymbol{\mathbf{Z}}^T\boldsymbol{\mathbf{y}}
$$

时间复杂度为 $O(DN^2)$ 。

### Connection with PCA

上式又可以写为

$$
\hat{\boldsymbol{\mathbf{w}}}_{ridge} = \boldsymbol{\mathbf{V}} (\boldsymbol{\mathbf{S}}^2 + \lambda \boldsymbol{\mathbf{I}})^{-1} \boldsymbol{\mathbf{SU}}^T \boldsymbol{\mathbf{y}}
$$

所以预测值为

$$
\begin{align}
\begin{split}
\hat{\boldsymbol{\mathbf{y}}} &=  \boldsymbol{\mathbf{X}} \hat{\boldsymbol{\mathbf{w}}}_{ridge} = \boldsymbol{\mathbf{USV}}^T \boldsymbol{\mathbf{V}} (\boldsymbol{\mathbf{S}}^2 + \lambda \boldsymbol{\mathbf{I}})^{-1} \boldsymbol{\mathbf{SU}}^T \boldsymbol{\mathbf{y}} \\
&= \boldsymbol{\mathbf{US}}(\boldsymbol{\mathbf{S}}^2 + \lambda \boldsymbol{\mathbf{I}})^{-1} \boldsymbol{\mathbf{SU}}^T \boldsymbol{\mathbf{y}} \\
&= \boldsymbol{\mathbf{U}} \tilde{\boldsymbol{\mathbf{S}}} \boldsymbol{\mathbf{U}}^T \boldsymbol{\mathbf{y}} = \sum_{j=1}^{D} \boldsymbol{\mathbf{u}}_j \tilde{S}_{jj} \boldsymbol{\mathbf{u}}_j^T \boldsymbol{\mathbf{y}}
\end{split}
\end{align}
$$

其中

$$
\tilde{S}_{jj} = [\boldsymbol{\mathbf{S}}(\boldsymbol{\mathbf{S}}^2 + \lambda I)^{-1}\boldsymbol{\mathbf{S}}]_{jj} = \frac{\sigma_j^2}{\sigma_j^2+\lambda}
$$

其中 $\sigma_j$ 是 $\boldsymbol{\mathbf{X}}$ 的奇异值。

因此

$$
\hat{\boldsymbol{\mathbf{y}}} = \boldsymbol{\mathbf{X}} \hat{\boldsymbol{\mathbf{w}}}_{ridge} = \sum_{j=1}^D \boldsymbol{\mathbf{u}}_j \frac{\sigma_j^2}{\sigma_j^2+\lambda} \boldsymbol{\mathbf{u}}_j^T \boldsymbol{\mathbf{y}}
$$

作为对比 OLS 的预测为

$$
\hat{\boldsymbol{\mathbf{y}}} = \boldsymbol{\mathbf{X}} \hat{\boldsymbol{\mathbf{w}}}_{OLS} = \sum_{j=1}^D \boldsymbol{\mathbf{u}}_j \boldsymbol{\mathbf{u}}_j^T \boldsymbol{\mathbf{y}}
$$

因此，定义**自由度 degrees of freedom** 为

$$
dof(\lambda) = \sum_{j=1}^D \frac{\sigma_j^2}{\sigma_j^2+\lambda}
$$

在 OLE 中，对 $\boldsymbol{\mathbf{w}}$ 最不确定的方向是被 $\sigma^2 (\boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{u}})^{-1}$ 的最小特征值对应的特征向量所决定的。而岭回归下，小的奇异值对应的方向会被对 $\boldsymbol{\mathbf{w}}$ 的先验影响（shrink）而降低不确定度。这种现象称为 **shrinkage**，类似于内嵌了 PCA。

## Bayesian linear regression

以上我们都在做点估计，有时我们想做一个完整的后验估计。

### Computing the posterior

假设数据中心化

$$
\begin{align}
\begin{split}
p(\boldsymbol{\mathbf{w}} \,|\, \boldsymbol{\mathbf{X}} , \boldsymbol{\mathbf{y}} , \sigma^2) &\propto \mathcal{N}(\boldsymbol{\mathbf{w}} \,|\, \boldsymbol{\mathbf{w}}_0, \boldsymbol{\mathbf{V}}_0) \mathcal{N}(\boldsymbol{\mathbf{y}} \,|\, \boldsymbol{\mathbf{Xw}}, \sigma^2 \boldsymbol{\mathbf{I}}_N) = \mathcal{N}(\boldsymbol{\mathbf{w}} \,|\, \boldsymbol{\mathbf{w}}_N , \boldsymbol{\mathbf{V}}_N) \\

\boldsymbol{\mathbf{w}}_N &= \boldsymbol{\mathbf{V}}_N \boldsymbol{\mathbf{V}}_0^{-1} \boldsymbol{\mathbf{w}}_0 + \frac{1}{\sigma^2} \boldsymbol{\mathbf{V}}_N \boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{y}} \\

\boldsymbol{\mathbf{V}}_N^{-1} &= \boldsymbol{\mathbf{V}}_0^{-1} + \frac{1}{\sigma^2}\boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{X}}  \\

\boldsymbol{\mathbf{V}}_N &= \sigma^2 (\sigma^2\boldsymbol{\mathbf{V}}_0^{-1} + \boldsymbol{\mathbf{X}}^T\boldsymbol{\mathbf{X}})^{-1}

\end{split}
\end{align}
$$

如果 $\boldsymbol{\mathbf{w}}_0 = \boldsymbol{\mathbf{0}} \, , \, \boldsymbol{\mathbf{V}}_0=\tau^2\boldsymbol{\mathbf{I}}$ ，则*后验均值*即为岭回归。

### Computing the posterior predictive

$$
\begin{align}
\begin{split}
p(y \,|\, \boldsymbol{\mathbf{x}}, \mathcal{D}, \sigma^2) &= \int \mathcal{N}(y \,|\, \boldsymbol{\mathbf{x}}^T\boldsymbol{\mathbf{w}},\sigma^2) \mathcal{N}(\boldsymbol{\mathbf{w}} \,|\,\boldsymbol{\mathbf{w}}_N,\boldsymbol{\mathbf{V}}_N) d\boldsymbol{\mathbf{w}} \\
&= \mathcal{N}(y \,|\, \boldsymbol{\mathbf{w}}_N^T\boldsymbol{\mathbf{x}},\sigma^2_N(\boldsymbol{\mathbf{x}})) \\
\sigma^2_N(\boldsymbol{\mathbf{x}}) &= \sigma^2 + \boldsymbol{\mathbf{x}}^T \boldsymbol{\mathbf{V}}_N \boldsymbol{\mathbf{x}} 
\end{split}
\end{align}
$$

对比 plugin approximation，例如岭回归，MLE等点估计

$$
p(y \,|\, \boldsymbol{\mathbf{x}}, \mathcal{D}, \sigma^2) \approx \int \mathcal{N}(y \,|\, \boldsymbol{\mathbf{x}}^T\boldsymbol{\mathbf{w}},\sigma^2) \delta_{\hat{\boldsymbol{\mathbf{w}}}}(\boldsymbol{\mathbf{w}}) d\boldsymbol{\mathbf{w}} = p(y \,|\, \boldsymbol{\mathbf{x}}, \hat{\boldsymbol{\mathbf{w}}}, \sigma^2)
$$

前者的方差和预测数据有关，预测数据和训练数据相差越大，方差越大。

<center>
<img src="https://ws2.sinaimg.cn/large/006tKfTcly1g0gcfpczp5j30yv0u046t.jpg" />
Figure1 Plugin approximation VS Posterior predictive
</center>

### Bayesian inference when $\sigma^2$ is unknown

上面我们假设 $\sigma^2$ 已知，据此求 $p(\boldsymbol{\mathbf{w}} \,|\,  \mathcal{D}, \sigma^2)$，下面假设 $\sigma^2$ 未知，计算 $p(\boldsymbol{\mathbf{w}} , \sigma^2 \,|\,  \mathcal{D})$

#### Conjugate prior

似然:

$$
p(\boldsymbol{\mathbf{y}} \,|\, \boldsymbol{\mathbf{X}},\boldsymbol{\mathbf{w}},\sigma^2) = \mathcal{N}(\boldsymbol{\mathbf{y}} \,|\, \boldsymbol{\mathbf{Xw}}, \sigma^2\boldsymbol{\mathbf{I}}_N)
$$

共轭先验：

$$
\begin{align}
\begin{split}
p(\boldsymbol{\mathbf{w}},\sigma^2) &=& NIG(\boldsymbol{\mathbf{w}}, \sigma^2 \,|\, \boldsymbol{\mathbf{w}}_0, \boldsymbol{\mathbf{V}}_0, a_0, b_0) \\
&\triangleq& \mathcal{N}(\boldsymbol{\mathbf{w}} \,|\, \boldsymbol{\mathbf{w}}_0,\sigma^2\boldsymbol{\mathbf{V}}_0) IG(\sigma^2 \,|\, a_0,b_0) \\
&= \;\; &\frac{b_0^{a_0}}{(2\pi)^{D/2} |\boldsymbol{\mathbf{V}}_0|^{1/2} \Gamma(a_0)} (\sigma^2)^{-(a_0+(D/2)+1)}  \\
&&\times \exp  \lbrack -\frac{(\boldsymbol{\mathbf{w}} - \boldsymbol{\mathbf{w}}_0)^T \boldsymbol{\mathbf{V}}_0^{-1}(\boldsymbol{\mathbf{w}}-\boldsymbol{\mathbf{w}}_0)+2b_0}{2\sigma^2} \rbrack
\end{split}
\end{align}
$$

后验：

$$
\begin{align}
\begin{split}
p(\boldsymbol{\mathbf{w}},\sigma^2 \,|\, \mathcal{D}) &= NIG(\boldsymbol{\mathbf{w}},\sigma^2 \,|\, \boldsymbol{\mathbf{w}}_N, \boldsymbol{\mathbf{V}}_N, a_N, b_N) \\

\boldsymbol{\mathbf{w}}_N &= \boldsymbol{\mathbf{V}}_N(\boldsymbol{\mathbf{V}}^{-1}\boldsymbol{\mathbf{w}}_0+\boldsymbol{\mathbf{X}}^T\boldsymbol{\mathbf{y}}) \\

\boldsymbol{\mathbf{V}}_N &= (\boldsymbol{\mathbf{V}}_0^{-1} + \boldsymbol{\mathbf{X}}^T\boldsymbol{\mathbf{X}})^{-1} \\

a_N &= a_0 + n/2 \\

b_N &= b_0 + \frac{1}{2}(\boldsymbol{\mathbf{w}}_0^T\boldsymbol{\mathbf{V}}_0^{-1}\boldsymbol{\mathbf{w}}_0 + \boldsymbol{\mathbf{y}}^T\boldsymbol{\mathbf{y}} - \boldsymbol{\mathbf{w}}_N\boldsymbol{\mathbf{V}}_N^{-1}\boldsymbol{\mathbf{w}}_N)
\end{split}
\end{align}
$$

后验边缘分布为

$$
\begin{align}
\begin{split}
p(\sigma^2 \,|\, \mathcal{D}) &= IG(a_N,b_N) \\
p(\boldsymbol{\mathbf{w}} \,|\, \mathcal{D}) &= \mathcal{T}(\boldsymbol{\mathbf{w}}_N, \frac{b_N}{a_N}\boldsymbol{\mathbf{V}}_N,2a_N)
\end{split}
\end{align}
$$

预测：给 $m$ 个测试数据 $\tilde{\boldsymbol{\mathbf{X}}}$

$$
p(\tilde{\boldsymbol{\mathbf{y}}} \,|\, \tilde{\boldsymbol{\mathbf{X}}}, \mathcal{D}) = \mathcal{T}(\tilde{\boldsymbol{\mathbf{y}}} \,|\, \tilde{\boldsymbol{\mathbf{X}}} \boldsymbol{\mathbf{w}}_N, \frac{b_N}{a_N}(\boldsymbol{\mathbf{I}}_m + \tilde{\boldsymbol{\mathbf{X}}}\boldsymbol{\mathbf{V}}_N \tilde{\boldsymbol{\mathbf{X}}}^T), 2a_N)
$$

方差项有两部分，第一部分表示测量噪声，第二部分是$\boldsymbol{\mathbf{w}}$ 的不确定性，或者说测试数据和训练数据的接近程度。

#### Uninformative prior

对$\sigma^2$ 的无信息先验：$a_0=b_0=0$。**g-prior** ：$\boldsymbol{\mathbf{w}}_0 = \boldsymbol{\mathbf{0}} \, , \, \boldsymbol{\mathbf{V}}_0 = g(\boldsymbol{\mathbf{X}}^T\boldsymbol{\mathbf{X}})^{-1}$ g 是个正值。$g = \infin$ 时即对 $\boldsymbol{\mathbf{w}}$ 无信息先验。

此时和 $\boldsymbol{\mathbf{w}}_0 = \boldsymbol{\mathbf{0}} , \boldsymbol{\mathbf{v}}_0 = \infin \boldsymbol{\mathbf{I}}, a_0 = 0, b_0 = 0$ 等价，得到 $p(\boldsymbol{\mathbf{w}},\sigma^2) \propto \sigma^{-(D+2)}$

或者，可以用半共轭先验 $p(\boldsymbol{\mathbf{w}},\sigma^2) = p(\boldsymbol{\mathbf{w}}) p(\sigma^2) $ ，各自取无信息极限，得到 $p(\boldsymbol{\mathbf{w}},\sigma^2) \propto \sigma^{-2}$ ，这等价于 NIG先验 $\boldsymbol{\mathbf{w}}_0 = \boldsymbol{\mathbf{0}} , \boldsymbol{\mathbf{v}}_0 = \infin \boldsymbol{\mathbf{I}}, a_0 = -D/2f, b_0 = 0$

此时后验为：

$$
\begin{align}
\begin{split}
p(\boldsymbol{\mathbf{w}},\sigma^2 \,|\, \mathcal{D}) &= NIG(\boldsymbol{\mathbf{w}}, \sigma^2 \,|\, \boldsymbol{\mathbf{w}}_N, \boldsymbol{\mathbf{V}}_N, a_N, b_N) \\

\boldsymbol{\mathbf{w}}_N &= \hat{\boldsymbol{\mathbf{w}}}_{mle} = (\boldsymbol{\mathbf{X}}^T\boldsymbol{\mathbf{X}})^{-1}\boldsymbol{\mathbf{X}}^T\boldsymbol{\mathbf{y}} \\

\boldsymbol{\mathbf{V}}_N &= (\boldsymbol{\mathbf{X}}^T \boldsymbol{\mathbf{X}})^{-1} \\

a_N &= \frac{N-D}{2} \\
b_N &= \frac{s^2}{2} \\

s^2 & \triangleq (\boldsymbol{\mathbf{y}} - \boldsymbol{\mathbf{X}} \hat{\boldsymbol{\mathbf{w}}}_{mle})^T (\boldsymbol{\mathbf{y}} - \boldsymbol{\mathbf{X}} \hat{\boldsymbol{\mathbf{w}}}_{mle})
\end{split}
\end{align}
$$

则对权重的边缘分布为

$$
p(\boldsymbol{\mathbf{w}} \,|\, \mathcal{D}) = \mathcal{T}(\boldsymbol{\mathbf{w}} \,|\, \hat{\boldsymbol{\mathbf{w}}}, \frac{s^2}{N-D}\boldsymbol{\mathbf{C}}, N-D)  \\
\boldsymbol{\mathbf{C}} = (\boldsymbol{\mathbf{X}}^T\boldsymbol{\mathbf{X}})^{-1}
$$

在这种情况下，贝叶斯和频率达到了一致的结果，credible intervals 和 confidence interval 一致。
