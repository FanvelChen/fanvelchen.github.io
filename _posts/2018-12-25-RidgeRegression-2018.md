---
layout:     post
title:      岭回归 Ridge Regression
date:       2018-12-25
author:     Fanvel Chen
header-img: img/posts/rigde_regression_bg.jpg
catalog: true
mathjax: true
tags:
    - machine learning
---

# 岭回归（Ridge Regression）

## 生成数据
生成三维独立 $[0,1]$ 均匀分布特征 $x1,x3,x5$ ，ground truth为：
$$
y = 4 \times x_1 + 2 \times x_2 + \varepsilon \\ \tag{1}
$$
其中 $\varepsilon \sim \mathcal{N}(0,1)$ 。
$x_2$ 与 $x_1^2$ 正相关，并归一化。$x_4$ 和 $x_3$ 负相关，并归一化。$x_5$ 为无关特征， $x_6​$ 为偏置项

```python
def geneData(ins_num):
	x1 = np.random.rand(1,ins_num)   
	x2 = (x1**2*10+np.random.rand(1,ins_num))/11  
	x3 = np.random.rand(1,ins_num)
	x4 = -(x3*2+np.random.rand(1,ins_num))/3
	x5 = np.random.rand(1,ins_num)
	x6 = np.ones(ins_num).reshape(1,ins_num)
	y = 4 * x1 + 2 * x3 + np.random.normal(size=ins_num)
	data = np.concatenate((x1,x3,x5,x2,x4,x6),axis = 0)
	actual_data = np.concatenate((x1,x3,x5,x6),axis = 0)
	return actual_data,data,y
```

## 数据预处理
### 标准化（normally standardized）
把每一维数据标准化到 $[0,1]$ 的范围，因为对系数 $\beta$ 进行 $L2$ 范数限制下，不同纬度不同的量纲会带来权值比例的失衡。

>The ridge solutions are not equivariant under scaling of the inputs, and so one normally standardizes the inputs

### 中心化（centered）
对每个数据的每一维的取值，都减去这一维的均值

$$
x_{ij} - \bar{x}_j  \tag{2}
$$

#### 原因：
原优化目标函数为

$$
\min L(\beta) = \sum_{i=1}^{N} \left( y_i - \beta_0 - \sum_{j=1}^p x_{ij} \beta_j  \right)^2 +  \lambda \sum_{j=1}^p \beta_j^2   \tag{3}
$$

可以写为

$$
\min L(\beta) = \sum_{i=1}^{N} \left( y_i - \beta_0 - \sum_{j=1}^p \bar{x}_j \beta_j + \sum_{j=1}^p (x_{ij} -\bar{x}_j)\beta_j  \right)^2 +  \lambda \sum_{j=1}^p \beta_j^2   \tag{4}
$$

所以对 $\beta$ 做代换成为 $\beta^{c}$

$$
\begin{eqnarray*}
\beta_0^c &=& \beta_0 + \sum_{j=1}^p \bar{x}_j \beta_j   \tag{5}  \\
\beta_j^c &=& \beta_j \quad j = 1,2,\cdots,p  \tag{6}
\end{eqnarray*}
$$

即

$$
\min L(\beta) = \sum_{i=1}^{N} \left( y_i - \beta_0^c - \sum_{j=1}^p (x_{ij} - \bar{x}_j) \beta_j^c  \right)^2 +  \lambda \sum_{j=1}^p (\beta_j^c)^2   \tag{7}
$$

则

$$
\begin{align}
\begin{split}
\mathbb{E}[\sum_{j=1}^p(x_{ij}-\bar{x}_j)\beta_j^c] &= \sum_{j=1}^p (\mathbb{E}(x_{ij}) - \bar{x}_j)\beta_j^c  \\
&= \sum_{j=1}^p(\bar{x}_j - \bar{x}_j)\beta_j^c   \\
&= 0   
\end{split}   \tag{8}
\end{align}
$$

所以可以用 $\bar{y}$ 作为 $\beta_0^c$ 的估计

#### $y$ 是否需要中心化

假设用中心化的 $\beta$ 估计为 $\hat{\beta'}$ :

$$
\begin{eqnarray*}
\hat{\beta} &=& (X^TX+\lambda I)^{-1}X^Ty  \tag{9}  \\
\hat{\beta'} &=& (X^T X+\lambda I)^{-1}X^T(y-\textbf{1} \bar{y})  \tag{10}
\end{eqnarray*}
$$

其中 $\textbf{1} = (1,1,\cdots,1)^T \in \mathbb{R}^{N \times 1}$，而 $X$ 为已经中心化的样本数据。

$$
\begin{align}
\begin{split}
\hat{\beta} - \hat{\beta'} &= (X^TX + \lambda I)^{-1}X^T \textbf{1}\bar{y} \\
&= (X^TX+\lambda I)^{-1} (\sum_{i=1}^N{x}_1,\sum_{i=1}^N{x}_2,\cdots,\sum_{i=1}^N{x}_p)^T \bar{y}  \\
&=(X^TX+\lambda I)^{-1}\textbf{0}\bar{y} \\
&=\textbf{0}
\end{split}   \tag{11}
\end{align}
$$

即 $\hat{\beta} = \hat{\beta'}$ ，$y$ 的中心化没有影响

#### 如果数据不中心化会如何

 则不可以用 $\bar{y}$ 作为 $\beta_0^c$ 的估计，则只能做 $X \in \mathbb{R}^{N \times (p+1)}$ 的整体求解。 




