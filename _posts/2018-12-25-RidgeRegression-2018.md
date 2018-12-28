---
layout:     post
title:      岭回归 Ridge Regression
date:       2018-12-25
author:     Fanvel Chen
header-img: img/posts/rigde_regression_bg.jpg
catalog:    true
mathjax:    true
tags:
    - machine learning
---


# 岭回归（Ridge Regression）

## 1. 基本原理

### 目标函数

通过引入一个参数相关的惩罚项：

$$
\beta^{ridge} = argmin_{\beta} \left\{ \sum_{i=1}^N (y_i - \beta_0 - \sum_{j=1}^p x_{ij}^p \beta )^2 + \lambda \sum_{j=1}^p \beta_{j}^2 \right\}   \tag{1.1}
$$

值得注意的是，在惩罚项中不包含 $\beta_0$ 的截距系数，否则在坐标轴平移的情况下，会带来其他系数的改变。

### 闭式解

$X$ 经过中心化（减去每一维的均值）和标准化（取值在 $[0,1]$ ）后，用 $\bar{y}$  作为 $\beta_0$ 的估计值，$X \in \mathbb{R}^{N \times p}$ 

$$
RSS(\lambda) = (y-X \beta)^T(y-X\beta) + \lambda \beta^T \beta \tag{1.2}
$$

对 $\beta$ 求导等于0可得

$$
\hat{\beta}^{ridge} = (X^TX+\lambda I)^{-1}X^Ty \tag{1.3}
$$

可以防止在 $X^TX$ 不满秩的情况下（即不同特征维度有线性相关性）无法求逆。 

### 一些性质

#### 岭回归系数 $\beta^{ridge}$ 和最小二乘系数 $\beta^{ls}$ 关系

对 $X$ 进行 SVD : $X = USV^T$

则 $y$ 的估计分别为

$$
\begin{align}
\begin{split}
\hat{y}^{ls} &= X \hat{\beta}^{ls} \\
&= X(X^TX)^{-1}X^Ty \\
&=USV^T(VS^TU^TUSV^T)^{-1}VS^TU^Ty \\
&=UU^Ty \\
&= \sum_{j=1}^{p}u_ju_j^Ty 
\end{split} \tag{1.4}
\end{align}
$$

$$
\begin{align}
\begin{split}
\hat{y}^{ridge} &= X \hat{\beta}^{ridge} \\
&= X(X^TX+\lambda I)^{-1}X^Ty \\
&=USV^T(VS^2V^T+\lambda I)^{-1}VS^TU^Ty \\
&=US(S^2+\lambda I)^{-1}SU^Ty \\
&= \sum_{j=1}^{p}u_j\frac{d_j^2}{d_j^2+\lambda}u_j^Ty 
\end{split} \tag{1.5}
\end{align}
$$

相当于估计值在 $X$ 的每个奇异值特征方向上进行 $\frac{d_j^2}{d_j^2+\lambda}$ 的尺度缩小。

$$
X^TX = VS^2V^T  \tag{1.6}
$$

则 $V$ 的列向量 $v_j$ 即 $X$ 的主成分方向， $z_j = Xv_j$ 即 $X$ 投影到 $v_j$ 的表示。

$$
Var(z_j)=Var(Xv_j) = Cov(X,X) = \frac{d_j^2}{N-1}  \tag{1.7}
$$


若 $d_j$ 较大，即该方向显著，该尺度趋近于 $1$；若 $d_j$ 较小，则该方向不重要，该尺度趋近于 $0$ 。而在 $V$ 空间中方差越大的方向 $d_j$ 越大，即越主要（主成分）。即岭回归本身带有 PCA 的特性。

#### 岭回归是对线性的有偏估计

$$
\begin{align}
\begin{split}
\mathbb{E(\hat{\beta})} &= \mathbb{E}\left[(X^TX + \lambda X)^{-1}X^T y \right] \\
&= \mathbb{E}\left[(X^TX + \lambda X)^{-1}X^T X\beta  \right] \\
&= \mathbb{E}\left[(X^TX + \lambda X)^{-1}X^T X\right] \beta
\end{split} \tag{1.8}
\end{align}
$$

当且仅当 $\lambda = 0$ 时 $\mathbb{E}(\hat{\beta}) = \beta$

而高斯-马尔可夫定理指出最小二乘估计是有最小方差的线形无偏估计，在这里岭回归估计作为线性有偏估计，可能存在 $\lambda$ 使得其方差小于最小二乘估计。



## 2. 一个实验

### 生成数据

生成三维独立 $[0,1]$ 均匀分布特征 $x1,x2,x3$ ，ground truth为：

$$
y = 4 \times x_1 + 2 \times x_2 + \varepsilon \\ \tag{2.1}
$$

其中 $\varepsilon \sim \mathcal{N}(0,1)$ 。
$ x_4$ 与 $x_1^2$ 正相关，并归一化。$ x_5 $ 和 $ x_2 $ 负相关，并归一化。$ x_3 $ 为无关特征， $x_6$ 为偏置项

```python
def geneData(ins_num):
	x1 = np.random.rand(1,ins_num) 
	x2 = np.random.rand(1,ins_num)  
	x3 = np.random.rand(1,ins_num)
	x4 = (x1**2*10+np.random.rand(1,ins_num))/11  
	x5 = -(x2*10+np.random.rand(1,ins_num))/11
	x6 = np.ones(ins_num).reshape(1,ins_num)
	y = 4 * x1 + 2 * x2 + np.random.normal(size=ins_num)
	data = np.concatenate((x1,x2,x3,x4,x5,x6),axis = 0)
	actual_data = np.concatenate((x1,x2,x3,x6),axis = 0)
	return actual_data,data,y
```

### 数据预处理

#### 标准化（normally standardized）

把每一维数据标准化到 $[0,1]$ 的范围，因为对系数 $\beta$ 进行 $L2$ 范数限制下，不同纬度不同的量纲会带来权值比例的失衡。

>The ridge solutions are not equivariant under scaling of the inputs, and so one normally standardizes the inputs

### 中心化（centered）
对每个数据的每一维的取值，都减去这一维的均值

$$
x_{ij} - \bar{x}_j  \tag{2.2}
$$

##### 原因：

原优化目标函数为

$$
\min L(\beta) = \sum_{i=1}^{N} \left( y_i - \beta_0 - \sum_{j=1}^p x_{ij} \beta_j  \right)^2 +  \lambda \sum_{j=1}^p \beta_j^2   \tag{2.3}
$$

可以写为

$$
\min L(\beta) = \sum_{i=1}^{N} \left( y_i - \beta_0 - \sum_{j=1}^p \bar{x}_j \beta_j + \sum_{j=1}^p (x_{ij} -\bar{x}_j)\beta_j  \right)^2 +  \lambda \sum_{j=1}^p \beta_j^2   \tag{2.4}
$$

所以对 $\beta$ 做代换成为 $\beta^{c}$

$$
\begin{eqnarray*}
\beta_0^c &=& \beta_0 + \sum_{j=1}^p \bar{x}_j \beta_j   \tag{2.5}  \\
\beta_j^c &=& \beta_j \quad j = 1,2,\cdots,p  \tag{2.6}
\end{eqnarray*}
$$

即

$$
\min L(\beta) = \sum_{i=1}^{N} \left( y_i - \beta_0^c - \sum_{j=1}^p (x_{ij} - \bar{x}_j) \beta_j^c  \right)^2 +  \lambda \sum_{j=1}^p (\beta_j^c)^2   \tag{2.7}
$$

则

$$
\begin{align}
\begin{split}
\mathbb{E}[\sum_{j=1}^p(x_{ij}-\bar{x}_j)\beta_j^c] &= \sum_{j=1}^p (\mathbb{E}(x_{ij}) - \bar{x}_j)\beta_j^c  \\
&= \sum_{j=1}^p(\bar{x}_j - \bar{x}_j)\beta_j^c   \\
&= 0   
\end{split}   \tag{2.8}
\end{align}
$$

所以可以用 $\bar{y}$ 作为 $\beta_0^c$ 的估计

##### $y$ 是否需要中心化

假设用中心化的 $\beta$ 估计为 $\hat{\beta'}$ :

$$
\begin{eqnarray*}
\hat{\beta} &=& (X^TX+\lambda I)^{-1}X^Ty  \tag{2.9}  \\
\hat{\beta'} &=& (X^T X+\lambda I)^{-1}X^T(y-\textbf{1} \bar{y})  \tag{2.10}
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
\end{split}   \tag{2.11}
\end{align}
$$

即 $\hat{\beta} = \hat{\beta'}$ ，$y$ 的中心化没有影响

##### 如果数据不中心化会如何

 则不可以用 $\bar{y}$ 作为 $\beta_0^c$ 的估计，则只能做 $X \in \mathbb{R}^{N \times (p+1)}$ 的整体求解。而截距项作为整体的一个特征去求解时，会带来更大的误差。


```python
def dataCenter(X):
	p = X.shape[0]
	centered_data = []
	for i in range(p-1):
		tmp_mean = np.mean(X[i,:])
		centered_data.append(X[i,:]-tmp_mean)
	centered_data.append(X[-1,:])
	centered_data = np.array(centered_data)
	return centered_data
```

### 解岭回归闭式解
这里采用 SVD 方法求解式

$$
\beta = (X^T X + \lambda I)^{-1} X^T y  \tag{2.12}
$$

将 $X$ 分解为 $USV^T$

$$
\begin{align}
\begin{split}
\beta &= (X^T X + \lambda I)^{-1} X^T y \\
&=(VS^TU^TUSV^T + \lambda I)^{-1} VS^TU^T y \\
&=(VS^2V^T + \lambda I)^{-1} VSU^T y \\
&=V(S^2 + \lambda I)^{-1}V^T VSU^T y \\
&=V(S^2 + \lambda I)^{-1}SU^T y \\
&=V \, diag(\frac{s_1}{s_1^2+\lambda},\cdots,\frac{s_p}{s_p^2+\lambda}) U^T y
\end{split}  \tag{2.13}
\end{align}
$$

```python
def ridgeRegression(X,y,k,centered=True):
	N = y.shape[1]
	p = X.shape[0]
	k = np.asarray(k, dtype=X.dtype).ravel()
	if centered == True:
		X_bias = y 
		beta_0 = np.mean(X_bias).reshape(1,1)
		X = X[:-1,:]
		X = X.transpose()
		U, s, Vt = np.linalg.svd(X, full_matrices=False)
		idx = s > 1e-15
		s = s[idx][:, np.newaxis]
		UTy = np.dot(U.T, y.reshape(N,1))
		d = np.zeros((s.size, k.size), dtype=X.dtype)
		d[idx] = s / (s ** 2 + k)
		d_UT_y = d * UTy
		beta = np.dot(Vt.T, d_UT_y)
		w = np.concatenate((beta,beta_0),axis = 0)
	else:
		X = X.transpose()
		U, s, Vt = np.linalg.svd(X, full_matrices=False)
		idx = s > 1e-15
		s = s[idx][:, np.newaxis]
		UTy = np.dot(U.T, y.reshape(N,1))
		d = np.zeros((s.size, k.size), dtype=X.dtype)
		d[idx] = s / (s ** 2 + k)
		d_UT_y = d * UTy
		w = np.dot(Vt.T, d_UT_y)
	return w
```

### 结果

```shell
RSS by average pred
51.4949184550813
Miracle pred
RSS: 31.499035507123953 	w: [4 2 0 3]
--------------------
actual data
linear:
RSS: 31.474200610988515 	w: [3.97929714 2.12637286 0.03813124 2.99371036]
My Ridge Regression:
RSS: 31.478074112157714 	w: [3.9314478  2.09987432 0.0357562  2.99371036]
Sklearn Ridge Regression:
RSS: 31.478074112157714 	w: [3.9314478  2.09987432 0.0357562  2.99371036]
--------------
all data
linear:
RSS: 31.38720737924729 	w: [ 4.51933687  4.17039656  0.04963582 -0.58202568  2.24772197  2.99371036]
My Ridge Regression:
RSS: 31.49125457515404 	w: [ 3.86828641  1.99646778  0.03546245  0.06938821 -0.11495557  2.99371036]
Sklearn Ridge Regression:
RSS: 31.491254575154056 	w: [ 3.86828641  1.99646778  0.03546245  0.06938821 -0.11495557  2.99371036]
```

### 系数随 $\lambda$ 的变化

![ridge_lambda](https://ws2.sinaimg.cn/large/006tNbRwly1fymd9f4brcj30h30cdgme.jpg)

可以看到 $w4,w5$ 是不稳定的
