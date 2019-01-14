---
layout:     post
title:      SimpleNet Inplementation
date:       2019-01-14
author:     Fanvel Chen
header-img: img/posts/implementaion_img.jpg
catalog:    true
mathjax:    true
tags:
    - deep learning inplementation
---

## 一个简单实现

先实现一个两层的简单网络。

### forward

$X $ : [p_input, ins_num]

$y$ : [p_output, ins_num]

$W_1$ : [p_input, p_hidden]

$B_1$ :  [p_hidden, 1]

$u_1 = W_1^T X + B1$ : [p_hidden, ins_num]

$h_1 = ReLU(u_1)$ : [p_hidden, ins_num]

---

$W_2$ : [p_hidden, p_output]

$B_2$ :  [p_output, 1]

$u_2 = W_2^T X + B2$ : [p_output, ins_num]

$o = \delta(u_2)$ : [p_out, ins_num]

---

$J =\frac{1}{2} \frac{1}{m}  \sum_{i=1}^{m} -(y_i \log o_i + (1-y_i) \log (1-o_i) )$

### backward

$\frac{\partial J}{ \partial o} = \frac{1}{m} ( \frac{1-y}{1-o} + \frac{y}{o}  )$ : [p_output, ins_num]

$\frac{\partial o}{\partial u_2} = o(1-o)$ : [p_output, ins_num]

$\frac{\partial J}{ \partial u_2} = \frac{\partial J}{ \partial o} \odot \frac{\partial o}{\partial u_2} = \frac{1}{m} (o - y)$ :  [p_output, ins_num]

$\frac{\partial u_2}{\partial W_2} = h1^T$ : [ ins_num, p_hidden]

$\frac{\partial u_2}{\partial B_2} = \boldsymbol{1}$ : [ p_output, 1]

$\frac{\partial u_2}{\partial h_1} = W_2^T$ : [p_output, p_hidden]

$\frac{\partial h_1}{\partial u_1} = u1>0 ? 1 , 0$ : [p_hidden, ins_num]

$\frac{\partial u_1}{\partial W_1} = X^T$ : [ins_num, p_input]

$\frac{\partial u_2}{\partial B_1} = \boldsymbol{1}$ : [ p_hidden, 1]

所以

对$B_2$

$$
\begin{align}
\begin{split}
\frac{\partial J}{\partial B_2} &= \frac{\partial J}{ \partial o}  \frac{\partial o}{\partial u_2} \frac{\partial u_2}{\partial B_2} \\
&= \frac{\partial J}{ \partial u_2} \frac{\partial u_2}{\partial B_2} \\
&\in Ave_{over\,m} ( \mathbb{R}^{p\_out \times m} ) \odot \mathbb{R}^{p\_out \times 1} \\
\end{split}
\end{align}
$$

对$W_2$

$$
\begin{align}
\begin{split}
\frac{\partial J}{\partial W_2} &= \frac{\partial J}{ \partial o}  \frac{\partial o}{\partial u_2} \frac{\partial u_2}{\partial W_2} \\
& = \frac{\partial J}{ \partial u_2} \frac{\partial u_2}{\partial W_2} \\
&\in  Ave_{over\,m} ( \mathbb{R}^{m \times p\_out \times 1} span \mathbb{R}^{m \times 1 \times p\_hidden} )\\
&\in Ave_{over\,m} ( \mathbb{R}^{p\_out \times m} \times \mathbb{R}^{m  \times p\_hidden})
\end{split}
\end{align}
$$

对$B_1$

$$
\begin{align}
\begin{split}
\frac{\partial J}{\partial B_1} &= \frac{\partial J}{ \partial u_2}  \frac{\partial u_2}{\partial h_1} \frac{\partial h_1}{\partial u_1} \frac{\partial u_1}{\partial B_1} \\
&\in Ave_{over \, m}( ( (\mathbb{R}^{p\_out \times m})^T \times \mathbb{R}^{p\_out \times p\_hidden} )^T \odot \mathbb{R}^{p\_hidden \times m}) \odot \mathbb{R}^{p\_hidden \times 1}
\end{split}
\end{align}
$$

对$W_1$

$$
\begin{align}
\begin{split}
\frac{\partial J}{\partial W_1} &= \frac{\partial J}{ \partial u_2}  \frac{\partial u_2}{\partial h_1} \frac{\partial h_1}{\partial u_1} \frac{\partial u_1}{\partial W_1} \\
&\in Ave_{over \, m}( ( ( (\mathbb{R}^{p\_out \times m})^T \times \mathbb{R}^{p\_out \times p\_hidden} )^T \odot \mathbb{R}^{p\_hidden \times m}) \times \mathbb{R}^{m \times p\_in}) \\
&\in Ave_{over \, m}( \mathbb{R}^{p\_hidden \times m} \times \mathbb{R}^{m \times p\_in} )
\end{split}
\end{align}
$$


### Net Code
```python
class simpleNet():

    def __init__(self,input_size,hidden_size,output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.train = True

        self.W1 = np.zeros((input_size,hidden_size))
        self.B1 = np.zeros((hidden_size,1))

        self.W2 = np.zeros((hidden_size,output_size))
        self.B2 = np.zeros((output_size,1))

        self.parametersInitialization()
       

    def parametersInitialization(self):
        self.W1 = np.random.rand(*self.W1.shape) - 0.5
        self.W2 = np.random.rand(*self.W2.shape) - 0.5

        self.B1 = np.random.rand(*self.B1.shape)
        self.B2 = np.random.rand(*self.B2.shape)

    def forwardPropagate(self,x):
        if self.train:
            self.x = x
        u1 = np.dot(self.W1.T, x) + self.B1
        if self.train:
            self.u1 = u1
        h1 = relu(u1)
        if self.train:
            self.h1 = h1
        u2 = np.dot(self.W2.T, h1) + self.B2
        o = sigmoid(u2)
        return o

    def backwardPropagate(self,y,pred,loss_function):
        print_flag = False
        ins_num = y.shape[1]

        if loss_function == 'crossEntropy':
            dlossdu2 = (-(1-pred) * y + pred * ( 1 - y))
        
        if loss_function == 'mse':
            dlossdo = pred - y                                      # [p_out, ins_sum]
            if print_flag:
                print("dJdo")
                print(dlossdo.shape)
                print(dlossdo)
            dodu2 = pred * (1 - pred)                               # [p_out, ins_num]
            if print_flag:
                print("dodu2")
                print(dodu2.shape)
                print(dodu2)
            dlossdu2 = dlossdo * dodu2                              # [p_out, ins_num]
            if print_flag:
                print("dJdu2 = dJdo * dodu2")
                print(dlossdu2.shape)
                print(dlossdu2)

        du2dw2 = self.h1                                            # [p_hidden, ins_num]
        if print_flag:
            print("du2dw2 = h1")
            print(du2dw2.shape)
            print(du2dw2)
        self.db2 = np.mean(dlossdu2,axis=1)[:,np.newaxis]           # [p_out, 1]
        if print_flag:
            print("db2")
            print(self.db2.shape)
            print(self.db2)
        self.dw2 = np.dot(du2dw2,dlossdu2.T)/ins_num
        if print_flag:
            print("dw2")
            print(self.dw2.shape)
            print(self.dw2)

        du2dh1 = self.W2                                            # [p_hidden, p_out]
        if print_flag:
            print("du2dh1")
            print(du2dh1.shape)
            print(du2dh1)
        dlossdh1 = np.dot(dlossdu2.T,du2dh1.T)                     # [ins_num, p_hidden]   = dlossdu2^T [ins_num, p_out] dot du2dh1^T [p_out, p_hidden]
        if print_flag:
            print("dJdh1 = dJdu2 dot du2dh1")
            print(dlossdh1.shape)
            print(dlossdh1)
        dh1du1 = np.where(self.u1>0,1,0).T                          # [ins_num, p_hidden]
        if print_flag:
            print("dh1du1")
            print(dh1du1.shape)
            print(dh1du1)
        dlossdu1 = (dlossdh1 * dh1du1).T                            # [p_hidden, ins_num] = (dlossdh1 [ins_num, p_hidden] * dh1du1 [ins_num, p_hidden])^T

        du1dw1 = self.x                                             # [p_in, ins_num]

        self.db1 = np.mean(dlossdu1,axis=1)[:,np.newaxis]           # [p_hidden, 1]
        self.dw1 = np.dot(du1dw1, dlossdu1.T ) / ins_num
    
    def updateParameters(self,lr):
        self.W1 -= lr * self.dw1
        self.W2 -= lr * self.dw2
        self.B1 -= lr * self.db1
        self.B2 -= lr * self.db2

    def run(self,x,y,lr,loss_function):
        pred = self.forwardPropagate(x)
        if self.train:
            self.backwardPropagate(y,pred,loss_function)
            self.updateParameters(lr)
        return pred

    def trainmode(self):
        self.train = True
    
    def evalmode(self):
        self.train = False
        
    def displayPara(self):
        print(self.W1)
        print(self.B1)
        print(self.W2)
        print(self.B2)

def relu(x):
    return np.where(x>0, x, 0)  

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
```

### Data

```python
def data_generator(num,input_size = 2):
    x = np.random.rand(input_size,num)
    y = np.zeros((1,num))
    for i in range(num):
        if ((0.5 - x[0,i])**2 + (0.5 - x[1,i])**2) <= (0.35*0.35):
            y[0,i] = 1
    return x,y
```

### main

```python
def main():
    import matplotlib.pyplot as plt 
    import time
    
    net = simpleNet(2,10,1)
    
    x_test,y_test = data_generator(400)
    acc_list = []
    lr = 0.001
    for episode in range(1000 * 800):
        x,y = data_generator(10)
        net.run(x,y,lr,'crossEntropy')
        
        if episode % 1000 == 0:
            net.evalmode()

            
            pred = net.run(x_test,None,None,None)
            pred = np.where(pred > 0.5, 1, 0)
            acc = np.mean(pred == y_test)
            acc_list.append(acc)
            
            if episode % 30000 == 0:
                x_test_0 = x_test[:,y_test[0]==0]
                x_test_1 = x_test[:,y_test[0]==1]
                x_test_pred_0 = x_test[:,pred[0]==0]
                x_test_pred_1 = x_test[:,pred[0]==1]

                plt.subplot(1,2,1)
                plt.scatter(x_test_0[0,:],x_test_0[1,:],label="gt0")
                plt.scatter(x_test_1[0,:],x_test_1[1,:],label="gt1")
                plt.legend()
                plt.subplot(1,2,2)
                plt.scatter(x_test_pred_0[0,:],x_test_pred_0[1,:],label="pred0")
                plt.scatter(x_test_pred_1[0,:],x_test_pred_1[1,:],label="pred1")
                plt.legend()
                plt.show()
                
            net.trainmode()

    plt.plot(acc_list)
    plt.show()
```


### Test Acc

![acc](https://ws2.sinaimg.cn/large/006tNc79ly1fz6h0z6xe6j30gv0c3q3c.jpg)

### Visualization

![visual1](https://ws4.sinaimg.cn/large/006tNc79ly1fz6hka70kvj30hs0fvq4t.jpg)

![visual2](https://ws2.sinaimg.cn/large/006tNc79ly1fz6hlc44j8j30hs0fvabv.jpg)

![visual3](https://ws1.sinaimg.cn/large/006tNc79ly1fz6hlvserjj30hs0fvtan.jpg)

![visual4](https://ws1.sinaimg.cn/large/006tNc79ly1fz6hmj145tj30hs0fvq4p.jpg)
