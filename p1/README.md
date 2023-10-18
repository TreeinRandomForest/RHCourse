### Recurrent Neural Networks

**Important note 1**: There'll be no focus on organizing code. The goal of these exercises is to implement and understand how these models work not to create well-structured libraries. So some exercises might even be done in one python file.

**Important note 2**: This is numerical code i.e. the central goal of our code is to compute a bunch of numbers. It is hard to debug this kind of code unless you have an expectation of what output you'll get (for example, in scientific computing i.e. physics, chemistry, biology etc. and even in these fields, debugging is tricky). Using tensor shapes and playing around with small examples in ipython/jupyter is crucial. Please do not write code and hope it'll work - be paranoid about checking.

**Suggested reading order**:
* rnn.py
* test_rnn.py
* train.py

**Reference**: [RNN paper](https://apps.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf)

Consider the following problem: Given an array of numbers, compute the sum. A basic loop would look like:

```
data = [...]
sum = 0

for d in data:
    sum = sum + d
```

This can be generalized to:

```
data = [...]
sum = 0

for d in data:
    sum = f(d, sum)
```

where ```f(d, sum) = lambda d, sum: d + sum```.

This version makes a few things explict:
* data is an array of numbers
* the variable sum represents persistent state
* f() updates the state (sum) based on the current state value and the current data value (variable d)

For a neural network, data and state are vectors. Another primitive operation in doing a linear projection and applying a non-linearity i.e.

```python
import numpy as np

data = Vec(...)
sum = Vec(...)

for d in data:
    sum = $\sigma$(W d + U sum + b)
```

where $\sigma$ is a general non-linearity (relu, sigmoid, pick your choice), W and U are matrices acting on d and sum respectively and b is a bias vector.

This loop is a recurrent neural network (RNN). Pictorially, this loop can be represented as shown below:

![RNN Anatomy](https://github.com/TreeinRandomForest/RHCourse/blob/main/p1/media/RNN%20Anatomy.png)

**General note**:
* If you are used to thinking of neural networks as multi-layer perceptrons, this might look funny. You should instead think of neural networks as computations that take vectors (or tensors) as inputs and apply a sequence of differentiable operations.



**Exercises**:
1. Generate a train and test set of sequences of numbers of fixed length (say 20). Compute the train and test mean sequared error losses.
2. Train on length L and evaluate on tests sets of various lengths (both < L as well as > L). What happens to test performance as a function of L?
3. On the training set where gradients are computed, plot the distribution of gradients across epochs and for each iteration (is this possible in PyTorch?)
4. Pick a realistic sequence prediction task and train an RNN. Papers are a good place to look for examples. Please, no stock prediction based on a single time-series - no one on Wall Street does this - see this [book](https://press.princeton.edu/books/paperback/9780691134796/asset-price-dynamics-volatility-and-prediction) for a better introduction to understanding asset prices.

**Backpropagation in Vanilla RNNs**

This is a simple example to demonstrate the calculation of gradients in a vanilla (the type described and implemented above) RNN. For the sake of clarity, all the details will be written out. We'll work with the case when $T=5$ i.e. we have 5 time-steps with inputs $x_1, x_2, x_3, x_4, x_5$ and there's a single output $\hat{y}$ for the whole sequence. This is shown in the picture below:

![RNN Anatomy](https://github.com/TreeinRandomForest/RHCourse/blob/main/p1/media/RNNWithOutput.png)

The sequence of computations is:

$$h_1 = f(W_{hh} h_0 + W_{hx} x_1 + b_h)$$

$$h_2 = f(W_{hh} h_1 + W_{hx} x_2 + b_h)$$

$$h_3 = f(W_{hh} h_2 + W_{hx} x_3 + b_h)$$

$$h_4 = f(W_{hh} h_3 + W_{hx} x_4 + b_h)$$

$$h_5 = f(W_{hh} h_4 + W_{hx} x_5 + b_h)$$

$$\hat{y} = g(W_{yh} h_5 + b_y)$$

The loss function, $L$ is used to compare the prediction $\hat{y}$ with the label $y$. The loss function used here is mean-squared error i.e. $L(\hat{y}, y) = \frac{1}{2}(\hat{y} - y)^2$. For gradient descent, we care about the following derivatives:

$\frac{\partial L}{\partial W_{yh}}, \frac{\partial L}{\partial W_{hh}}, \frac{\partial L}{\partial W_{hx}}, \frac{\partial L}{\partial b_{y}}, \frac{\partial L}{\partial b_{h}}$

For now, we will assume that all the weights and biases are 1-dimensional i.e. $W_{yh}, W_{hh}, W_{hx}, b_y, b_h \in \mathbb{R}$.

$\frac{\partial L}{\partial W_{yh}}$:

$$\begin{align}
\frac{\partial L}{\partial W_{yh}} &= \frac{\partial}{\partial W_{yh}} \frac{1}{2}(\hat{y} - y)^2 \\
&= (\hat{y} - y) \frac{\partial \hat{y}}{\partial W_{yh}} \\
&= (\hat{y} - y) \frac{\partial g(W_{yh} h_5 + b_y)}{\partial W_{yh}} \\
&= (\hat{y} - y) g'() \frac{\partial \left( W_{yh} h_5 + b_y \right)}{\partial W_{yh}} \\
&= (\hat{y} - y) g'() h_5 \\
\end{align}$$

Note: we will use the notation $g'()$ to denote the derivative of g with respect to its argument. For example, if $g$ is a sigmoid, $g'(x) = g(x) (1-g(x))$. For derivatives with respect to the weights or biases, we'll explicitly write the full partial derivative. To be very clear, $g'()$ above refers to $g'(W_{yh} h_5 + b_y)$ but since the argument can be inferred, we will use the shorthand $g'()$.

Note: To keep track of arguments for the hidden states, instead of just writing $h_5 = f()$, we'll write $h_5 = f_5()$. This doesn't mean that the activation for each time-step is different. It's just a book-keeping device to ensure we don't forget which hidden state we are referring to in the equations. This will become more clear in the example below:

$\frac{\partial L}{\partial W_{hh}}$:

$$\begin{align}
\frac{\partial L}{\partial W_{hh}} &= \frac{\partial}{\partial W_{hh}} \frac{1}{2}(\hat{y} - y)^2 \\


\end{align}$$

&= (\hat{y} - y) \frac{\partial \hat{y}}{\partial W_{hh}} \\
&= (\hat{y} - y) \frac{\partial g(W_{yh} h_5 + b_y)}{\partial W_{hh}} \\
&= (\hat{y} - y) g'() \frac{\partial \left( W_{yh} h_5 + b_y \right)}{\partial W_{hh}} \\
&= (\hat{y} - y) g'() \left[ W_{yh} \frac{\partial h_5}{\partial W_{hh}}\right] \\
&= (\hat{y} - y) g'() W_{yh} h'_{5}() \frac{\partial \left( W_{hh} h_4 + W_{hx} x_5 + b_h \right)}{\partial W_{hh}} \\
&= (\hat{y} - y) g'() W_{yh} h'_{5}() \left[ h_4 + W_{hh} \frac{\partial h_4}{\partial W_{hh}}\right] \\
