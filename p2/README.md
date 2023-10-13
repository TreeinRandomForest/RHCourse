## Notes on paper

[Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

These are very abbreviated notes:
- Deeper networks have better model performance than shallow networks.
- Deep networks were successfully trained using either new weight initialization schemes or different training mechanisms (i.e. not just fully end-to-end learning).
- Goal of paper: to understand why standard gradient descent doesn't work well as depth increases. the first target of study would be the gradients (see exercise 1 below).
- Claims:
  - Logistic sigmoid function ($\sigma(x)$) has a non-zero mean (=0.5). This has the effect of compounding (needs to be made more precise) the activations in the top (last?) hidden layer to saturation. Saturation refers to the fact that if the input activations are very large (say, >= 3) or very small (say, <= -3), then $\sigma(x)$ is close to 1 or 0 respectively and the derivative is close to 0. Proposed solution: different non-linearity that saturates less often.
  - Activations and gradients are studied across layers as a function of time (epochs or even batches). Want to study the singular values of the Jacobian (matrix of derivatives of a function from $\mathbb{R}^m \rightarrow \mathbb{R}^n$).
  - New weight initialization scheme that makes training converge much faster is proposed.

### Section 1: Deep Neural Networks
1. Why study deep networks? - better empirical performance on ML tasks (vision, NLP), theoretical appeal, inspiration from biology. [Future reading](https://www.iro.umontreal.ca/~lisa/pointeurs/TR1312.pdf)
2. Many methods for training deep networks but they have special weight initialization schemes or various training schemes. Q: why do these new methods work over vanilla GD (gradient descent)?

### Section 2: Experimental Setting and Datasets
ShapeNet- 3x2 (synthetic): 32x32 - 9 classes

MNIST: 28x28 - 10 clases

CIFAR-10: 32x32x3 - 10 classes

Small-ImageNet: 37x37 - 10 classes

NN setup:
* Feedforward MLPs
* number of hidden layers: 1 through 5
* number of nodes in hidden layers: 1000
* output activation: softmax (n-class classification)
* loss function: negative log-likelihood (cross-entropy)
* optimization with stochastic gradient descent (SGD) i.e. $\theta \leftarrow \theta - \epsilon g$
* batch-size: 10
* learning rate $\epsilon$ optimized based on validation set error after 5 million updates i.e. do a sweep of $\epsilon$ and pick one minimizing validation set error after 5 million updates (one $\epsilon$ per dataset and per #hidden layers?).
* non-linear activation: sigmoid ($\frac{1}{1 + e^{-x}}$), tanh ($\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$) and softsign ($\frac{x}{1 + \lvert x \rvert}$). We will also use relu and leakyrelu.

For each model (#hidden layers, non-linear activation, dataset), the best hyperparameters are found using the validation set error.

* Initialization scheme:
  * Biases: 0
  * Weights: At layer l, $W_{ij} \sim \mathcal{U}[-\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}}]$ i.e. uniform draws in the specified interval where $n$ is the number of incoming nodes from previous layer i.e. the number of columns of W. Note this means that the variance of each $W_{ij} \sim \frac{1}{n}$ (exercise: compute the mean and variance of $\mathcal{U}(-a, a)$.

### Section 3: Effect of Activation Functions and Saturation During Training

What we want to avoid:
* Saturation of units e.g. for sigmoid units, we don't want activations that have large absolute values since $\sigma(x)$ is very flat for large positive or negative $x$ and the derivative $\sigma'(x) \approx 0$. As gradients get propagated backwards, lots of saturated units will suppress this propagation (see exercise 1 below).
* We don't want activations only in the linear regime. E.g., a sigmoid is linear near $x = 0$. Linear units compose to form an effective linear unit thus collapsing the layers. Exercise: given two linear functions $f(x) = ax + b$ and $g(x) = cx + d$, what is $(g \circ f)(x) = g(f(x))$?

#### Section 3.1 Experiments with the Sigmoid

(Exercise) Prove the following properties of the sigmoid:
* $\lim_{x\rightarrow -\infty} \sigma(x) \rightarrow 0$
* $\lim_{x\rightarrow \infty} \sigma(x) \rightarrow 1$
* $\sigma(0) = \frac{1}{2}$
* Near $x=0$, $\sigma(x)$ is linear with slope $\frac{1}{4}$ (Taylor expansion). How large is the quadratic term? What about the cubic term?

Figure 2:
* Fix 300 test examples
* After every 20k updates to the model, collect the activations (after applying the non-linearity) for the 300 examples at each layer. Plot the mean and standard deviation (across 300 examples, across nodes in given hidden layer) at each layer as a function of the 20k epochs.
* Layer 1 refers to activations (after applying the non-linearity) at the first hidden layer after input layer.
* At the end of training (x-axis near 140):
  * Layers 1-3 have activations centered (approximately) around 0.5.
  * Layer 4 shows saturations of units with activations close to 0 (negative inputs).
* Training dynamics
  * Layers 1 and 2 generally have activations around 0.5 for most of training (but something special happens near $x = 100$)
  * Layer 3 starts biased/saturated towards 1 but rapidly moves towards being centered around 0.5 after $x = 100$. This might be because of the statistics of the input data (the pixels are not centered etc.)
  * Layer 4 rapidly moves towards saturation and recovers mildly near the end.
  * The variance of activations goes down as one goes deeper into the network. Recall that very low variance near 0.5 is the linear regime for the sigmoid.

Summary:
* First few layers start unsaturated and get closer to saturation with time i.e. they stay centered but the standard deviation increases.
* Last layers start saturated and can recover (sometimes barely)

Figure 3:


## Experiments to try out:

1. Read the first two sections of [this](https://treeinrandomforest.github.io/deep-learning/2018/10/30/backpropagation.html) article (Backpropagation I and II). By looking at the expressions for the gradients, can you identify what potentially goes wrong as network depth increases? How would we test our hypothesis?

2. Plot the various activation functions and their derivatives (sigmoid, tanh, softsign, relu, leakyrelu)

2. Implement a feedforward neural network with dense layers and a non-linear activation to learn the identity function: $$id: \mathbb{R}^n \rightarrow \mathbb{R}^n$$ with $id(x) = x$. The free parameters here are n (dimensionality of the input and output space), the number of nodes in the hidden layers (we'll keep this constant across layers), the number of layers and the non-linear activation function. The loss will be mean-squared error loss. Given a fixed number of epochs E, record the train and test errors as a function of n, number of hidden nodes, number of hidden layers, two choices of activation functions (sigmoid and relu).

## Further Readings
* LeCun, Y., Bottou, L., Orr, G. B., & M ̈uller, K.-R. (1998b). Efficient backprop. In Neural networks, tricks of the trade, Lecture Notes in Computer Science LNCS 1524. Springer Verlag