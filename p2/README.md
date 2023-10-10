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
* non-linear activation: sigmoid ($\frac{1}{1 + \exp^{-x}}$), tanh ($\frac{\exp^{x} - \exp{-x}}{\exp^{x} + \exp^{-x}}$) and softsign ($\frac{x}{1 + \lvert x \rvert}$). We will also use relu and leakyrelu.

For each model (#hidden layers, non-linear activation, dataset), the best hyperparameters are found using the validation set error.

* Initialization scheme:
  * Biases: 0
  * Weights: At layer l, $W_{ij}$ \sim \mathcal{U}[-\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}}] i.e. uniform draws in the specified interval where $n$ is the number of incoming nodes from previous layer i.e. the number of columns of W. Note this means that the variance of each $W_{ij} \sim \frac{1}{n}$ (exercise: compute the mean and variance of $\mathcal(-a, a)$.

### Section 3: Effect of Activation Functions and Saturation During Training



## Experiments to try out:

1. Read the first two sections of [this](https://treeinrandomforest.github.io/deep-learning/2018/10/30/backpropagation.html) article (Backpropagation I and II). By looking at the expressions for the gradients, can you identify what potentially goes wrong as network depth increases? How would we test our hypothesis?

2. Plot the various activation functions and their derivatives (sigmoid, tanh, softsign, relu, leakyrelu)

2. Implement a feedforward neural network with dense layers and a non-linear activation to learn the identity function: $$id: \mathbb{R}^n \rightarrow \mathbb{R}^n$$ with $id(x) = x$. The free parameters here are n (dimensionality of the input and output space), the number of nodes in the hidden layers (we'll keep this constant across layers), the number of layers and the non-linear activation function. The loss will be mean-squared error loss. Given a fixed number of epochs E, record the train and test errors as a function of n, number of hidden nodes, number of hidden layers, two choices of activation functions (sigmoid and relu).

