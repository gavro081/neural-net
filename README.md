# Neural Network from Scratch in Java

This project is my attempt at understanding how neural networks actually work, through a from-scratch implementation of a feed-forward neural network in Java without using any external deep learning libraries\*.

\* deeplearning4j is used only for loading, reading and parsing the datasets.

## Table of Contents

- [How Neural Networks Work](#how-neural-networks-work)
  - [The Big Picture](#the-big-picture)
  - [Step 1: Forward Pass](#step-1-forward-pass)
  - [Step 2: Computing the Loss](#step-2-computing-the-loss)
  - [Step 3: Backward Pass (Backpropagation)](#step-3-backward-pass-backpropagation)
  - [Step 4: Updating Weights (Optimization)](#step-4-updating-weights-optimization)
- [Implementation Details](#implementation-details)
  - [Project Architecture](#project-architecture)
  - [Layers](#layers)
  - [Loss Functions](#loss-functions)
  - [Optimizers](#optimizers)
- [Getting Started](#getting-started)
- [Extending the Project](#extending-the-project)
- [Project Structure](#project-structure)
- [Playing around](#playing-around)
- [Key Insights](#key-insights)
- [An Important Note on Performance](#an-important-note-on-performance)
- [AI Usage](#ai-usage)

---

## How Neural Networks Work

### The Big Picture

A neural network is essentially a function that takes an input (like an image) and produces an output (like a class prediction). The "learning" part comes from adjusting internal parameters (called **weights** and **biases**) so that the network's output gets closer to the correct answer.

The training process repeats four main steps:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Input Data → [Forward Pass] → Prediction → [Loss] → Error Value    │
│                                                 ↓                   │
│  Update Weights ← [Optimizer] ← Gradients ← [Backward Pass]         │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Step 1: Forward Pass

The forward pass transforms input data into a prediction by passing it through a sequence of layers.

#### Linear Layer (Dense Layer)

The linear layer is the core building block. It computes a weighted sum of inputs and then adds a bias:

$$z_i = \sum_{j} w_{ij} \cdot x_j + b_i$$

Where:

- $x_j$ = input values
- $w_{ij}$ = weight connecting input $j$ to output $i$
- $b_i$ = bias term for output $i$
- $z_i$ = output (before activation)

**In code** (`LinearLayer.java`): The weights are stored as a 2D matrix where each row represents one output neuron's weights. The forward method computes the dot product between each weight row and the input.

#### Activation Functions

After a linear layer, we apply a non-linear activation function. Without this, stacking linear layers would just produce another linear function, which is useless for learning complex patterns.

**ReLU (Rectified Linear Unit)**

$$\text{ReLU}(x) = \max(0, x)$$

Very simple, but effective. It passes positive values unchanged and zeros out negatives. The implementation caches which values were positive (using a cached "mask") to use during backpropagation.

**Sigmoid**

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Squashes values into the range (0, 1). Useful for binary classification or when you need probability-like outputs.

**Softmax**

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

Converts a vector of raw scores (logits) into probabilities that sum to 1. Used for multi-class classification. The implementation subtracts the maximum value before exponentiating for numerical stability (prevents overflow).

---

### Step 2: Computing the Loss

The loss function measures how wrong our prediction is. We want to minimize this value. We pick the loss function based on the type of output/number of classes that we are trying to predict.

#### Cross-Entropy Loss

For multi-class classification:

$$L = -\frac{1}{N} \sum_{i=1}^{N} \log(p_{y_i})$$

Where $p_{y_i}$ is the predicted probability for the correct class.

The intuition here is simple: if the network predicts 90% confidence for the correct class, loss is low. If it predicts 10%, loss is high. The log function penalizes confident wrong predictions more severely.

**In code** (`MulticlassCrossEntropy.java`): Uses the log-sum-exp trick for numerical stability. Instead of computing log(softmax(x)), it computes: $\log(\text{softmax}(x_i)) = x_i - \max(x) - \log(\sum_j e^{x_j - \max(x)})$.

#### Mean Squared Error

For regression:

$$L = \frac{1}{N} \sum_{i} (y_{\text{pred}} - y_{\text{true}})^2$$

Simply measures the average squared difference between predictions and targets.

---

### Step 3: Backward Pass (Backpropagation)

This is the most complex step of the network. In this step, we need to figure out how much each weight contributed to the prediction.

Backpropagation uses the **chain rule** from calculus to compute gradients layer by layer, moving backwards, from output to input.

#### The Chain Rule

If $L$ depends on $z$ which depends on $w$:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w}$$

The intuition here is that in order to figure out how much does our loss $L$ change if the weight $w$ changes, we need to split that into two parts: how much $L$ changes when $z$ (intermediate variable e.g. pre-activation) changes, and how much $z$ changes when $w$ changes. Since $w$ influences $L$ only through $z$, these sensitivies are multiplied. If the network has multiple stacked layers, as we propagate backwards more terms are added to this formula, forming a "chain" of dependencies.

#### Gradients for Each Layer Type

**Linear Layer**: Given the gradient from the next layer ($\frac{\partial L}{\partial z}$), we compute:

- Weight gradient: $\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial z_i} \cdot x_j$ (gradient × input that was fed in)
- Bias gradient: $\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial z_i}$
- Input gradient (to pass backward): $\frac{\partial L}{\partial x_j} = \sum_i w_{ij} \cdot \frac{\partial L}{\partial z_i}$

**ReLU**:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \mathbf{1}_{x > 0}$$

If the input was positive pass the gradient through, otherwise block it.

**Sigmoid**:
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \sigma(x) \cdot (1 - \sigma(x))$$

Uses the cached output from the forward pass.

**Softmax**: The Jacobian matrix is computed as:
$$J_{ij} = s_i \cdot (\delta_{ij} - s_j)$$

Where $s$ is the softmax output and $\delta_{ij}$ is 1 when $i=j$, else 0.

---

### Step 4: Updating Weights (Optimization)

Once we have gradients, we update the weights to reduce the loss.

#### Stochastic Gradient Descent (SGD)

The simplest approach:

$$w = w - \alpha \cdot \nabla_w L$$

Where $\alpha$ is the learning rate. Just move in the opposite direction of the gradient.

#### Adam Optimizer

A more sophisticated method that adapts the learning rate for each parameter:

1. **Momentum ($m$)**: Exponential moving average of gradients (smooths updates)
2. **Velocity ($v$)**: Exponential moving average of squared gradients (adapts per-parameter)

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

With bias correction:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Update rule:
$$w = w - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Adam generally gives better results because it combines the benefits of momentum (faster convergence, escapes local minima) with adaptive learning rates (automatically scales updates for each weight).

On the tests i ran on the MNIST dataset, changing the optimizer from SGD to Adam gave ~3% increase (from ~94% to ~97%) on the same model.

---

## Implementation Details

For those that aren't very interested in the math, this is where the fun part begins.

### Project Architecture

![Project architecture](resources/nn.drawio.png)

### Layers

All layers implement the `ILayer` interface:

| Method                        | Purpose                                                   |
| ----------------------------- | --------------------------------------------------------- |
| `forward(double[] input)`     | Compute output from input, cache values for backward pass |
| `backward(double[] gradient)` | Compute gradients and pass them to previous layer         |
| `zeroGradients()`             | Reset accumulated gradients before each batch             |
| `updateWeights(double lr)`    | Apply gradient updates (used by SGD)                      |

**Available Layers:**

- `LinearLayer` – Fully connected layer with weights and biases
- `ReluLayer` – ReLU activation
- `SigmoidLayer` – Sigmoid activation
- `SoftmaxLayer` – Softmax activation (for classification output)

### Loss Functions

All losses implement the `ILoss` interface:

| Loss                     | Use Case                                            |
| ------------------------ | --------------------------------------------------- |
| `MulticlassCrossEntropy` | Multi-class classification (e.g., MNIST digits 0-9) |
| `BinaryCrossEntropy`     | Binary classification (yes/no, true/false)          |
| `MeanSquaredError`       | Regression or as an alternative for classification  |

### Optimizers

All optimizers implement the `IOptimizer` interface:

| Optimizer       | Description                          |
| --------------- | ------------------------------------ |
| `SGDOptimizer`  | Vanilla stochastic gradient descent  |
| `AdamOptimizer` | Adaptive learning rate with momentum |

---

## Getting Started

### Prerequisites

- Java 23+
- Maven

### Building the Project

```bash
mvn clean compile
```

### Running the MNIST Example

The `Main.java` demonstrates training on the MNIST handwritten digit dataset:

```bash
mvn exec:java -Dexec.mainClass="com.github.gavro081.nn.Main"
```

This will:

1. Load MNIST training data (60,000 handwritten digits). On the first load this might take a couple of minutes as the dataset is being downloaded.
2. Train a neural network with architecture: 784 → 128 → 64 → 32 → 10
3. Evaluate on the test set (10,000 images)
4. Optionally predict custom PNG images

### Configuration

In `Main.java`, you can adjust:

```java
final int BATCH_SIZE = 64;           // samples per gradient update
final int EPOCHS = 10;               // full passes through training data
final boolean LOAD_EXISTING_MODEL = true;   // load saved model
final boolean SAVE_MODEL = false;    // save model after training
```

### Basic Usage

```java
// 1. create a neural network
NeuralNet nn = new NeuralNet();

// 2. add layers
nn.addLayer(new LinearLayer(784, 128));
nn.addLayer(new ReluLayer());
nn.addLayer(new LinearLayer(128, 64));
nn.addLayer(new ReluLayer());
nn.addLayer(new LinearLayer(64, 10));

// 3. set loss function and optimizer
nn.setLossFunction(new MulticlassCrossEntropy());
nn.setOptimizer(new AdamOptimizer(0.001));

// 4. train
nn.fit(trainingData, labels);

// 5. evaluate
double accuracy = nn.evaluate(testData, testLabels);

// 6. save/load model
nn.save("model.nn.gz");
NeuralNet loaded = NeuralNet.load("model.nn.gz");
```

---

## Extending the Project

### Adding a New Activation Layer

1. Create a class in `layers/impl/` that implements `ILayer`
2. Implement `forward()` with your activation function
3. Implement `backward()` with its derivative
4. Leave `zeroGradients()` and `updateWeights()` empty (no learnable parameters)

### Adding a New Loss Function

1. Create a class in `loss/impl/` that implements `ILoss`
2. Implement `calculateLoss()` to compute the scalar loss value
3. Implement `calculateGradient()` to return the gradient w.r.t. outputs

### Adding a New Optimizer

1. Create a class in `optimizer/impl/` that implements `IOptimizer`
2. Implement `step(List<ILayer> layers)` to update weights

---

## Project Structure

```
src/main/java/com/github/gavro081/nn/
├── NeuralNet.java         # Main network class (forward, backward, fit, predict)
├── Main.java              # MNIST training example
├── layers/
│   ├── ILayer.java        # Layer interface
│   └── impl/
│       ├── LinearLayer.java
│       ├── ReluLayer.java
│       ├── SigmoidLayer.java
│       └── SoftmaxLayer.java
├── loss/
│   ├── ILoss.java         # Loss function interface
│   └── impl/
│       ├── MulticlassCrossEntropy.java
│       ├── BinaryCrossEntropy.java
│       └── MeanSquaredError.java
├── optimizer/
│   ├── IOptimizer.java    # Optimizer interface
│   └── impl/
│       ├── AdamOptimizer.java
│       └── SGDOptimizer.java
├── utils/
│   ├── ExtractedData.java      # Dataset extraction helper
│   └── ImagePredictor.java     # PNG image prediction utility
└── exceptions/
    ├── DimensionMismatchException.java
    └── ClassDimensionsMismatchException.java
```

---

## Playing Around

There is a saved model that you could load from `mnist_model.nn.gz` by running the main function with `LOAD_EXISTING_MODEL = true`. This will skip the training process entirely and just evaluate the model on the test set.

You may also add your own images of numbers and have the model predict them by first drawing them in whichever app you prefer and then moving the images to `src/main/resources/iloveimg-resized/`. You don't have to resize them to 28x28 as that is handled in the code. However, be aware that the model is trained on white digits on a black background, so if your images have black digits on a white background make sure that the parameter passed in the line `predictor.setInvertColors(true);` in the main function is set to `true`.

## Key Insights

1. **Batch processing**: The network processes multiple samples together before updating weights. For each sample in the batch, we run forward pass, compute loss, and run backward pass to accumulate gradients. Only after processing all samples do we update the weights using the accumulated (averaged) gradients. This makes training more stable and efficient.

2. **Gradient accumulation**: Gradients are accumulated across the batch, then averaged when computing the update. This is why we call `zeroGradients()` at the start of each batch to reset the accumulators.

3. **Caching for backprop**: Each layer caches values from the forward pass (inputs, intermediate results) because they're needed during backpropagation.

4. **Weight initialization**: The `LinearLayer` uses He initialization (scaling by $\sqrt{2/n_{in}}$) which works well with ReLU activations.

5. **Dimension validation**: The network validates layer dimensions before training to catch architecture errors early.

6. **Serialization**: The entire network (including optimizer state) can be saved/loaded using Java serialization with GZIP compression.

## An Important Note on Performance

This implementation was NOT built with performance in mind, it was built for educational purposes and getting a deeper understanding of the underlying structure of a neural network.

No real optimizations are used:
- CPU only
- nested loops for matrix operations,
- sequential sample by sample processing instead of true batching

All of which result in 10-50x worse performances compared to PyTorch on a CPU, and 100-500x compared to PyTorch on GPU (or at least those are the values that Claude calculated), the trade-off of course being, clarity, learning value and developer experience.

## AI Usage

Throughout the project AI was used mostly for help around the implementations of the math formulas (more specifically the gradients and loss functions), the `ImagePredictor.java` class which is fully AI-generated, and of course with writing this documentation :).

---

## License

This project is for educational purposes. Feel free to use, modify, and learn from it.
