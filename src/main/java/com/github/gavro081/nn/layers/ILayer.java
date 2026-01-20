package com.github.gavro081.nn.layers;

public interface ILayer {
    double[] forward(double[] input);

    double[] backward(double[] input);

    void zeroGradients();

    // returns -1 for dimension-preserving layers (e.g. activation functions)
    int getInputDimensions();

    // returns -1 for dimension-preserving layers until after the first forward pass or when the setter is called
    int getOutputDimensions();

    // called during dimension validation before training
    void setInputDimensions(int inputDimensions);

    void updateWeights(double learningRate);
}
