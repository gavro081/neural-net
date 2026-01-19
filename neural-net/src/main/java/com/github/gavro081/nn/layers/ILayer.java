package com.github.gavro081.nn.layers;

public interface ILayer {
    double[] forward(double[] input);
    int getInputDimensions();
    int getOutputDimensions();
}
