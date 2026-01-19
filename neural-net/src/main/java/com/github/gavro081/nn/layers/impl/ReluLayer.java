package com.github.gavro081.nn.layers.impl;

import com.github.gavro081.nn.layers.ILayer;

public class ReluLayer implements ILayer {
    private int dimensions = -1; // -1 -> not yet determined
    
    public ReluLayer(){}

    @Override
    public double[] forward(double[] input) {
        if (dimensions == -1) {
            dimensions = input.length;
        }
        double []output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.max(0, input[i]);
        }
        return output;
    }
    
    @Override
    public int getInputDimensions() {
        return dimensions;
    }
    
    @Override
    public int getOutputDimensions() {
        return dimensions;
    }
    
    @Override
    public void setInputDimensions(int inputDimensions) {
        this.dimensions = inputDimensions;
    }
}
