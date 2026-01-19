package com.github.gavro081.nn.layers.impl;

import com.github.gavro081.nn.layers.ILayer;

public class ReluLayer implements ILayer {
    private int dimensions = -1; // -1 -> not yet determined
    private boolean[] cachedMask; // cache which elements were > 0
    
    public ReluLayer(){}

    @Override
    public double[] forward(double[] input) {
        if (dimensions == -1) {
            dimensions = input.length;
        }
        
        // cache mask for backward pass
        cachedMask = new boolean[input.length];
        double[] output = new double[input.length];
        
        for (int i = 0; i < input.length; i++) {
            cachedMask[i] = (input[i] > 0);
            output[i] = cachedMask[i] ? input[i] : 0;
        }
        
        return output;
    }

    @Override
    public double[] backward(double[] dL_dy) {
        // ReLU gradient: dL/dx = dL/dy * (1 if input > 0 else 0)
        double[] dL_dx = new double[dL_dy.length];
        for (int i = 0; i < dL_dy.length; i++) {
            dL_dx[i] = cachedMask[i] ? dL_dy[i] : 0;
        }
        return dL_dx;
    }
    
    @Override
    public void zeroGradients() {
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

    @Override
    public void updateWeights(double learningRate) {
    }
}
