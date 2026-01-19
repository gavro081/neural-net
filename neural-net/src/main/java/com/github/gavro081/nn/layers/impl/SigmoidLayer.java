package com.github.gavro081.nn.layers.impl;

import java.io.Serial;
import java.io.Serializable;

import com.github.gavro081.nn.layers.ILayer;

public class SigmoidLayer implements ILayer, Serializable {
    @Serial
    private static final long serialVersionUID = 1L;
    private int dimensions = -1;
    private transient double[] cachedOutput;

    public SigmoidLayer() {}

    @Override
    public double[] forward(double[] input) {
        if (dimensions == -1) {
            dimensions = input.length;
        }

        double[] output = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = 1.0 / (1.0 + Math.exp(-input[i]));
        }

        // cache the output for backward pass
        cachedOutput = output.clone();
        return output;
    }

    @Override
    public double[] backward(double[] dL_dy) {
        double[] dL_dx = new double[dL_dy.length];
        for (int i = 0; i < dL_dy.length; i++) {
            double sigmoidOutput = cachedOutput[i];
            dL_dx[i] = dL_dy[i] * sigmoidOutput * (1.0 - sigmoidOutput);
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
