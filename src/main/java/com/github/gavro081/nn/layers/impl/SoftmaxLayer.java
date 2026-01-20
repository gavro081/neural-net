package com.github.gavro081.nn.layers.impl;

import java.io.Serial;
import java.io.Serializable;

import com.github.gavro081.nn.layers.ILayer;

public class SoftmaxLayer implements ILayer, Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    private int dimensions = -1;
    private transient double[] cachedOutput;

    public SoftmaxLayer() {}

    @Override
    public double[] forward(double[] input) {
        if (dimensions == -1) {
            dimensions = input.length;
        }

        double[] output = new double[input.length];

        // numerical stability: subtract max value
        double max = input[0];
        for (int i = 1; i < input.length; i++) {
            if (input[i] > max) max = input[i];
        }

        // compute exp(x - max) and sum
        double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.exp(input[i] - max);
            sum += output[i];
        }

        // normalize
        for (int i = 0; i < input.length; i++) {
            output[i] /= sum;
        }

        // cache the output for backward pass
        cachedOutput = output.clone();
        return output;
    }

    @Override
    public double[] backward(double[] dL_dy) {
        // Softmax Jacobian: J_ij = s_i * (δ_ij - s_j)
        // where s = softmax output, δ_ij = Kronecker delta
        double[] dL_dx = new double[dL_dy.length];

        for (int i = 0; i < dL_dx.length; i++) {
            dL_dx[i] = 0.0;
            for (int j = 0; j < dL_dy.length; j++) {
                double kronecker = (i == j) ? 1.0 : 0.0;
                dL_dx[i] += dL_dy[j] * cachedOutput[i] * (kronecker - cachedOutput[j]);
            }
        }

        return dL_dx;
    }

    @Override
    public void zeroGradients() {}

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
    public void updateWeights(double learningRate) {}
}
