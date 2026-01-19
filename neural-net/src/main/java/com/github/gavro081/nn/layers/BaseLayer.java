package com.github.gavro081.nn.layers;

import java.io.Serial;
import java.io.Serializable;

abstract public class BaseLayer implements ILayer, Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    protected double[] input;
    protected double[] output;
    protected int inputDimensions;
    protected int outputDimensions;

    public BaseLayer(int inputDimensions, int outputDimensions) {
        this.inputDimensions = inputDimensions;
        this.outputDimensions = outputDimensions;
        this.input = new double[inputDimensions];
        this.output = new double[outputDimensions];
    }

    @Override
    public int getInputDimensions() {
        return inputDimensions;
    }

    @Override
    public int getOutputDimensions() {
        return outputDimensions;
    }
    
    @Override
    public void setInputDimensions(int inputDimensions) {
        // fixed-dimension layers ignore this
    }

    protected double dotProduct(double[] input, double[] weight){
        double sum = 0.0d;
        for (int i = 0; i < input.length; i++) {
            sum += input[i] * weight[i];
        }
        return sum;
    }
}
