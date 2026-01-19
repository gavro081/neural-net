package com.github.gavro081.nn.layers;

abstract public class BaseLayer implements ILayer {
    protected double[] input;
    protected double[] output;

    public BaseLayer(int inputDimensions, int outputDimensions) {
        this.input = new double[inputDimensions];
        this.output = new double[outputDimensions];
    }

    @Override
    public int getInputDimensions() {
        return input.length;
    }

    @Override
    public int getOutputDimensions() {
        return output.length;
    }
}
