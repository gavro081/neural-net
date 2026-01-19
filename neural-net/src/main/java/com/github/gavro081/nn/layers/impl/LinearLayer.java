package com.github.gavro081.nn.layers.impl;

import com.github.gavro081.nn.layers.BaseLayer;

import java.util.Random;


public class LinearLayer extends BaseLayer {
    double [][]weights;
    double bias;
    public final Random random = new Random(12345); // todo: proper seed

    public LinearLayer(int inputDimensions, int outputDimensions) {
        super(inputDimensions, outputDimensions);
        weights = initWeights(inputDimensions, outputDimensions);
        bias = 0;
    }


    private double[][] initWeights(int inputDimensions, int outputDimensions){
        // row i in the matrix represents the vector of weights for the i-th neuron in the output
        // column j in the matrix represents the vector of weights for the j-th neuron in the input
        double [][]weights = new double[outputDimensions][inputDimensions];
        for (int i = 0; i < outputDimensions; i++) {
            for (int j = 0; j < inputDimensions; j++) {
                // random.nextDouble() returns values in range [0,1]; i want that range to be [-1,1]
                weights[i][j] = (random.nextDouble() - 0.5) * 2;
            }
        }
        return weights;
    }

    @Override
    public double[] forward(double[] input) {
        /*
        * input:     [ 1, -1, 0.5] - x0, x1, x2
        * weights: [ [ 1, 2,  -2], - w00, w01, w02
        *            [-1, 1,   0], - w10, w11, s12
        *            [ 2, 3,   1] ]- w20, w21, w22
        * output : [
        *           w[00] * x[0] + w[01] * x[1] + x[02] * x[2] + bias, -> z[1]
        *           w[10] * x[0] + w[11] * x[1] + x[12] * x[2] + bias, -> z[2]
        *           w[20] * x[0] + w[21] * x[1] + x[22] * x[2] + bias, -> z[3]
        *           ]
        * */
        double []output = new double[getOutputDimensions()];
        // todo: test other way around
        for (int i = 0; i < output.length; i++) {
            output[i] = dotProduct(input, weights[i]) + bias;
        }
        return output;
    }

    // todo: move this method
    private double dotProduct(double[] input, double[] weight){
        assert input.length == weight.length; // todo: test & remove
        double sum = 0.0d;
        for (int i = 0; i < input.length; i++) {
            sum += input[i] * weight[i];
        }
        return sum;
    }
}
