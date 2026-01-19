package com.github.gavro081.nn.layers.impl;

import com.github.gavro081.nn.layers.BaseLayer;

import java.util.Arrays;
import java.util.Random;


public class LinearLayer extends BaseLayer {
    double[][] weights;
    double[] bias;
    
    // gradient accumulation (accumulated across batch)
    private double[][] weightGradients;
    private double[] biasGradients;
    
    // cache from forward pass (used in backward pass)
    private double[] cachedInput;
    
    public final Random random = new Random(12345); // todo: proper seed

    public LinearLayer(int inputDimensions, int outputDimensions) {
        super(inputDimensions, outputDimensions);
        weights = initWeights(inputDimensions, outputDimensions);
        bias = new double[outputDimensions];
        Arrays.fill(bias, 0.0);
        
        // Initialize gradient storage
        weightGradients = new double[outputDimensions][inputDimensions];
        biasGradients = new double[outputDimensions];
    }


    private double[][] initWeights(int inputDimensions, int outputDimensions){
        // row i in the matrix represents the vector of weights for the i-th neuron in the output
        // column j in the matrix represents the vector of weights for the j-th neuron in the input
        double[][] weights = new double[outputDimensions][inputDimensions];
        
        // xavier/He initialization: scale weights by sqrt(2/inputDimensions)
        double scale = Math.sqrt(2.0 / inputDimensions);
        
        for (int i = 0; i < outputDimensions; i++) {
            for (int j = 0; j < inputDimensions; j++) {
                // gaussian random with proper scaling
                weights[i][j] = (random.nextGaussian()) * scale;
            }
        }
        return weights;
    }

    @Override
    public double[] forward(double[] input) {
        this.cachedInput = input;
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
        for (int i = 0; i < output.length; i++) {
            this.output[i] = dotProduct(input, weights[i]) + bias[i];
        }
        return this.output;
    }

    @Override
    public double[] backward(double[] dL_dy) {
        for (int i = 0; i < outputDimensions; i++) {
            for (int j = 0; j < inputDimensions; j++) {
                weightGradients[i][j] += dL_dy[i] * cachedInput[j];
            }
        }

        for (int i = 0; i < outputDimensions; i++) {
            biasGradients[i] += dL_dy[i];
        }

        double []dL_dx = new double[inputDimensions];
        for (int j = 0; j < inputDimensions; j++) {
            for (int i = 0; i < outputDimensions; i++) {
                dL_dx[j] += weights[i][j] * dL_dy[i];
            }
        }

        return dL_dx;
    }
    
    @Override
    public void zeroGradients() {
        // reset weight and bias gradients
        for (double[] weightGradient : weightGradients) {
            Arrays.fill(weightGradient, 0.0);
        }
        Arrays.fill(biasGradients, 0.0);
    }


    @Override
    public void updateWeights(double learningRate) {
        for (int i = 0; i < outputDimensions; i++) {
            for (int j = 0; j < inputDimensions; j++) {
                weights[i][j] -= learningRate * weightGradients[i][j];
            }
            bias[i] -= learningRate * biasGradients[i];
        }
    }
}
