package com.github.gavro081.nn.optimizer.impl;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.github.gavro081.nn.layers.ILayer;
import com.github.gavro081.nn.layers.impl.LinearLayer;
import com.github.gavro081.nn.optimizer.IOptimizer;

public class AdamOptimizer implements IOptimizer, Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    private final double learningRate;
    private final double beta1;
    private final double beta2;
    private final double epsilon;
    
    private int timeStep = 0;
    
    // State storage for each layer
    private final Map<LinearLayer, AdamState> layerStates = new HashMap<>();
    
    public AdamOptimizer(double learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }
    
    public AdamOptimizer(double learningRate, double beta1, double beta2, double epsilon) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }
    
    @Override
    public void step(List<ILayer> layers) {
        timeStep++;
        
        for (ILayer layer : layers) {
            if (layer instanceof LinearLayer) {
                LinearLayer linearLayer = (LinearLayer) layer;

                if (!layerStates.containsKey(linearLayer)) {
                    layerStates.put(linearLayer, new AdamState(
                        linearLayer.getOutputDimensions(), 
                        linearLayer.getInputDimensions()
                    ));
                }
                
                AdamState state = layerStates.get(linearLayer);

                updateWeightsAdam(
                    linearLayer.weights,
                    linearLayer.weightGradients,
                    state.weightM,
                    state.weightV
                );

                updateBiasAdam(
                    linearLayer.bias,
                    linearLayer.biasGradients,
                    state.biasM,
                    state.biasV
                );
            }
        }
    }
    
    private void updateWeightsAdam(double[][] weights, double[][] gradients, 
                                   double[][] m, double[][] v) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                double grad = gradients[i][j];
                
                // update biased first moment estimate
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * grad;
                
                // update biased second raw moment estimate
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * grad * grad;
                
                // compute bias-corrected first moment estimate
                double mHat = m[i][j] / (1 - Math.pow(beta1, timeStep));
                
                // compute bias-corrected second raw moment estimate
                double vHat = v[i][j] / (1 - Math.pow(beta2, timeStep));
                
                // update weights
                weights[i][j] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
            }
        }
    }
    
    private void updateBiasAdam(double[] bias, double[] gradients, 
                                double[] m, double[] v) {
        for (int i = 0; i < bias.length; i++) {
            double grad = gradients[i];
            
            // update biased first moment estimate
            m[i] = beta1 * m[i] + (1 - beta1) * grad;
            
            // update biased second raw moment estimate
            v[i] = beta2 * v[i] + (1 - beta2) * grad * grad;
            
            // compute bias-corrected first moment estimate
            double mHat = m[i] / (1 - Math.pow(beta1, timeStep));
            
            // compute bias-corrected second raw moment estimate
            double vHat = v[i] / (1 - Math.pow(beta2, timeStep));
            
            // update bias
            bias[i] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
        }
    }
    
    private static class AdamState implements Serializable {
        @Serial
        private static final long serialVersionUID = 1L;

        double[][] weightM;  // first moment (mean) for weights
        double[][] weightV;  // second moment (variance) for weights
        double[] biasM;      // first moment for biases
        double[] biasV;      // second moment for biases
        
        AdamState(int outputDim, int inputDim) {
            weightM = new double[outputDim][inputDim];
            weightV = new double[outputDim][inputDim];
            biasM = new double[outputDim];
            biasV = new double[outputDim];
        }
    }
}
