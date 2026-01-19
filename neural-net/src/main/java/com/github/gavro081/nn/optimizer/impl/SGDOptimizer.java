package com.github.gavro081.nn.optimizer.impl;

import com.github.gavro081.nn.layers.ILayer;
import com.github.gavro081.nn.optimizer.IOptimizer;

import java.util.List;

public class SGDOptimizer implements IOptimizer {
    private final double learningRate;

    public SGDOptimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public void step(List<ILayer> layers) {
        for (ILayer layer : layers){
            layer.updateWeights(learningRate);
        }
    }
}
