package com.github.gavro081.nn.optimizer.impl;

import com.github.gavro081.nn.layers.ILayer;
import com.github.gavro081.nn.optimizer.IOptimizer;

import java.io.Serial;
import java.io.Serializable;
import java.util.List;

public class SGDOptimizer implements IOptimizer, Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

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
