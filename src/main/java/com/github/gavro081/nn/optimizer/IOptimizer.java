package com.github.gavro081.nn.optimizer;

import com.github.gavro081.nn.layers.ILayer;

import java.util.List;

public interface IOptimizer {
    void step(List<ILayer> layers);
}
