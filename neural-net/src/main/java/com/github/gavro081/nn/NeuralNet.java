package com.github.gavro081.nn;

import java.util.ArrayList;
import java.util.List;

import com.github.gavro081.nn.exceptions.DimensionMismatchException;
import com.github.gavro081.nn.layers.ILayer;

public class NeuralNet {
    List<ILayer> layers;
    double []input;

    public NeuralNet() {
        layers = new ArrayList<>();
        input = null;
    }

    public NeuralNet addLayer(ILayer layer){
        layers.add(layer);
        return this;
    }


    private double[] forward(double []input){
        double []output = input;
        for (ILayer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }
    public void fit(double[][] input, int[] labels) throws Exception{
        validateDimensions(input[0].length);
        for (int i = 0; i < input.length; i++) {
            double[] inputVector = input[i];
            double[] output = forward(inputVector);

            // loss(output, labels[i])
            // compute loss

            // backprop

            // optimizer step (?)
        }
    }

    private void validateDimensions(int inputDimension) throws Exception {
        int currentDimension = inputDimension;
        
        for (int i = 0; i < layers.size(); i++) {
            ILayer layer = layers.get(i);

            // for layers with no predetermined dimensions (activations), set the dimension
            if (layer.getInputDimensions() == -1) {
                layer.setInputDimensions(currentDimension);
            }

            if (currentDimension != layer.getInputDimensions()) {
                throw new DimensionMismatchException(currentDimension, layer.getInputDimensions(), i);
            }
            currentDimension = layer.getOutputDimensions();
        }

        // if (currentDimension != numClasses) {
        //     throw new ClassDimensionsMismatchException(numClasses, currentDimension);
        // }
    }
}
