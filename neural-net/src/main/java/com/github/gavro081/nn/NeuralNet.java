package com.github.gavro081.nn;

import java.util.ArrayList;
import java.util.List;

import com.github.gavro081.nn.exceptions.ClassDimensionsMismatchException;
import com.github.gavro081.nn.exceptions.DimensionMismatchException;
import com.github.gavro081.nn.layers.ILayer;
import com.github.gavro081.nn.loss.ILoss;

public class NeuralNet {
    List<ILayer> layers;
    ILoss lossFunction = null;
    double []input;

    public NeuralNet() {
        layers = new ArrayList<>();
        input = null;
    }

    public NeuralNet addLayer(ILayer layer){
        layers.add(layer);
        return this;
    }

    public void setLossFunction(ILoss lossFunction){
        this.lossFunction = lossFunction;
    }

    private double[] forward(double []input){
        double []output = input;
        for (ILayer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    private void backward(double []input){
        double[] gradient = input;
        for (int i = layers.size() - 1; i >= 0; i--) {
            gradient = layers.get(i).backward(gradient);
        }
    }

    public void fit(double[][] input, int[] labels) throws Exception {
        int numClasses = inferNumClasses(labels);
        validateDimensions(input[0].length, numClasses);
        if (this.lossFunction == null) throw new RuntimeException("Missing loss function. Make sure to set it using nn.setLossFunction(ILoss loss).");

        for (ILayer layer : layers) {
            layer.zeroGradients();
        }

        double[][] outputs = new double[input.length][];
        for (int i = 0; i < input.length; i++) {
            outputs[i] = forward(input[i]);
        }

        double loss = lossFunction.calculateLoss(outputs, labels, numClasses);
        System.out.println("Loss: " + loss);

        double[][] lossGradient = lossFunction.calculateGradient(outputs, labels, numClasses);

        for (int i = 0; i < input.length; i++) {
            backward(lossGradient[i]);
        }

        // optimizer.step();
    }


    private int inferNumClasses(int[] labels) {
        int maxLabel = -1;
        for (int label : labels) {
            if (label > maxLabel) {
                maxLabel = label;
            }
        }
        return maxLabel + 1; // +1 because labels are 0-indexed
    }
    
    private void validateDimensions(int inputDimension, int numClasses) throws Exception {
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

        if (currentDimension != numClasses) {
            throw new ClassDimensionsMismatchException(numClasses, currentDimension);
        }
    }
}
