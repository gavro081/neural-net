package com.github.gavro081.nn;

import com.github.gavro081.nn.exceptions.DimensionMismatchException;
import com.github.gavro081.nn.layers.ILayer;

import java.util.ArrayList;
import java.util.List;

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

//    public void fit(List<DataSet> dataSet) throws Exception{
////        validateDimensions(dataSet.getFirst().getFeatures().length(), dataSet.getFirst().numOutcomes());
//        // double[][] trainX
//        // int[] trainY
//        for (DataSet ds : dataSet){
//            INDArray features = ds.getFeatures();
//            double[] input = features.reshape(features.length()).toDoubleVector(); // input neurons
//
//            double[] output = forward(input);
//            for (double v : output) {
//                System.out.println(v);
//            }
//
//
//        }
//    }

    private void validateDimensions(long inputDimension)
            throws Exception {
        for (int i = 0; i < layers.size(); i++) {
            ILayer layer = layers.get(i);
            if (inputDimension != layer.getInputDimensions()) {
                throw new DimensionMismatchException(inputDimension, layer.getInputDimensions(), i);
            }
            inputDimension = layer.getOutputDimensions();
        }
//        if (inputDimension != numClasses) {
//            throw new ClassDimensionsMismatchException(numClasses, inputDimension);
//        }
    }
}
