package com.github.gavro081.nn;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import com.github.gavro081.nn.exceptions.ClassDimensionsMismatchException;
import com.github.gavro081.nn.exceptions.DimensionMismatchException;
import com.github.gavro081.nn.layers.ILayer;
import com.github.gavro081.nn.loss.ILoss;
import com.github.gavro081.nn.optimizer.IOptimizer;

public class NeuralNet implements Serializable{
    @Serial
    private static final long serialVersionUID = 1L;

    private List<ILayer> layers;
    private ILoss lossFunction;
    private IOptimizer optimizer;
    private double []input;
    private boolean validated = false;
    private double lastLoss = 0.0;  // store loss for monitoring

    public NeuralNet() {
        layers = new ArrayList<>();
        input = null;
    }

    public NeuralNet(IOptimizer optimizer, ILoss lossFunction){
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
    }
    public NeuralNet(ILoss lossFunction, IOptimizer optimizer){
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
    }

    public void addLayer(ILayer layer){
        layers.add(layer);
    }

    public void setOptimizer(IOptimizer optimizer){
        this.optimizer = optimizer;
    }

    public void setLossFunction(ILoss lossFunction){
        this.lossFunction = lossFunction;
    }

    public double[] forward(double []input){
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
        if (!validated) validateDimensions(input[0].length, numClasses);
        if (this.lossFunction == null) throw new RuntimeException("Missing loss function. Make sure to set it using nn.setLossFunction(ILoss loss).");

        for (ILayer layer : layers) {
            layer.zeroGradients();
        }

        // compute all forward passes to get outputs for gradient calculation
        double[][] outputs = new double[input.length][];
        for (int i = 0; i < input.length; i++) {
            // clone the output since layers reuse internal buffers
            outputs[i] = forward(input[i]).clone();
        }

        this.lastLoss = lossFunction.calculateLoss(outputs, labels, numClasses);  // store loss for monitoring

        double[][] lossGradient = lossFunction.calculateGradient(outputs, labels, numClasses);

        // for each sample, we need to re-run forward to restore cached values
        // before running backward. otherwise backward() uses cached values from the last
        // forward pass, not the current sample.
        for (int i = 0; i < input.length; i++) {
            forward(input[i]);  // re-cache the input for this sample
            backward(lossGradient[i]);
        }

        optimizer.step(layers);
    }

    public int[] predict(double[][] input) {
        int[] predictions = new int[input.length];
        
        for (int i = 0; i < input.length; i++) {
            double[] output = forward(input[i]);

            int predictedClass = 0;
            double maxValue = output[0];
            for (int j = 1; j < output.length; j++) {
                if (output[j] > maxValue) {
                    maxValue = output[j];
                    predictedClass = j;
                }
            }
            predictions[i] = predictedClass;
        }
        
        return predictions;
    }

    public double evaluate(double[][] testX, int[] testY) {
        int[] predictions = predict(testX);
        
        int correct = 0;
        for (int i = 0; i < testY.length; i++) {
            if (predictions[i] == testY[i]) {
                correct++;
            }
        }
        
        return (100.0 * correct) / testY.length;
    }
    
    public double getLastLoss() {
        return lastLoss;
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
        validated = true;
    }

    public void save(String filepath) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(filepath);
             GZIPOutputStream gzos = new GZIPOutputStream(fos);
             ObjectOutputStream oos = new ObjectOutputStream(gzos)) {
            
            oos.writeObject(this);
            System.out.printf("Model saved to: %s%n", filepath);
        }
    }

    public static NeuralNet load(String filepath) throws IOException, ClassNotFoundException {
        try (FileInputStream fis = new FileInputStream(filepath);
             GZIPInputStream gzis = new GZIPInputStream(fis);
             ObjectInputStream ois = new ObjectInputStream(gzis)) {
            
            NeuralNet nn = (NeuralNet) ois.readObject();
            System.out.printf("Loaded model from: %s%n", filepath);
            return nn;
        }
    }
}
