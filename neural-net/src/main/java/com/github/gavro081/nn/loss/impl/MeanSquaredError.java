package com.github.gavro081.nn.loss.impl;

import java.io.Serial;
import java.io.Serializable;

import com.github.gavro081.nn.loss.ILoss;


public class MeanSquaredError implements ILoss, Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    @Override
    public double calculateLoss(double[][] outputs, int[] targets, int numClasses) {
        int N = outputs.length;
        double totalLoss = 0.0;

        for (int i = 0; i < N; i++) {
            // for classification, convert target to one-hot and compute MSE
            double[] oneHot = new double[numClasses];
            oneHot[targets[i]] = 1.0;

            for (int j = 0; j < outputs[i].length; j++) {
                double diff = outputs[i][j] - oneHot[j];
                totalLoss += diff * diff;
            }
        }

        return totalLoss / N;
    }

    @Override
    public double[][] calculateGradient(double[][] outputs, int[] targets, int numClasses) {
        int batchSize = outputs.length;
        double[][] gradient = new double[batchSize][];

        for (int i = 0; i < batchSize; i++) {
            int outputSize = outputs[i].length;
            gradient[i] = new double[outputSize];

            // convert target to one-hot
            double[] oneHot = new double[numClasses];
            oneHot[targets[i]] = 1.0;

            // gradient: dL/d(y_pred) = 2 * (y_pred - y_true) / N
            for (int j = 0; j < outputSize; j++) {
                gradient[i][j] = 2.0 * (outputs[i][j] - oneHot[j]) / batchSize;
            }
        }

        return gradient;
    }
}
