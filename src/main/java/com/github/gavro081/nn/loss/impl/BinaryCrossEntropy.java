package com.github.gavro081.nn.loss.impl;

import java.io.Serial;
import java.io.Serializable;

import com.github.gavro081.nn.loss.ILoss;

/**
 * Binary Cross-Entropy Loss for binary classification.
 * Supports both single output (after sigmoid) and two outputs.
 * Target should be 0 or 1.
 */
public class BinaryCrossEntropy implements ILoss, Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    private static final double EPSILON = 1e-15; // for numerical stability

    @Override
    public double calculateLoss(double[][] outputs, int[] targets, int numClasses) {
        int N = outputs.length;
        double totalLoss = 0.0;

        for (int i = 0; i < N; i++) {
            int y = targets[i];
            
            if (outputs[i].length == 1) {
                // single output case
                double p = clip(outputs[i][0]);
                totalLoss -= y * Math.log(p) + (1 - y) * Math.log(1 - p);
            } else {
                // two output case - use probability at target index
                double p = clip(outputs[i][y]);
                totalLoss -= Math.log(p);
            }
        }

        return totalLoss / N;
    }

    @Override
    public double[][] calculateGradient(double[][] outputs, int[] targets, int numClasses) {
        int batchSize = outputs.length;
        double[][] gradient = new double[batchSize][];

        for (int i = 0; i < batchSize; i++) {
            int y = targets[i];
            
            if (outputs[i].length == 1) {
                // single output case
                gradient[i] = new double[1];
                double p = clip(outputs[i][0]);
                gradient[i][0] = (p - y) / batchSize;
            } else {
                // two output case - gradient for all outputs
                gradient[i] = new double[outputs[i].length];
                for (int j = 0; j < outputs[i].length; j++) {
                    double p = clip(outputs[i][j]);
                    if (j == y) {
                        gradient[i][j] = (p - 1.0) / batchSize;  // target is 1
                    } else {
                        gradient[i][j] = p / batchSize;  // target is 0
                    }
                }
            }
        }

        return gradient;
    }

    private double clip(double value) {
        return Math.max(EPSILON, Math.min(1 - EPSILON, value));
    }
}
