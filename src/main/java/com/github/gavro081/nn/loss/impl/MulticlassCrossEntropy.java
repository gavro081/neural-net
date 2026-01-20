package com.github.gavro081.nn.loss.impl;

import com.github.gavro081.nn.loss.ILoss;

import java.io.Serial;
import java.io.Serializable;

public class MulticlassCrossEntropy implements ILoss, Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    @Override
    public double calculateLoss(double[][] outputs, int[] targets, int numClasses) {
        int N = outputs.length;
        double totalLoss = 0.0;
        
        for (int i = 0; i < N; i++) {
            double[] logits = outputs[i];
            int trueClass = targets[i];
            
            // compute log(softmax(logits[trueClass])) in a numerically stable way
            double logSoftmaxTrue = logSoftmax(logits, trueClass);
            
            // cross-entropy: -log(p_true)
            totalLoss -= logSoftmaxTrue;
        }
        
        return totalLoss / N; // average loss across batch
    }

    private double logSoftmax(double[] logits, int index) {
        // find max for numerical stability
        double max = logits[0];
        for (double logit : logits) {
            if (logit > max) {
                max = logit;
            }
        }
        
        // compute log-sum-exp: log(Σ exp(x_j - max))
        double sumExp = 0.0;
        for (double logit : logits) {
            sumExp += Math.exp(logit - max);
        }
        double logSumExp = Math.log(sumExp);
        
        // log(softmax(x[index])) = x[index] - max - log(Σ exp(x_j - max))
        return logits[index] - max - logSumExp;
    }

    private double[] softmax(double []logits){
        double max = logits[0];
        for (double logit : logits) {
            if (logit > max) max = logit;
        }
        double[] result = new double[logits.length];
        double sum = 0.0;
        for (int i = 0; i < logits.length; i++) {
            result[i] = Math.exp(logits[i] - max);
            sum += result[i];
        }

        for (int i = 0; i < result.length; i++) {
            result[i] /= sum;
        }
        return result;
    }

    public double[][] calculateGradient(double[][] outputs, int[] targets, int numClasses){
        // dL/d(logits) = softmax(logits) - one_hot(true_label)
        // averaged over batch size
        int batchSize = outputs.length;
        double[][] gradient = new double[batchSize][];
        
        for (int i = 0; i < batchSize; i++) {
            gradient[i] = softmax(outputs[i]);
            
            int trueLabelIndex = targets[i];
            gradient[i][trueLabelIndex] -= 1.0;
            
            // average gradient by batch size
            for (int j = 0; j < gradient[i].length; j++) {
                gradient[i][j] /= batchSize;
            }
        }
        return gradient;
    }
}
