package com.github.gavro081.nn.loss;

public interface ILoss {
    double calculateLoss(double[][] outputs, int[] targets, int numClasses);
    double[][] calculateGradient(double[][] outputs, int[] targets, int numClasses);
}
