package com.github.gavro081.nn;

import com.github.gavro081.nn.layers.impl.LinearLayer;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;

public class Main {
    public static void main(String[] args) throws Exception {
        DataSetIterator mnistTrain = new MnistDataSetIterator(100, true, 12345);

        NeuralNet nn = new NeuralNet();
        nn.addLayer(new LinearLayer(28 * 28, 10));
//        nn.addLayer(new LinearLayer(3, 1));
        // double[][] trainX
        // int[] trainY

//        nn.fit(next);

//        double[][]trainX_ = new double[][]{{1,5,-6}};
//        int[]trainY_ = new int[]{0,0,0};
//        nn.fit(trainX_, trainY_);

        while (mnistTrain.hasNext()){
            List<DataSet> batch = mnistTrain.next().asList();
            double [][]trainX = new double[batch.size()][(int) batch.getFirst().getFeatures().length()];
            int []trainY = new int[batch.size()];
            for (int i = 0; i < batch.size(); i++) {
                DataSet ds = batch.get(i);
                INDArray features = ds.getFeatures();
                trainX[i] = features.reshape(features.length()).toDoubleVector(); // input neurons
                trainY[i++] = ds.outcome();
            }
            nn.fit(trainX, trainY);
            break;
        }
    }
}