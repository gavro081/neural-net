package com.github.gavro081.nn;

import com.google.flatbuffers.FlatBufferBuilder;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) throws Exception {
        DataSetIterator mnistTrain = new MnistDataSetIterator(100, true, 12345);
        int i = 1;
        DataSet next = mnistTrain.next();
//        while (mnistTrain.hasNext()) {
//            DataSet next = mnistTrain.next();
//            System.out.println("processing batch " + i++);
//        }
        List<DataSet> list = next.asList();
        DataSet ds = list.getFirst();
        INDArray features = ds.getFeatures();
//
        double[] vector = features.reshape(features.length()).toDoubleVector();
//        System.out.println(vector);
        System.out.println(vector.length);
        Arrays.stream(vector).forEach(s -> System.out.print(s + ",\t"));
    }
}