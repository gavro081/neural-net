package com.github.gavro081.nn.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.ArrayList;
import java.util.List;

public class ExtractedData {
    private final double[][] features;
    private final int[] labels;
    ExtractedData(double[][] features, int[] labels) {
        this.features = features;
        this.labels = labels;
    }

    public static ExtractedData extractData(DataSetIterator iterator) {
        iterator.reset();
        List<double[]> allData = new ArrayList<>();
        List<Integer> allLabels = new ArrayList<>();

        while (iterator.hasNext()) {
            List<DataSet> batch = iterator.next().asList();
            for (DataSet ds : batch) {
                INDArray features = ds.getFeatures();
                double[] featureVector = features.reshape(features.length()).toDoubleVector();
                allData.add(featureVector);
                allLabels.add(ds.outcome());
            }
        }

        double[][] featuresArray = allData.toArray(new double[0][]);
        int[] labelsArray = allLabels.stream().mapToInt(Integer::intValue).toArray();
        return new ExtractedData(featuresArray, labelsArray);
    }

    public double[][] getFeatures() {
        return features;
    }

    public int[] getLabels() {
        return labels;
    }
}

