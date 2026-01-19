package com.github.gavro081.nn;

import java.io.File;
import java.util.List;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import com.github.gavro081.nn.layers.impl.LinearLayer;
import com.github.gavro081.nn.layers.impl.ReluLayer;
import com.github.gavro081.nn.loss.impl.MulticlassCrossEntropy;
import com.github.gavro081.nn.optimizer.impl.AdamOptimizer;
import com.github.gavro081.nn.utils.ExtractedData;
import static com.github.gavro081.nn.utils.ExtractedData.extractData;

public class Main {
    public static void main(String[] args) throws Exception {
        final int BATCH_SIZE = 64;
        final int EPOCHS = 10;
        final int INPUT_DIMENSIONS = 28 * 28;
        final String MODEL_PATH = "mnist_model.nn.gz";
        final boolean LOAD_EXISTING_MODEL = false;

        NeuralNet nn;

        if (LOAD_EXISTING_MODEL && new File(MODEL_PATH).exists()) {
            System.out.println("Loading existing model...");
            nn = NeuralNet.load(MODEL_PATH);
        } else {
            System.out.println("Training new model...");
            nn = new NeuralNet();
            nn.addLayer(new LinearLayer(INPUT_DIMENSIONS, 128));
            nn.addLayer(new ReluLayer());
            nn.addLayer(new LinearLayer(128, 64));
            nn.addLayer(new ReluLayer());
            nn.addLayer(new LinearLayer(64, 32));
            nn.addLayer(new ReluLayer());
            nn.addLayer(new LinearLayer(32, 10));

            nn.setLossFunction(new MulticlassCrossEntropy());
            nn.setOptimizer(new AdamOptimizer(0.001));

            DataSetIterator mnistTrain = new MnistDataSetIterator(BATCH_SIZE, true, 12345);

            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                System.out.printf("\n===Epoch %d/%d ===%n", epoch + 1, EPOCHS);
                mnistTrain.reset();

                int batchCount = 0;
                double epochLoss = 0.0;

                while (mnistTrain.hasNext()) {
                    List<DataSet> batch = mnistTrain.next().asList();

                    double[][] trainX = new double[batch.size()][28 * 28];
                    int[] trainY = new int[batch.size()];

                    for (int i = 0; i < batch.size(); i++) {
                        DataSet ds = batch.get(i);
                        INDArray features = ds.getFeatures();
                        double[] featureVector = features.reshape(features.length()).toDoubleVector();

                        trainX[i] = featureVector;
                        trainY[i] = ds.outcome();
                    }

                    nn.fit(trainX, trainY);
                    epochLoss += nn.getLastLoss();
                    batchCount++;

                    if (batchCount % 100 == 0) {
                        System.out.printf("Batch %d, Avg Loss: %.4f%n", batchCount, epochLoss / batchCount);
                    }
                }

                System.out.printf("Epoch Loss: %.4f%n", epochLoss / batchCount);
            }

            nn.save(MODEL_PATH);
        }

        System.out.println("\nEvaluating on test set...");
        DataSetIterator mnistTest = new MnistDataSetIterator(BATCH_SIZE, false, 12345);
        ExtractedData testData = extractData(mnistTest);
        double testAccuracy = nn.evaluate(testData.getFeatures(), testData.getLabels());
        System.out.printf("Test Accuracy: %.2f%%%n", testAccuracy);
    }
}