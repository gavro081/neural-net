package com.github.gavro081.nn.utils;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import com.github.gavro081.nn.NeuralNet;

public class ImagePredictor {
    private final NeuralNet model;
    private final int targetWidth;
    private final int targetHeight;
    private boolean invertColors = false;

    /**
     * Creates an ImagePredictor with a loaded model
     * @param model The trained NeuralNet model
     * @param targetWidth Expected image width (e.g., 28 for MNIST)
     * @param targetHeight Expected image height (e.g., 28 for MNIST)
     */
    public ImagePredictor(NeuralNet model, int targetWidth, int targetHeight) {
        this.model = model;
        this.targetWidth = targetWidth;
        this.targetHeight = targetHeight;
    }

    /**
     * Sets whether to invert colors (useful for black digits on white background)
     * MNIST is trained on white digits on black background
     * @param invert true to invert colors (black->white, white->black)
     */
    public void setInvertColors(boolean invert) {
        this.invertColors = invert;
    }

    /**
     * Reads a PNG image and converts it to a flattened grayscale array
     * @param imagePath Path to the PNG image file
     * @return Flattened array of pixel values normalized to [0, 1]
     * @throws IOException If the image cannot be read
     */
    public double[] readAndPreprocessImage(String imagePath) throws IOException {
        File imageFile = new File(imagePath);
        if (!imageFile.exists()) {
            throw new IOException("Image file not found: " + imagePath);
        }

        BufferedImage image = ImageIO.read(imageFile);
        if (image == null) {
            throw new IOException("Failed to read image: " + imagePath);
        }

        // Resize image if necessary
        if (image.getWidth() != targetWidth || image.getHeight() != targetHeight) {
            image = resizeImage(image, targetWidth, targetHeight);
        }

        // Convert to grayscale and flatten
        return imageToGrayscaleArray(image);
    }

    /**
     * Resizes an image to the target dimensions
     */
    private BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) {
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D graphics2D = resizedImage.createGraphics();
        graphics2D.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        graphics2D.drawImage(originalImage, 0, 0, targetWidth, targetHeight, null);
        graphics2D.dispose();
        return resizedImage;
    }

    /**
     * Converts an image to a normalized grayscale array
     */
    private double[] imageToGrayscaleArray(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        double[] pixels = new double[width * height];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                
                // Extract RGB components
                int red = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int blue = rgb & 0xFF;
                
                // Convert to grayscale using luminosity method
                double gray = (0.299 * red + 0.587 * green + 0.114 * blue);
                
                // Normalize to [0, 1] range
                double normalized = gray / 255.0;
                
                // Invert if needed (for black digits on white background)
                pixels[y * width + x] = invertColors ? (1.0 - normalized) : normalized;
            }
        }

        return pixels;
    }

    /**
     * Makes a prediction for a single image and returns full output probabilities
     * @param imagePath Path to the PNG image file
     * @return Array of probabilities for each class
     * @throws IOException If the image cannot be read
     */
    public double[] predictWithProbabilities(String imagePath) throws IOException {
        double[] imageData = readAndPreprocessImage(imagePath);
        return model.forward(imageData);
    }

    /**
     * Makes a prediction for a single image
     * @param imagePath Path to the PNG image file
     * @return Predicted class label
     * @throws IOException If the image cannot be read
     */
    public int predictSingle(String imagePath) throws IOException {
        double[] imageData = readAndPreprocessImage(imagePath);
        double[][] batchData = new double[][]{imageData};
        int[] predictions = model.predict(batchData);
        return predictions[0];
    }

    /**
     * Debug method to inspect pixel values of an image
     * @param imagePath Path to the PNG image file
     * @throws IOException If the image cannot be read
     */
    public void debugImagePixels(String imagePath) throws IOException {
        double[] pixels = readAndPreprocessImage(imagePath);
        
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        double sum = 0;
        
        for (double pixel : pixels) {
            min = Math.min(min, pixel);
            max = Math.max(max, pixel);
            sum += pixel;
        }

        for (int i = 0; i < Math.min(56, pixels.length); i++) {
            if ((i + 1) % 28 == 0) System.out.println();
        }
        System.out.println();
    }

    /**
     * Makes predictions for multiple images in a directory
     * @param directoryPath Path to directory containing PNG images
     * @return Array of predictions corresponding to each image
     * @throws IOException If images cannot be read
     */
    public PredictionResult[] predictFromDirectory(String directoryPath) throws IOException {
        File directory = new File(directoryPath);
        if (!directory.exists() || !directory.isDirectory()) {
            throw new IOException("Invalid directory: " + directoryPath);
        }

        File[] imageFiles = directory.listFiles((dir, name) -> 
            name.toLowerCase().endsWith(".png") || name.toLowerCase().endsWith(".jpg") || name.toLowerCase().endsWith(".jpeg")
        );

        if (imageFiles == null || imageFiles.length == 0) {
            return new PredictionResult[0];
        }

        PredictionResult[] results = new PredictionResult[imageFiles.length];
        
        for (int i = 0; i < imageFiles.length; i++) {
            try {
                int prediction = predictSingle(imageFiles[i].getAbsolutePath());
                results[i] = new PredictionResult(imageFiles[i].getName(), prediction, null);
            } catch (IOException e) {
                results[i] = new PredictionResult(imageFiles[i].getName(), -1, e.getMessage());
            }
        }

        return results;
    }

    /**
     * Result container for predictions
     */
    public static class PredictionResult {
        private final String filename;
        private final int prediction;
        private final String error;

        public PredictionResult(String filename, int prediction, String error) {
            this.filename = filename;
            this.prediction = prediction;
            this.error = error;
        }

        public String getFilename() {
            return filename;
        }

        public int getPrediction() {
            return prediction;
        }

        public boolean hasError() {
            return error != null;
        }

        public String getError() {
            return error;
        }

        @Override
        public String toString() {
            if (hasError()) {
                return String.format("%s: ERROR - %s", filename, error);
            }
            return String.format("%s: Predicted class = %d", filename, prediction);
        }
    }

    /**
     * Utility method to load a model and create a predictor
     * @param modelPath Path to the saved model file
     * @param targetWidth Expected image width
     * @param targetHeight Expected image height
     * @return ImagePredictor instance ready to make predictions
     */
    public static ImagePredictor fromSavedModel(String modelPath, int targetWidth, int targetHeight) 
            throws IOException, ClassNotFoundException {
        NeuralNet model = NeuralNet.load(modelPath);
        return new ImagePredictor(model, targetWidth, targetHeight);
    }
}
