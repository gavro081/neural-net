package com.github.gavro081.nn.exceptions;

public class DimensionMismatchException extends Exception{
    private final String message;
    public DimensionMismatchException(long inputDimension, int expectedDimensions, int index) {
        message = String.format("Layer %d expected %d features. Received %d.", index, expectedDimensions, inputDimension);
    }

    @Override
    public String getMessage() {
        return message;
    }
}
