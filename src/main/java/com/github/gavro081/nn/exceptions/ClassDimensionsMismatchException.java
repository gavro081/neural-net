package com.github.gavro081.nn.exceptions;

public class ClassDimensionsMismatchException extends Exception{
    private final String message;
    public ClassDimensionsMismatchException(long expected, long passed){
        message = String.format("Number of classes in the dataset: %d, Number of classes in the final layer: %d.",
                expected, passed);
    }

    @Override
    public String getMessage() {
        return message;
    }
}
