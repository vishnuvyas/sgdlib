package com.vishnuvyas.optimize;

/**
 * Created by vishnu on 4/19/14.
 */
public class GradientDescentOptimizer {

    public double [] descentStep(ObjectiveFunction objectiveFunction, double [] iv, double alpha, int t) {
        double [] gradient = objectiveFunction.gradient(iv);
        double stepSize = Math.pow(alpha, 1.0/(0.1 + Math.abs(t)));

        // scale the gradient by the step-size
        for(int i = 0; i < gradient.length; ++i)
            gradient[i] *= -stepSize;

        // now get the proximal of our updated initial value and set it to the
        // updated value.
        for(int i = 0; i < iv.length; ++i) {
            iv[i] += gradient[i];
        }

        return objectiveFunction.prox(iv);
    }

}
