package com.vishnuvyas.optimize;

/**
 * Created by vishnu on 4/19/14.
 */
public class GradientDescentOptimizer {

    private static final double STEPSIZE_MIN = 1e-7;

    public double [] descentStep(ObjectiveFunction objectiveFunction, double [] iv, double alpha, int t) {

        // compute the gradient and the step size.
        double [] gradient = objectiveFunction.gradient(iv);
        double stepSize = 1.0/(alpha*t + 0.1);

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
