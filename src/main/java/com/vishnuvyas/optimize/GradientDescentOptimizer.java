package com.vishnuvyas.optimize;

/**
 * Created by vishnu on 4/19/14.
 */
public class GradientDescentOptimizer {


    /**
     * Computes a single descent step given an objective function, an initial value
     * an alpha (typically 1/stepSize) and the iteration number (t) where it decays
     * the value of the stepSize by the iteration number using (1/t)
     *
     * This step size is assumed to modify the iv inplace and return the same reference
     * as the iv in its return value (if the proximal function of the objective also modifies
     * the vector inplace).
     *
     * @param objectiveFunction the objective function we want to minimize
     * @param iv    the initial value of the minimizer
     * @param alpha the inverse stepSize (1/stepSize)
     * @param t     the iteration count. (for adaptively decreasing the step-size)
     * @return An updated value of the minimizer.
     */
    public float[] descentStep(ObjectiveFunction objectiveFunction, float [] iv, float alpha, int t) {

        // compute the gradient and the step size.
        float [] gradient = objectiveFunction.gradient(iv);
        float stepSize = 1.0f/(alpha*t + 0.1f);

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
