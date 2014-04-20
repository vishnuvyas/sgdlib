package com.vishnuvyas.optimize;

/**
 * An objective function is the function which gets optimized using our gradient descent
 * optimizer.
 *
 * Every objective function should be "differentiable" (atleast, we must be able to get a sub-gradient)
 * also, there is an proximal function, which given the gradient, computes the proximal of the gradient
 *
 * Created by vishnu on 4/19/14.
 */
public interface ObjectiveFunction {

    /**
     * Compute the gradient at point x.
     * @param x - the point at which the gradient is computed.
     * @return  the gradient at the point x
     */
    public double[] gradient(double [] x);


    /**
     * Given the gradient g, compute its proximal (for simple cases proximal can be an identity)
     * @param g a point
     * @return the proximal of the input
     */
    public double[] prox(double [] g);


    /**
     * Evaluate the function at the given point (x). This function can left to have a dummy
     * implementation, because the actual gradient descent doesn't use this function directly.
     * @param x  the input point x
     * @return the value of the function at x
     */
    public double eval(double [] x);

}
