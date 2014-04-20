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
     * @param g the gradient
     * @return the proximal of the gradient
     */
    public double[] prox(double [] g);

}
