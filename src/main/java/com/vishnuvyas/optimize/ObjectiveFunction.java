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
public interface ObjectiveFunction<VectorType> {

    /**
     * Compute the gradient at point x. (This assumes that the input value is not modified)
     * @param x - the point at which the gradient is computed.
     * @return  the gradient at the point x
     */
    public double[] gradient(final double [] x);


    /**
     * Given the a point g, compute its proximal (for simple cases proximal can be an identity).
     * Assumes that the point can be modified in place. This is useful when trying to deal with
     * high dimensional problems so that the updates take inplace, but at the same time, makes this
     * call thread unsafe.
     *
     * @param g a point
     * @return the proximal of the input
     */
    public double[] prox(VectorType g);


    /**
     * Evaluate the function at the given point (x). This function can left to have a dummy
     * implementation, because the actual gradient descent doesn't use this function directly.
     * @param x  the input point x
     * @return the value of the function at x
     */
    public double eval(VectorType x);

}
