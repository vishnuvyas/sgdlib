package com.vishnuvyas.optimize;

/**
 * Created by vishnu on 4/20/14.
 */
public abstract class L2RegularizedObjective<T> implements ObjectiveFunction<T> {

    /**
     * Get the regularization parameter
     * @return the regularization parameter.
     */
    public abstract double getRegularizationParam();

    /**
     * Prox function in our case (L2) is simply scaling by the regularization
     * parameter.
     * @param g a point
     * @return the soft-max subject to shrinkage lambda for this point.
     */
    @Override
    public double[] prox(double[] g) {
        double lambda = getRegularizationParam();
        for(int i = 0; i < g.length; ++i) {
            g[i] *= 1/(2*lambda + 1);
        }

        return g;
    }
}
