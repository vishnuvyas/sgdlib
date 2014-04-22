package com.vishnuvyas.optimize;

/**
 * Created by vishnu on 4/20/14.
 */
abstract public class L1RegularizedObjective implements ObjectiveFunction {


    /**
     * Get the regularization parameter
     * @return the regularization parameter.
     */
    public abstract double getRegularizationParam();

    /**
     * Prox function in our case simply implements the soft-max functin
     * with a shrinkage parameter of lambda. This correspoinds to a L1-regularized
     * loss. The updates are done inplace, and hence not thread-safe.
     *
     * @param g a point
     * @return the soft-max subject to shrinkage lambda for this point.
     */
    @Override
    public double[] prox(double[] g) {
        double lambda = getRegularizationParam();
        for(int i = 0; i < g.length; ++i) {
            if(Math.abs(g[i]) <= lambda)
                g[i] = 0.0;
            else if(g[i] > lambda) {
                // since we shrink the value towards lambda, and lambda is assumed
                // positive, we have to shrink g[i] by lambda
                g[i] -= lambda;
            } else  {
                // in this case g[i] is negative, so we are going to shrink it
                // upwards to zero.
                g[i] += lambda;
            }
        }

        return g;
    }

}
