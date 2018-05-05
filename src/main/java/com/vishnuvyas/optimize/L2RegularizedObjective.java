package com.vishnuvyas.optimize;

/**
 * Created by vishnu on 4/20/14.
 */
public abstract class L2RegularizedObjective implements ObjectiveFunction {

    /**
     * Get the regularization parameter
     * @return the regularization parameter.
     */
    public abstract float getRegularizationParam();

    /**
     * Prox function in our case (L2) is simply scaling by the regularization
     * parameter.
     * @param g a point
     * @return the soft-max subject to shrinkage lambda for this point.
     */
    @Override
    public float[] prox(float[] g) {
        double lambda = getRegularizationParam();
        for(int i = 0; i < g.length; ++i) {
            g[i] *= 1/(2*lambda + 1);
        }

        return g;
    }
}
