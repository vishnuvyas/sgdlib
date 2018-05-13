package com.vishnuvyas.classifiers.sparse;

import com.vishnuvyas.classifiers.svm.HingeLoss;
import com.vishnuvyas.dataset.Dataset;
import com.vishnuvyas.optimize.L2RegularizedObjective;

import java.util.Arrays;

/**
 * Created by vishnu on 4/21/14.
 */
public class SparseHingeLoss extends L2RegularizedObjective {

    private Dataset<SparseVector,Boolean> dataset;
    private float regularizationParam;
    private float[] lastGradient;

    /**
     * Create a new sparse hinge loss, used in our trainers.
     * @param dataset the dataset over which the loss is constructed
     * @param lambda  the capacity parameter for this svm (typically C)
     */
    public SparseHingeLoss(Dataset<SparseVector,Boolean> dataset, float lambda) {
        this.dataset = dataset;
        this.regularizationParam = lambda;
        this.lastGradient = new float[dataset.dim()];
    }

    @Override
    public float getRegularizationParam() {
        return regularizationParam;
    }

    /**
     * The gradient with sparse datasets is similar to the
     * regular gradient, except that it uses the get operator
     * instead of the [] (indexing operator).
     *
     * @param x - the point at which the gradient is computed.
     * @return  the gradient at the point x
     */
    @Override
    public float[] gradient(float[] x) {
        Arrays.fill(lastGradient,0);
        for(int i = 0; i < dataset.count(); ++i) {
            SparseVector sv = dataset.getPoint(i);
            float yi = HingeLoss.booleanLoss(dataset.getLabel(i));
            float v = sv.dot(x)*yi;
            if(v < 1) {
                for(int j = 0; j < dataset.dim(); ++j) {
                    lastGradient[j] -= yi * sv.get(j);
                }
            }
        }

        return lastGradient;
    }

    /**
     * Given the current weight vector,
     * @param x  the input point x
     * @return
     */
    @Override
    public float eval(float[] x) {
        float loss = 0;
        for(int  i = 0; i < dataset.count(); ++i) {
            loss += Math.max(0,1-dataset.getPoint(i).dot(x)* HingeLoss.booleanLoss(dataset.getLabel(i)));
        }

        return loss;
    }

}
