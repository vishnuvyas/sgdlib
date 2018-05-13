package com.vishnuvyas.classifiers.svm;

import com.vishnuvyas.dataset.Dataset;
import com.vishnuvyas.optimize.L2RegularizedObjective;

import java.util.Arrays;

/**
 * HingeLoss is the linear SVM loss function. Since linear svm's are
 * binary classifiers, its important the dataset return boolean values
 * for the positive and negative class.
 *
 * Created by vishnu on 4/20/14.
 */
public class HingeLoss extends L2RegularizedObjective {

    private Dataset<float[],Boolean> dataset;
    private float regularizationParam;
    private float [] lastGradient;


    private float dot(float [] x, float [] y) {
        assert x.length == y.length : "The dimensions of the two vectors in a dot product should be same";

        float d = 0;
        for(int i = 0; i < x.length ; ++i)
            d += (x[i]*y[i]);

        return d;
    }


    public HingeLoss(Dataset<float[],Boolean> dataset,float reg) {
        this.dataset = dataset;
        this.regularizationParam = reg;
        this.lastGradient = new float[dataset.dim()];
    }

    @Override
    public float[] gradient(float[] x) {
        Arrays.fill(lastGradient,0.0f);

        for(int i = 0; i < dataset.count(); ++i) {
            float [] xi = dataset.getPoint(i);
            float yi = booleanLoss(dataset.getLabel(i));
            float v = dot(xi,x)*yi;
            if(v < 1) {
                // then this is an error we need to correct and the update
                // is -yx
                for(int  j = 0; j < dataset.dim(); ++j) {
                    lastGradient[j] -= yi * xi[j];
                }
            }
        }

        return lastGradient;
    }

    @Override
    public float getRegularizationParam() {
        return regularizationParam;
    }

    @Override
    public float eval(float[] x) {
        float loss = 0;
        for(int i = 0; i < dataset.count(); ++i) {
            loss += Math.max(0,(1 - dot(dataset.getPoint(i),x)*booleanLoss(dataset.getLabel(i))));
        }

        return loss;
    }

    public static float booleanLoss(boolean b) {
        if(b)
            return 1;
        else
            return -1;
    }
}
