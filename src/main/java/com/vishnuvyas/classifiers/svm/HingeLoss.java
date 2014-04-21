package com.vishnuvyas.classifiers.svm;

import com.vishnuvyas.classifiers.training.dataset.Dataset;
import com.vishnuvyas.optimize.L1RegularizedObjective;
import com.vishnuvyas.optimize.L2RegularizedObjective;

import java.util.Arrays;

/**
 * HingeLoss is the linear SVM loss function. Since linear svm's are
 * binary classifiers, its important the dataset return boolean values
 * for the positive and negative class.
 *
 * Created by vishnu on 4/20/14.
 */
public class HingeLoss extends L2RegularizedObjective<double[]> {

    private Dataset<double[],Boolean> dataset;
    private double regularizationParam;

    private double booleanLoss(boolean b) {
        if(b)
            return 1;
        else
            return -1;
    }

    private double dot(double [] x, double [] y) {
        assert x.length == y.length : "The dimensions of the two vectors in a dot product should be same";

        double d = 0;
        for(int i = 0; i < x.length ; ++i)
            d += (x[i]*y[i]);

        return d;
    }


    public HingeLoss(Dataset<double[],Boolean> dataset,double reg) {
        this.dataset = dataset;
        this.regularizationParam = reg;
    }

    @Override
    public double[] gradient(double[] x) {
        double [] gradient = new double[x.length];
        Arrays.fill(gradient,0.0);

        for(int i = 0; i < dataset.count(); ++i) {
            double [] xi = dataset.getPoint(i);
            double yi = booleanLoss(dataset.getLabel(i));
            double v = dot(xi,x)*yi;
            if(v < 1) {
                // then this is an error we need to correct and the update
                // is -yx
                for(int  j = 0; j < xi.length; ++j) {
                    gradient[j] -= yi * xi[j];
                }
            }
        }

        return gradient;
    }

    @Override
    public double getRegularizationParam() {
        return regularizationParam;
    }

    @Override
    public double eval(double[] x) {
        double loss = 0;
        for(int i = 0; i < dataset.count(); ++i) {
            loss += Math.max(0,(1 - dot(dataset.getPoint(i),x)*booleanLoss(dataset.getLabel(i))));
        }

        return loss;
    }
}
