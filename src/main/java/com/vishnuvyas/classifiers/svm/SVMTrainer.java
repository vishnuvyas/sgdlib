package com.vishnuvyas.classifiers.svm;

import com.vishnuvyas.classifiers.Classifier;
import com.vishnuvyas.classifiers.training.Trainer;
import com.vishnuvyas.classifiers.training.dataset.Dataset;
import com.vishnuvyas.optimize.GradientDescentOptimizer;

import java.util.Arrays;

/**
 * Created by vishnu on 4/20/14.
 */
public class SVMTrainer implements Trainer<double[],Boolean> {

    private int niter;
    private double batchFraction;
    private double regularizationParameter;

    private boolean printing = false;

    public void setPrinting(boolean p) {
        printing = p;
    }

    public SVMTrainer(int niter, double batchFraction, double regParam) {
        this.niter = niter;
        this.batchFraction = batchFraction;
        this.regularizationParameter = regParam;
    }

    public static String printWeights(double [] w) {
        StringBuilder sb = new StringBuilder();
        for(double wi : w) {
            sb.append(String.format("%2.2f",wi));
            sb.append(";");
        }

        return sb.toString();
    }

    @Override
    public Classifier<double[], Boolean> train(Dataset<double[], Boolean> dataset) {
        GradientDescentOptimizer optimizer = new GradientDescentOptimizer();
        double [] w = new double[dataset.dim()];
        Arrays.fill(w, 0.0d);
        for(int t = 0; t < niter; ++t) {
            Dataset<double[],Boolean> batch = dataset.sample(batchFraction);
            HingeLoss loss = new HingeLoss(batch,regularizationParameter);
            if(t > 0 && t % 1000 == 0 && printing) {
                System.out.println("The weights at step " + t
                        + " are " + printWeights(w)
                        + " and loss is " + String.format("%.3f", loss.eval(w)));
            }
            w = optimizer.descentStep(loss,w,0.1,t);
        }

        return new SVMClassifier(w);
    }
}
