package com.vishnuvyas.classifiers.svm;

import com.vishnuvyas.classifiers.Classifier;
import com.vishnuvyas.classifiers.training.Trainer;
import com.vishnuvyas.classifiers.training.dataset.Dataset;
import com.vishnuvyas.optimize.GradientDescentOptimizer;

import java.util.Arrays;

/**
 * Created by vishnu on 4/20/14.
 */
public class SVMTrainer implements Trainer<float[],Boolean> {

    private int niter;
    private float batchFraction;
    private float regularizationParameter;

    private boolean printing = false;

    public void setPrinting(boolean p) {
        printing = p;
    }

    public SVMTrainer(int niter, float batchFraction, float regParam) {
        this.niter = niter;
        this.batchFraction = batchFraction;
        this.regularizationParameter = regParam;
    }

    public static String printWeights(float [] w) {
        StringBuilder sb = new StringBuilder();
        for(float wi : w) {
            sb.append(String.format("%2.2f",wi));
            sb.append(";");
        }

        return sb.toString();
    }

    @Override
    public Classifier<float[], Boolean> train(Dataset<float[], Boolean> dataset) {
        GradientDescentOptimizer optimizer = new GradientDescentOptimizer();
        float [] w = new float[dataset.dim()];
        Arrays.fill(w, 0.0f);
        for(int t = 0; t < niter; ++t) {
            Dataset<float[],Boolean> batch = dataset.sample(batchFraction);
            HingeLoss loss = new HingeLoss(batch,regularizationParameter);
            if(t > 0 && t % 1000 == 0 && printing) {
                System.out.println("The weights at step " + t
                        + " are " + printWeights(w)
                        + " and loss is " + String.format("%.3f", loss.eval(w)));
            }
            w = optimizer.descentStep(loss,w,0.1f,t);
        }

        return new SVMClassifier(w);
    }
}
