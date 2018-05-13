package com.vishnuvyas.classifiers.sparse;

import com.vishnuvyas.classifiers.Classifier;
import com.vishnuvyas.classifiers.svm.SVMTrainer;
import com.vishnuvyas.classifiers.training.Trainer;
import com.vishnuvyas.dataset.Dataset;
import com.vishnuvyas.optimize.GradientDescentOptimizer;

import java.util.Arrays;

/**
 * Created by vishnu on 4/21/14.
 */
public class SparseSVMTrainer implements Trainer<SparseVector,Boolean> {
    private float batchFraction;
    private int niter;
    private float regularizationParameter;
    private float learningRate;
    private boolean printing=false;

    public void setPrinting(boolean p) {
        printing = p;
    }


    /**
     * creates a new sparse svm trainer
     * @param niter the number of iterations to run this trainer
     * @param batchFraction the fraction of the data that is used in each iteration
     * @param regularizationParameter the regularization parameter
     */
    public SparseSVMTrainer(int niter,float batchFraction, float regularizationParameter,float learningRate) {
        this.niter = niter;
        this.batchFraction = batchFraction;
        this.regularizationParameter = regularizationParameter;
        this.learningRate = learningRate;
    }

    @Override
    public Classifier<SparseVector, Boolean> train(Dataset<SparseVector, Boolean> dataset) {
        GradientDescentOptimizer optimizer = new GradientDescentOptimizer();
        float [] w = new float[dataset.dim()];
        Arrays.fill(w,0.0f);

        for(int t = 0; t < niter; ++t) {
            Dataset<SparseVector,Boolean> sample = dataset.sample(batchFraction);
            SparseHingeLoss objectiveFunction = new SparseHingeLoss(sample,regularizationParameter);

            if(t%1000==0 && t > 0 && printing) {
                System.out.println("The weights at step " + t
                        + " are " + SVMTrainer.printWeights(w)
                        + " and loss is " + String.format("%.3f", objectiveFunction.eval(w)));
            }

            w = optimizer.descentStep(objectiveFunction,w,learningRate,t);
        }
        return new SparseSVMClassifier(w);
    }
}
