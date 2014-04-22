package com.vishnuvyas.classifiers.training.dataset;

import com.vishnuvyas.classifiers.Classifier;

import java.util.Map;

/**
 * Created by vishnu on 4/21/14.
 */
public abstract class AbstractDataset<V,L> implements Dataset<V,L> {

    public int trainingError(Classifier<V,L> classifier) {
        int nerrors = 0;

        for(int i = 0; i < count(); ++i) {
            V xi = getPoint(i);
            L yi = getLabel(i);
            Map<L,Double> predictions = classifier.predictions(xi);

            // identify the best prediction from the result
            L bestPred = null;
            double currentBest = Double.NEGATIVE_INFINITY;
            for(Map.Entry<L,Double> e : predictions.entrySet() ) {
                if(bestPred == null) {
                    bestPred = e.getKey();
                    currentBest = e.getValue();
                } else {
                    if(e.getValue() > currentBest) {
                        bestPred = e.getKey();
                        currentBest = e.getValue();
                    }
                }
            }

            // if best prediction is the same as the actual , then we have
            // a winner.
            if(!bestPred.equals(yi))
                ++nerrors;
        }

        return nerrors;
    }
}
