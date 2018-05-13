package com.vishnuvyas.classifiers.training;

import com.vishnuvyas.classifiers.Classifier;
import com.vishnuvyas.dataset.Dataset;

/**
 * Created by vishnu on 4/20/14.
 */
public interface Trainer<V,L> {

    public Classifier<V,L> train(Dataset<V,L> dataset);
}
