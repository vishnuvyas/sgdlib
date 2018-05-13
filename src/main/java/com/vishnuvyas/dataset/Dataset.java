package com.vishnuvyas.dataset;

import java.util.Set;

/**
 * A Dataset is a way to have random access to our data with the ability move in the
 * dataset whatever way we want.
 *
 * Created by vishnu on 4/20/14.
 */
public interface Dataset<VectorType,LabelType> {

    /**
     * Count the number of points on this dataset.
     * @return the number of points in this dataset.
     */
    public int count();


    /**
     * Return the dimensions of the input vectors in this dataset
     *
     * @return the dimensions of the input vectors in this dataset.
     */
    public int dim();

    /**
     * Returns a list of labels associated with this dataset
     *
     * @return a list of labels associated with this dataset.
     */
    public Set<LabelType> getLabels();


    /**
     * Get the label associated with the data point i. For binary classification
     * the label type is expected to be boolean. For the multi-class classification
     * the label is expected to be strings - where each string represents the class
     * label. And for multi-output classification the labeltype is assuemd to be a
     * string that is semi-colon separated and the classifiers/evaluators tend to
     * interepret them appropriately.
     *
     * @return the label associated with this data-point.
     */
    public LabelType getLabel(int i);


    /**
     * Get the data point correspoinding to position i in the dataset.
     * @param i the position i in the dataset we are looking at.
     * @return a vector corresponding to that point in the dataset
     */
    public VectorType getPoint(int i);


    /**
     * a way to sample a subset of the data (based on fraction) that returns
     * another dataset.
     * @param fraction the sampling fraction.
     * @return a subset of the dataset sampled.
     */
    public Dataset<VectorType,LabelType> sample(float fraction);
}
