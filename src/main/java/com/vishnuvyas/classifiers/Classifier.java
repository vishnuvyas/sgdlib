package com.vishnuvyas.classifiers;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.Map;

/**
 * Created by vishnu on 4/20/14.
 */
public interface Classifier<VectorType, LabelType> {
    public Map<LabelType,Float> predictions(VectorType point);

    public void save(OutputStream os);
    public void load(InputStream is);
}
