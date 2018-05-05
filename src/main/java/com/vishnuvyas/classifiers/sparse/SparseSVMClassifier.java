package com.vishnuvyas.classifiers.sparse;

import com.vishnuvyas.classifiers.Classifier;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by vishnu on 4/21/14.
 */
public class SparseSVMClassifier implements Classifier<SparseVector,Boolean> {

    private float [] w;

    public SparseSVMClassifier(float [] w) {
        this.w = w;
    }

    @Override
    public Map<Boolean, Float> predictions(SparseVector point) {
        float v = point.dot(w);
        Map<Boolean,Float> m = new HashMap<Boolean, Float>();
        if(v > 0) {
            m.put(true,v);
        } else {
            m.put(false,v);
        }

        return m;
    }

    @Override
    public void save(OutputStream os) {
        throw new UnsupportedOperationException("Not Implemented");

    }

    @Override
    public void load(InputStream is) {
        throw new UnsupportedOperationException("Not Implemented");
    }
}
