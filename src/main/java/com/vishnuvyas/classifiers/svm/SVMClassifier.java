package com.vishnuvyas.classifiers.svm;

import com.vishnuvyas.classifiers.Classifier;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by vishnu on 4/20/14.
 */
public class SVMClassifier implements Classifier<float[],Boolean> {
    private float [] w;

    public float[] getWeights() { return w; }

    public SVMClassifier(float [] w) {
        this.w = w;
    }

    @Override
    public Map<Boolean, Float> predictions(float[] point) {
        assert point.length == w.length  : "The dimensions of the wieght vector does not match point";
        Map<Boolean,Float> m = new HashMap<Boolean, Float>();
        float s = 0;
        for(int i = 0; i < point.length; ++i)
            s += (point[i]*w[i]);
        if(s>0) {
            m.put(true,s);
        } else {
            m.put(false,s);
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
