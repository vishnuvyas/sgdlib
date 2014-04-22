package com.vishnuvyas.classifiers.svm;

import com.vishnuvyas.classifiers.Classifier;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by vishnu on 4/20/14.
 */
public class SVMClassifier implements Classifier<double[],Boolean> {
    private double [] w;

    public double[] getWeights() { return w; }

    public SVMClassifier(double [] w) {
        this.w = w;
    }

    @Override
    public Map<Boolean, Double> predictions(double[] point) {
        assert point.length == w.length  : "The dimensions of the wieght vector does not match point";
        Map<Boolean,Double> m = new HashMap<Boolean, Double>();
        double s = 0;
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
