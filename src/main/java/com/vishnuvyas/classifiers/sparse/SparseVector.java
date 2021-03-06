package com.vishnuvyas.classifiers.sparse;

import java.util.Arrays;
import java.util.Map;
import java.util.TreeMap;

/**
 * A SparseVector, which is implemented using 2 parallel arrays, one for
 * indices and other for values. The best way to use this class is to first
 * create the sparse vector  and optimize it using the optimize() call and then
 * using the optimized version of the vector, where access (both sequential) and
 * random access is much faster.
 *
 * Created by vishnu on 4/20/14.
 */
public class SparseVector {

    private float [] values;
    private int [] indices;
    private int dim;
    private int tail = 0;
    private boolean optimized = false;

    private static final int DEFAULT_ALLOC_SIZE = 1024;
    private static final int INDEX_DEFAULT = -1;

    /**
     * Create a sparse vector of a specified dimension
     *
     * @param dim the dimensions of the sparse vector.
     */
    public SparseVector(int dim) {
        this.dim = dim;
        int initialAllocSize = Math.min(dim, DEFAULT_ALLOC_SIZE);
        values = new float[initialAllocSize];
        indices = new int[initialAllocSize];
        tail = 0;

        Arrays.fill(values,0.0f);
        Arrays.fill(indices,INDEX_DEFAULT);
    }

    /**
     * Create a sparse vector from a dense vector.
     * @param vec the dense vector
     */
    public SparseVector(float [] vec) {
        this.dim = vec.length;
        values = new float[dim];
        indices = new int[dim];
        tail = 0;
        for(int i = 0; i < vec.length; ++i) {
            if(vec[i] != 0) {
                indices[tail] = i;
                values[tail] = vec[i];
                tail++;
            }
        }
    }


    /**
     * Gets the current cardinality (number of non-zeros) in the sparse
     * vector.
     *
     * @return the number of sparse vector.
     */
    public int getCardinality() {
        return tail;
    }


    /**
     * Returns the dimensions with this vector was created with.
     * @return the dimension of this sparse vector.
     */
    public int getDim()  {
        return dim;
    }


    /**
     * set the value of the index i to d
     * @param i the index
     * @param d the float
     */
    public void set(int i, float d) {
        if(d == 0)
            return;

        if(getCardinality() >= indices.length) {
            int expansionSize = Math.min(DEFAULT_ALLOC_SIZE,dim-indices.length);
            expandArrays(expansionSize);
        }

        optimized = false;
        indices[tail] = i;
        values[tail] = d;
        tail++;
    }


    /**
     * get the value at index i
     * @param i the index i
     * @return the value at the point or 0
     */
    public float get(int i) {
        int ii = searchForIndex(i);
        if(ii < 0) { return  0; }
        return values[ii];
    }

    /**
     * same as get, but lets you substitute your own default value
     * @param i the index
     * @param defaultValue  the default value if a value is not already set
     * @return
     */
    public float getOrElse(int i, float defaultValue) {
        int ii = searchForIndex(i);
        if(ii < 0) {
            return defaultValue;
        }
        return values[ii];
    }


    /**
     * Optimize optimizes (sorts) the arrays in such a way that scans can happen
     * really fast, useful when we are doing a lot of dot-products.
     *
     * @return - the new cardinality of this array (which should be the same as the old one)
     */
    public int optimize() {
        TreeMap<Integer,Float> k = new TreeMap<Integer, Float>();
        for(int  i = 0; i < getCardinality(); ++i) {
            k.put(indices[i],values[i]);
        }

        int newTail = 0;
        for(Map.Entry<Integer,Float> e : k.entrySet()) {
            indices[newTail] = e.getKey();
            values[newTail] = e.getValue();
            newTail++;
        }

        optimized = true;
        return newTail;
    }

    /**
     * Checks if the vector is sorted and optimized.
     * @return true if the vector has been optimized.
     */
    public boolean isOptimized() {
        return optimized;
    }



    /**
     * dot product with another dense vector
     * @param denseVector  the dense vector
     * @return the dot product between the current vector and the dense vector.
     */
    public float dot(float [] denseVector) {
        assert denseVector.length == dim : "Dimensions of vectors should match for the dot-product";
        if(isOptimized()) {
            return optimizedDot(denseVector);
        } else {
            // this is the slow path that needs to optimize the vector first
            // and then do an optimized dot-product.
            optimize();
            return optimizedDot(denseVector);
        }
    }

    // package public methods go here - mostly used by tests to
    // inspect the internals of this object

    int [] getIndices() {
        return indices;
    }

    float [] getValues() {
        return values;
    }


    // private methods

    private int searchForIndex(int index) {
        if(index > dim)
            throw new ArrayIndexOutOfBoundsException(index);
        else if(optimized) {
            return Arrays.binarySearch(indices,0,tail,index);
        }

        int c = -1;
        for(int i = 0; i < tail; ++i) {
            if(indices[i] == index) {
                c = i;
                break;
            }
        }

        return c;
    }

    private void expandArrays(int expansionSize) {
        if(expansionSize <= 0) {
            throw new RuntimeException("Requested expansion size of 0 - dimension is too low");
        }

        int [] newIndexArray = new int[indices.length+expansionSize];
        float [] newValues = new float[indices.length+expansionSize];

        Arrays.fill(newIndexArray,INDEX_DEFAULT);
        Arrays.fill(newValues,0f);

        for(int i = 0; i < indices.length; ++i) {
            newIndexArray[i] = indices[i];
            newValues[i] = values[i];
        }

        optimized = false;
        values = newValues;
        indices = newIndexArray;

    }

    private float optimizedDot(float [] denseVector) {
        float sum = 0;
        int ptr = 0;
        for(int i = 0; i < denseVector.length && ptr < getCardinality(); ++i) {
            while(indices[ptr] < i) {
                ptr++;
            }
            if(indices[ptr] > i)
                continue;
            else {
                sum += (values[ptr]*denseVector[i]);
            }
        }
        return sum;
    }
}
