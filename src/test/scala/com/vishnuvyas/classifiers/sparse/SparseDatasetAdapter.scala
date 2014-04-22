package com.vishnuvyas.classifiers.sparse

import com.vishnuvyas.classifiers.training.dataset.{AbstractDataset, Dataset}
import java.{lang, util}

/**
 * A sparse dataset adapter, takes in a dense dataset and returns a
 * sparse dataset (which uses sparse vectors)
 * Created by vishnu on 4/21/14.
 */
class SparseDatasetAdapter(dataset:Dataset[Array[Double],java.lang.Boolean]) extends AbstractDataset[SparseVector,java.lang.Boolean]{
  override def sample(fraction: Double): Dataset[SparseVector, lang.Boolean] = new SparseDatasetAdapter(dataset.sample(fraction))

  override def getPoint(i: Int): SparseVector = new SparseVector(dataset.getPoint(i))

  override def getLabel(i: Int): lang.Boolean = dataset.getLabel(i)

  override def getLabels: util.Set[lang.Boolean] = dataset.getLabels

  override def dim(): Int = dataset.dim()

  override def count(): Int = dataset.count()
}
