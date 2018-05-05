package com.vishnuvyas.classifiers.sparse

import org.scalatest.{ShouldMatchers, FlatSpec}
import com.vishnuvyas.classifiers.svm.{SVMTrainer, DiabetesDataset}

/**
 * Created by vishnu on 4/21/14.
 */
class SparseSVMTrainerSpec extends FlatSpec with ShouldMatchers {

  "A SparseSVMTrainer" should "be able to reproduce the same results as a regular svm trainer" in {
    val dataset = new DiabetesDataset
    val sparseDataset = new SparseDatasetAdapter(dataset)

    val trainer = new SVMTrainer(4000,1,0.01f)
    val sparseTrainer = new SparseSVMTrainer(4000,1,0.01f,0.1f)

    val classifier = trainer.train(dataset)
    val sparseClassifer = sparseTrainer.train(sparseDataset)

    val trainingError = dataset.trainingError(classifier)
    val sparseTrainingError = sparseDataset.trainingError(sparseClassifer)

    trainingError should be(sparseTrainingError)
  }

}
