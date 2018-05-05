package com.vishnuvyas.classifiers.svm

import org.scalatest.{ShouldMatchers, FlatSpec}
import com.vishnuvyas.classifiers.Classifier
import java.io.{OutputStream, InputStream}
import java.util

/**
 * Created by vishnu on 4/20/14.
 */
class SVMTrainerSpec extends FlatSpec with ShouldMatchers {

  def time[R](name:String) (block: => R): R = {
    val t0 = System.currentTimeMillis()
    val result = block    // call-by-name
    val t1 = System.currentTimeMillis()
    println(name + ": Elapsed time: " + (t1 - t0) + "ms")
    result
  }

  "A SVMTrainer"  should "be able to train from a dataset to get a classifer" in {
    val trainer = new SVMTrainer(4000,1,0.01f)
    val dataset = new DiabetesDataset

    trainer.setPrinting(true)

    val baselineClassifier = new Classifier[Array[Float],java.lang.Boolean] {
      override def predictions(x : Array[Float]) = {
        val m = new util.HashMap[java.lang.Boolean,java.lang.Float]()
        m.put(false,0)
        m
      }

      override def load(f:InputStream) = ???

      override def save(f:OutputStream) = ???
    }

    val classifer = time("training classifier") {
      trainer.train(dataset)
    }

    val trainingError = dataset.trainingError(classifer)*100/(dataset.count())
    val baselineTrainingError = dataset.trainingError(baselineClassifier)*100/(dataset.count())
    val (tp,fp,tn,fn) = dataset.confusionMatrix(classifer)

    println(f"The precison($tp , $fp) is ${tp*100.0/(tp+fp)}%2.2f%%, recall($fn) = ${tp*100.0/(tp+fn)}%2.2f%%, trainingError = $trainingError%2.2f%%")
    println(f"The baseline error rate was $baselineTrainingError%2.2f%%")
    trainingError should be < baselineTrainingError
  }

}
