package com.vishnuvyas.classifiers.svm

import com.vishnuvyas.classifiers.training.dataset.{AbstractDataset, Dataset}
import scala.io.Source
import java.util
import collection.JavaConversions._
import scala.util.Random
import com.vishnuvyas.classifiers.Classifier

/**
 * Created by vishnu on 4/20/14.
 */
class DiabetesDataset extends AbstractDataset[Array[Float],java.lang.Boolean] {

  val dataFileStream = getClass.getResourceAsStream("/pima-indians-diabetes.data")
  val fieldNames = Array("npreg","glucose","dbp","tricep","insulin2hr","bmi","dbpf","age","diabetec")
  val fieldDescriptions = Map(
    "npreg" -> "Number of times pregnant",
    "glucose" -> "Plasma glucose concentration a 2 hours in an oral glucose tolerance test",
    "dbp" -> "Diastolic blood pressure (mm Hg)",
    "tricep" -> "Triceps skin fold thickness (mm)",
    "insulin2hr" -> "2-Hour serum insulin (mu U/ml)",
    "bmi" -> "Body mass index (weight in kg/(height in m)^2)",
    "dbpf" -> "Diabetes pedigree function",
    "age" -> "Age (years)",
    "diabetic" -> "Class variable (0 or 1)"
  )

  val dataset = initialize()

  def initialize() = {
    val rawdataset = Source.fromInputStream(dataFileStream).getLines().map { line =>
      val a = line.trim().split(",").map(_.toFloat)
      val q = a.take(8).flatMap(ai => a.take(8).map(aj => ai*aj))
      Tuple2(a.take(8),(if (a(8) == 1) java.lang.Boolean.TRUE else java.lang.Boolean.FALSE))
    }.toList

    // now center the dataset by computing component wise means
    val scaleInfo = rawdataset.foldLeft(Map[Int,(Float,Float,Int)]()) { (m,d) =>
      d._1.zipWithIndex.foldLeft(m) { (mi,f) =>
        val default = mi.getOrElse(f._2,(0.0f,0.0f,0))
        mi + (f._2 -> (f._1+default._1,f._1*f._1 + default._2, 1 + default._3))
      }
    }

    val rescaleInfo = scaleInfo.map { case (f,(s,ss2,n)) =>
      f -> (s/n,math.sqrt((ss2 - (s*s)/n)/(n-1)))
    }

    val rescaledDataset = rawdataset.map { d =>
      val rescaledVector = d._1.zipWithIndex.map { case (v,f) =>
        ((v - rescaleInfo(f)._1)/rescaleInfo(f)._2).toFloat
      }
      (rescaledVector -> d._2)
    }

    rescaledDataset
  }

  override def getPoint(i: Int): Array[Float] = dataset(i)._1

  override def getLabel(i: Int): java.lang.Boolean = dataset(i)._2

  override def getLabels: util.Set[java.lang.Boolean] = Set(java.lang.Boolean.TRUE,java.lang.Boolean.FALSE)

  override def dim(): Int =dataset(0)._1.size

  override def count(): Int = dataset.size

  override def sample(fraction: Float): Dataset[Array[Float], java.lang.Boolean] = new Dataset[Array[Float],java.lang.Boolean] {
    val sampleDataset = dataset.filter(_ => Random.nextFloat() < fraction)

    override def count(): Int = sampleDataset.size
    override def getPoint(i: Int) : Array[Float] = sampleDataset(i)._1
    override def getLabel(i : Int) : java.lang.Boolean = sampleDataset(i)._2
    override def getLabels : util.Set[java.lang.Boolean] = Set(java.lang.Boolean.TRUE,java.lang.Boolean.FALSE)
    override def dim(): Int = 8

    override def sample(fraction: Float): Dataset[Array[Float], java.lang.Boolean] = ???
  }


  override def trainingError(c:Classifier[Array[Float],java.lang.Boolean]):Int = {
    dataset.map(_ match {
      case (xi,yi) =>
        val (prediction,margin) = c.predictions(xi).toSeq.sortWith(_._2 < _._2)(0)
        if(prediction != yi.booleanValue()) 1 else 0
    }).sum
  }

  def confusionMatrix(c:Classifier[Array[Float],java.lang.Boolean]) = {
    var (tp,fp,tn,fn) = (0,0,0,0)
    dataset.foreach {
      case (xi,yi) =>
        val (prediction, margin) = c.predictions(xi).toSeq.sortWith(_._2 > _._2)(0)
        (prediction.booleanValue(),yi.booleanValue()) match {
          case (true,true) => tp += 1
          case (true,false) =>
            fp += 1
            //println(s"Missed ${xi.map(xii => f"$xii%2.2f").mkString(";")} with margin $margin")
          case (false, false) => tn += 1
          case (false,true) => fn+=1
        }
    }

    (tp,fp,tn,fn)
  }


}
