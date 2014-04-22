package com.vishnuvyas.classifiers.sparse

import org.scalatest.{ShouldMatchers, FlatSpec}

/**
 * Created by vishnu on 4/20/14.
 */
class SparseVectorSpec extends FlatSpec with ShouldMatchers {

  val sv = new SparseVector(3)
  sv.set(2,4)
  sv.set(1,0)
  sv.set(0,6)

  private [this] def dot(d1:Array[Double],d2:Array[Double]) = d1.zip(d2).map({case (xi,xj) => xi*xj}).sum

  "A sparse vector " should "keep the cardinalty of the sparse vector unchanged after optimize" in {
    sv.optimize() should be (sv.getCardinality)
  }

  it should "also sort the indices array after an optimize" in {
    sv.optimize()
    sv.getIndices.take(sv.getCardinality).sliding(2).forall({case Array(x,y) => x <= y || y < 0}) should be(true)
  }

  it should "not have indicies corresponding to zero values" in {
    sv.getIndices.take(sv.getCardinality).find(_ == 1) should be (None)
  }

  it should "have the same dot-product as a dense-vector dot-product" in {
    val d1 = Array(6.0d,0,4)
    val d2 = Array(8.0d,6,-7)

    dot(d1,d2) should be (sv.dot(d2))
  }

  it should "still have a proper get method after an optimize call" in {
    sv.optimize()
    sv.get(2) should be (4)
    sv.get(1) should be (0)
  }

  it should "throw an array out of bounds exceptions when we get elements greater than the dimensions" in {
      an [ArrayIndexOutOfBoundsException] should be thrownBy(sv.get(4))
  }

}
