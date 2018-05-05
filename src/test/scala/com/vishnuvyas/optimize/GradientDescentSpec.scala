package com.vishnuvyas.optimize

import org.scalatest.{ShouldMatchers, FlatSpec}

/**
 * Created by vishnu on 4/19/14.
 */
class GradientDescentSpec extends FlatSpec with ShouldMatchers {

  val tolerance:Float = 1e-5f

  "A Gradient Descent optimizer" should "optimize a simple quadratic function" in {
    val quadraticObjective = new ObjectiveFunction {

      override def eval(x: Array[Float]) = x.map(xi => math.pow(xi,2)).sum.toFloat

      override def prox(g: Array[Float]): Array[Float] = g

      override def gradient(x: Array[Float]): Array[Float] = x.map(xi => 2*xi)
    }

    val optimizer = new GradientDescentOptimizer

    val w = (1 to 1000).foldLeft(Array(1,10f,100f,1000f,10000f)) { (w,t) =>
      optimizer.descentStep(quadraticObjective,w,0.5f,t)
    }

    // see if the value of the quadratic is near zero.
    quadraticObjective.eval(w) should equal (0f +- tolerance)
  }

}
