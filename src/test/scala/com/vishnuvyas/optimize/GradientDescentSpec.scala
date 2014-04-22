package com.vishnuvyas.optimize

import org.scalatest.{ShouldMatchers, FlatSpec}

/**
 * Created by vishnu on 4/19/14.
 */
class GradientDescentSpec extends FlatSpec with ShouldMatchers {

  val tolerance:Double = 1e-5

  "A Gradient Descent optimizer" should "optimize a simple quadratic function" in {
    val quadraticObjective = new ObjectiveFunction {

      override def eval(x: Array[Double]) = x.map(xi => math.pow(xi,2)).sum

      override def prox(g: Array[Double]): Array[Double] = g

      override def gradient(x: Array[Double]): Array[Double] = x.map(xi => 2*xi)
    }

    val optimizer = new GradientDescentOptimizer

    val w = (1 to 1000).foldLeft(Array(1,10d,100d,1000d,10000d)) { (w,t) =>
      optimizer.descentStep(quadraticObjective,w,0.5,t)
    }

    // see if the value of the quadratic is near zero.
    quadraticObjective.eval(w) should equal (0d +- tolerance)
  }

}
