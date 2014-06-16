package edu.ucsc.mltest

import breeze.{linalg => br_linalg, stats}
import breeze.stats.DescriptiveStats._
import breeze.linalg.support._
import breeze.stats.meanAndVariance

object MathUtils {

  def corr(a: br_linalg.Vector[Double], b: br_linalg.Vector[Double]): Double = {
    if (a.length != b.length)
      return Double.NaN
    val n = a.length
    val (amean, avar) = meanAndVariance(a)
    val (bmean, bvar) = meanAndVariance(b)
    val astddev = math.sqrt(avar)
    val bstddev = math.sqrt(bvar)
    1.0 / (n - 1.0) * br_linalg.sum( ((a - amean) / astddev) :* ((b - bmean) / bstddev) )
  }


  def euclidean(a: br_linalg.Vector[Double], b: br_linalg.Vector[Double]): Double = {
    return a.toArray.zip(b.toArray).map( x => math.pow(x._1 - x._2, 2) ).reduce(_+_)
  }

  def meanAndVariance(a:br_linalg.Vector[Double]) : (Double,Double) = {
    var n :Double = 0.0
    var sum :Double = 0
    var sum_sqr :Double = 0
    a.foreach( x => {
      n+=1
      sum += x
      sum_sqr += x*x
    })
    return (sum/n,  (sum_sqr - (sum*sum)/n)/(n - 1))
  }
}