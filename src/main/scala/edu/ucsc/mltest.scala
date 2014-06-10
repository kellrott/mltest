
package edu.ucsc.mltest


import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.log4j.PropertyConfigurator

import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg

import breeze.io.CSVReader
import java.io.{File, FileReader}
import org.apache.spark.storage.StorageLevel
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.rdd._

import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.{SquaredL2Updater, L1Updater}
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import scala.util.parsing.json.JSONObject

class Segmenter[T] (in: Iterator[T], blockSize:Int) extends Iterator[Seq[T]] {

  var block : Seq[T] = null

  def queue_block() {
    if (block == null) {
      var i = 0
      val o = new ArrayBuffer[T]()
      while (i < blockSize && in.hasNext) {
        o += in.next()
        i += 1
      }
      block = o.toSeq
    }
  }
  override def hasNext: Boolean = {
    queue_block()
    return block != null
  }

  override def next(): Seq[T] = {
    queue_block()
    if (block == null) {
      return null
    }
    val o = block
    block = null
    return o
  }

}

class ConfusionMatrixRDD()

class BinaryConfusionMatrix(val tp:Long, val tn:Long, val fp:Long, val fn : Long ) {

}


object Test {

  def main(args:Array[String]) = {
    val sc = new SparkContext(args(0), "MLTest")
    val obs_data = DataFrame.load_csv(sc, args(1), separator = '\t')(x => math.log(x+1) + 0.01)
    val pred_data = DataFrame.load_csv(sc, args(2), separator = '\t')

    PropertyConfigurator.configure("log4j.config")

    obs_data.index.slice(0,10).foreach( gene_name => {
      val gene = pred_data.labelJoin(obs_data, "TP53")
      val numIterations = 200
      val lin = new LogisticRegressionWithSGD()
      lin.optimizer.setNumIterations(numIterations).setRegParam(0.8).setUpdater(new L1Updater)
      val model = lin.run(gene.rdd.map(_._2))
      //gene.rdd.saveAsTextFile("gene.vec")
      //val model = LogisticRegressionWithSGD.train(gene.rdd.map(_._2), numIterations)

      // Evaluate model on training examples and compute training error

      val scoreAndLabel = gene.rdd.map { x =>
        val prediction = model.predict(x._2.features)
        (prediction, x._2.label)
      }

      val w = Map(obs_data.index.zip(model.weights.toArray):_*)
      //val MSE = scoreAndLabel.map{case(v, p) => math.pow((v - p), 2)}.mean()
      //println("training Mean Squared Error = " + MSE)
      val metrics = new BinaryClassificationMetrics(scoreAndLabel)
      val auROC = metrics.areaUnderROC()
      println(metrics.pr().collect.mkString(" "))
      println("Area under ROC = " + auROC)

      val obj = Map(("auROC", auROC), ("weights", new JSONObject(w)), ("intercept", model.intercept), ("metrics", metrics.pr().collect.mkString(" ")))

      val f = new java.io.FileWriter("weights/" + gene_name + ".weights.vec")
      f.write(new JSONObject(obj).toString())
      f.close()
    })
    sc.stop()

  }

}