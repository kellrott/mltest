
package edu.ucsc.mltest


import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
import org.apache.log4j.PropertyConfigurator

import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg

import breeze.io.CSVReader
import java.io.{File, FileReader}
import org.apache.spark.storage.StorageLevel
import org.rogach.scallop
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.rdd._

import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.{SquaredL2Updater, L1Updater}
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import scala.collection.parallel.ForkJoinTaskSupport
import scala.util.parsing.json.JSONObject

import org.apache.log4j.Logger
import org.apache.log4j.Level


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


object FullRegression {

  def main(args:Array[String]) = {

    object cmdline extends scallop.ScallopConf(args) {
      val master : scallop.ScallopOption[String] = opt[String]("master", default = Some("local"))
      val cores : scallop.ScallopOption[String] = opt[String]("cores", default = Some("32"))
      val workdir : scallop.ScallopOption[String] = opt[String]("workdir", default = Some("/tmp"))
      val outdir : scallop.ScallopOption[String] = opt[String]("outdir", default = Some("weights"))
      val tasks : scallop.ScallopOption[Int] = opt[Int]("tasks", default = Some(10))

      val obsFile: scallop.ScallopOption[String] = trailArg[String](required = true)
      val featureFile: scallop.ScallopOption[String] = trailArg[String](required = true)
    }

    println(cmdline)

    val conf = new SparkConf()
      .setMaster(cmdline.master())
      .setAppName("MLTest")
      .set("spark.executor.memory", "8g")
      .set("spark.mesos.coarse", "true")
      .set("spark.cores.max", cmdline.cores())
      .set("spark.local.dir", cmdline.workdir())
      .set("spark.scheduler.mode", "FAIR")


    val sc = new SparkContext(conf)
    val obs_data = DataFrame.load_csv(sc, cmdline.obsFile(), separator = '\t')(x => math.log(x+1) + 0.01)
    val pred_data = DataFrame.load_csv(sc, cmdline.featureFile(), separator = '\t')

    //PropertyConfigurator.configure("log4j.config")
    Logger.getLogger("org.apache").setLevel(Level.WARN)

    val name_array = obs_data.index.toParArray
    name_array.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(cmdline.tasks()))

    name_array.map( gene_name => {
      if (pred_data.index.contains(gene_name)) {
        val gene = pred_data.labelJoin(obs_data, gene_name)
        val numIterations = 200
        val lin = new LogisticRegressionWithSGD()
        lin.optimizer.setNumIterations(numIterations).setRegParam(0.8).setUpdater(new L1Updater)
        val model = lin.run(gene.rdd.map(_._2))
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

        val obj = Map(("gene", gene_name), ("auROC", auROC), ("weights", new JSONObject(w)), ("intercept", model.intercept), ("metrics", metrics.pr().collect.mkString(" ")))

        val f = new java.io.FileWriter(new File(cmdline.outdir(), gene_name + ".weights.vec"))
        f.write(new JSONObject(obj).toString())
        f.close()
      }
    })
    sc.stop()
  }

}