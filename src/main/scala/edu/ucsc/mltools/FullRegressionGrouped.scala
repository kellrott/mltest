package edu.ucsc.mltools

import org.rogach.scallop
import org.apache.spark.{Partitioner, HashPartitioner, SparkContext, SparkConf}
import org.apache.log4j.{Level, Logger}
import java.util.concurrent.{Callable, Executors}
import org.apache.spark.mllib.util.MLUtils
import java.io.File
import scala.io.Source.fromFile
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import scala.util.parsing.json.JSONObject
import scala.collection.JavaConverters._
import org.apache.spark.mllib.grouped.GroupedSVMWithSGD
import org.apache.spark.mllib.grouped.GroupedLogisticRegressionWithSGD

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import breeze.linalg.DenseVector
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.SparkContext._

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
import org.apache.spark.mllib.optimization.HingeGradient


/**
 * Created by kellrott on 6/29/14.
 */

class NamePartitioner(groupMap:Map[String,Int]) extends Partitioner {
  override def numPartitions: Int = groupMap.size
  override def getPartition(key: Any): Int = {
    groupMap(key.asInstanceOf[String])
  }
  override def equals(other: Any) : Boolean = other.isInstanceOf[NamePartitioner]
}

object FullRegressionGrouped {

  def main(args: Array[String]) = {

    object cmdline extends scallop.ScallopConf(args) {
      val master: scallop.ScallopOption[String] = opt[String]("master", default = Some("local"))
      val cores: scallop.ScallopOption[String] = opt[String]("cores")
      val workdir: scallop.ScallopOption[String] = opt[String]("workdir", default = Some("/tmp"))
      val outdir: scallop.ScallopOption[String] = opt[String]("outdir", default = Some("weights"))
      val symbol: scallop.ScallopOption[String] = opt[String]("symbol")
      val symbolFile: scallop.ScallopOption[String] = opt[String]("symbolfile")
      val groupSize: scallop.ScallopOption[Int] = opt[Int]("groupsize", default = Some(10))
      val taskCount: scallop.ScallopOption[Int] = opt[Int]("taskcount", default = Some(1000))
      val obsFile: scallop.ScallopOption[String] = trailArg[String](required = true)
      val traincycles: scallop.ScallopOption[Int] = opt[Int]("traincycles", default = Some(100))
      val featureFile: scallop.ScallopOption[String] = trailArg[String](required = true)
    }

    var conf = new SparkConf()
      .setMaster(cmdline.master())
      .setAppName("MLTestGrouped")
      .set("spark.executor.memory", "10g")
      .set("spark.local.dir", cmdline.workdir())
      .set("spark.akka.frameSize", "50")
      .set("spark.akka.threads", "10")

    if (cmdline.cores.isDefined) {
      conf = conf.set("spark.mesos.coarse", "true")
        .set("spark.cores.max", cmdline.cores())
    }

    //.set("spark.eventLog.enabled", "true")
    //.set("spark.scheduler.mode", "FAIR")


    val sc = new SparkContext(conf)
    val obs_data = DataFrame.load_csv(sc, cmdline.obsFile(), separator = '\t')(x => math.log(x + 1) + 0.01)
    val pred_data = DataFrame.load_csv(sc, cmdline.featureFile(), separator = '\t')

    //PropertyConfigurator.configure("log4j.config")
    Logger.getLogger("org.apache").setLevel(Level.WARN)

    val name_array = if (cmdline.symbol.isSupplied) {
      Array(cmdline.symbol())
    } else if (cmdline.symbolFile.isSupplied) {
      fromFile(cmdline.symbolFile()).getLines().map( _.stripLineEnd ).toArray
    } else {
      obs_data.index.toArray
    }.filter( x => pred_data.index.contains() )

    name_array.sliding(cmdline.groupSize(), cmdline.groupSize()).foreach( name_set => {
      println("Training %s".format( name_set.mkString(",") ))
      val folds = name_set.filter(x => pred_data.index.contains(x)).map(x => (x, pred_data.labelJoin(obs_data, x)))
        .map(x => (x._1, MLUtils.kFold(x._2.rdd.map(_._2), 10, seed = 11)))

      val groupSize = cmdline.groupSize()

      val training: RDD[(String, LabeledPoint)] = sc.union(
        folds.flatMap(x => x._2.zipWithIndex.map(y => {
          val name = "%s:%d".format(x._1, y._2)
          y._1._1.map(z => (name, z))
        }))
      ).coalesce(cmdline.taskCount())

      //val models = GroupedSVMWithSGD.train[String](training, cmdline.traincycles())
      val models = GroupedLogisticRegressionWithSGD.train[String](training, cmdline.traincycles())
      models.foreach(x => {
        val w = new JSONObject(Map(obs_data.index.zip(x._2.weights.toArray): _*))
        val gene_name = x._1.split(":")(0)
        val obj = Map(("gene", gene_name), ("weights", w), ("intercept", x._2.intercept))
        val outfile = new File(cmdline.outdir(), x._1 + ".weights.vec")
        val f = new java.io.FileWriter(outfile)
        f.write(new JSONObject(obj).toString())
        f.close()
        //println(x._1, x._2)
      })
    })
    sc.stop()
  }


  def generateCrossFold(pred_data: DataFrame, obs_data: DataFrame, gene_name: String, outdir: String): Array[Callable[Unit]] = {
    {
      if (pred_data.index.contains(gene_name)) {
        val gene = pred_data.labelJoin(obs_data, gene_name)
        //sc.setLocalProperty("spark.scheduler.pool", "pool_" + (gene_info._2 % 10))
        val folds = MLUtils.kFold(gene.rdd.map(_._2).coalesce(2), 10, seed = 11).toIndexedSeq //.toParArray
        //folds.tasksupport = task_support
        //val splits = gene.rdd.map(_._2).randomSplit(Array(0.6, 0.4), seed = 11L)
        val results = folds.zipWithIndex.map(x => {
            val outfile = new File(outdir, gene_name + "." + x._2 + ".weights.vec")
            new Callable[Unit] {
              override def call(): Unit = {
                val training = x._1._1
                val testing = x._1._2
                val model = SVMWithSGD.train(training, 200)
                model.clearThreshold()
                // Evaluate model on training examples and compute training error
                val scoreAndLabel = testing.map { x =>
                  val prediction = model.predict(x.features)
                  (prediction, x.label)
                }
                val metrics = new BinaryClassificationMetrics(scoreAndLabel)
                val auROC = metrics.areaUnderROC()

                println("Max Balanced Accuracy", metrics.pr().map(x => (x._1 + x._2) / 2.0).max())
                println("Area under ROC = " + auROC)
                println("Area under PR = " + metrics.areaUnderPR)

                println(metrics.precisionByThreshold().map(_._2).max)
                //val t = metrics.precisionByThreshold().filter(_._2 > 0.1).join(metrics.recallByThreshold()).collect()
                //println(t.length, t.mkString(","))
                //val MSE = scoreAndLabel.map{case(v, p) => math.pow((v - p), 2)}.mean()
                //println("training Mean Squared Error = " + MSE)

                val w =  new JSONObject(Map(obs_data.index.zip(model.weights.toArray): _*))
                val obj = Map(("gene", gene_name), ("auROC", auROC), ("weights", w), ("intercept", model.intercept)) //, ("metrics", metrics.pr().collect.mkString(" ")))
                val f = new java.io.FileWriter(outfile)
                f.write(new JSONObject(obj).toString())
                f.close()
              }
            }
          })
        return results.toArray
      }
      return Array[Callable[Unit]]()
    }

  }
}
