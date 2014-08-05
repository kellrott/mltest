package edu.ucsc.mltools

import org.rogach.scallop
import org.apache.log4j.{Level, Logger}

import java.io.File
import scala.io.Source.fromFile
import scala.util.parsing.json.JSONObject
import scala.collection.JavaConverters._

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.{Partitioner, HashPartitioner, SparkContext, SparkConf}

import org.apache.spark.mllib.grouped.GroupedSVMWithSGD
import org.apache.spark.mllib.grouped.GroupedLogisticRegressionWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.{L1Updater}

/*
class NamePartitioner(groupMap:Map[String,Int]) extends Partitioner {
  override def numPartitions: Int = groupMap.size
  override def getPartition(key: Any): Int = {
    groupMap(key.asInstanceOf[String])
  }
  override def equals(other: Any) : Boolean = other.isInstanceOf[NamePartitioner]
}
*/


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
//      .set("spark.executor.memory", "14g")
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
    val obs_data = DataFrame.load_csv(sc, cmdline.obsFile(), separator = '\t')
    val pred_data = DataFrame.load_csv(sc, cmdline.featureFile(), separator = '\t')

    obs_data.rdd.cache()
    pred_data.rdd.cache()

    //PropertyConfigurator.configure("log4j.config")
    Logger.getLogger("org.apache").setLevel(Level.WARN)

    val name_array = if (cmdline.symbol.isSupplied) {
      Array(cmdline.symbol())
    } else if (cmdline.symbolFile.isSupplied) {
      fromFile(cmdline.symbolFile()).getLines().map( _.stripLineEnd ).toArray
    } else {
      obs_data.index.toArray
    }.filter( x => pred_data.index.contains(x) )

    name_array.sliding(cmdline.groupSize(), cmdline.groupSize()).foreach( name_set => {
      println("Training %s".format( name_set.mkString(",") ))
      val training_data = name_set.filter(x => pred_data.index.contains(x)).map(x => (x, pred_data.labelJoin(obs_data, x)))
      val folds = training_data.map(x => (x._1, MLUtils.kFold(x._2.rdd.map(_._2), 10, seed = 11)))

      val groupSize = cmdline.groupSize()

      val training: RDD[(String, LabeledPoint)] = sc.union(
        folds.flatMap(x => x._2.zipWithIndex.map(y => {
          val name = "%s_%d".format(x._1, y._2)
          y._1._1.map(z => (name, z))
        }))
      ).coalesce(cmdline.taskCount()).partitionBy( new HashPartitioner(groupSize * 10 * 2))

      /*
      println("Fold Count: %d".format(folds.size))
      println("Training keys: %s".format( (training.keys.distinct().collect()).mkString(",")))
      println("Training %s samples".format(training.count()))
      */

      //val models = GroupedSVMWithSGD.train[String](training, cmdline.traincycles())
      val trainer = new GroupedLogisticRegressionWithSGD[String]().setIntercept(true)
      trainer.optimizer.setNumIterations(cmdline.traincycles())
      trainer.optimizer.setUpdater(new L1Updater)
      trainer.optimizer.setRegParam(0.1)
      val models = trainer.run(training)
      println("Models Complete: %d".format(models.size))

      //test the model
      val fold_map = folds.toMap

      models.foreach(x => {
        println("Testing: %s".format(x._1))
        val gene_name = x._1.split("_")(0)
        val fold_num = x._1.split(("_"))(1).toInt
        val model = sc.broadcast(x._2)
        x._2.clearThreshold()
        val testing = fold_map(gene_name)(fold_num)._2
        // Evaluate model on training examples and compute training error
        val scoreAndLabel = testing.map { y =>
          val prediction = model.value.predict(y.features)
          (prediction, y.label)
        }
        model.unpersist()
        val metrics = new BinaryClassificationMetrics(scoreAndLabel)
        val auROC = metrics.areaUnderROC()
        val auPR = metrics.areaUnderPR()
        val stats = new JSONObject(Map( ("auROC", auROC), ("auPR", auPR)))

        val w = new JSONObject(Map(obs_data.index.zip(x._2.weights.toArray): _*))

        val obj = Map(
          ("gene", gene_name),
          ("foldNumber", fold_num),
          ("weights", w),
          ("method", "logistic"),
          ("intercept", x._2.intercept),
          ("stats", stats)
        )
        val outfile = new File(cmdline.outdir(), x._1 + ".weights.vec")
        println("Outputting %s".format(outfile))
        val f = new java.io.FileWriter(outfile)
        f.write(new JSONObject(obj).toString())
        f.close()
      })
    })
    sc.stop()
  }

}
