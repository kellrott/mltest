package edu.ucsc.mltools

import org.rogach.scallop
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.log4j.{Level, Logger}
import java.util.concurrent.{Callable, Executors}
import org.apache.spark.mllib.util.MLUtils
import java.io.File
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import scala.util.parsing.json.JSONObject
import scala.collection.JavaConverters._

/**
 * Created by kellrott on 6/29/14.
 */
object FullRegression {

  def main(args: Array[String]) = {

    object cmdline extends scallop.ScallopConf(args) {
      val master: scallop.ScallopOption[String] = opt[String]("master", default = Some("local"))
      val cores: scallop.ScallopOption[String] = opt[String]("cores", default = Some("32"))
      val workdir: scallop.ScallopOption[String] = opt[String]("workdir", default = Some("/tmp"))
      val outdir: scallop.ScallopOption[String] = opt[String]("outdir", default = Some("weights"))
      val tasks: scallop.ScallopOption[Int] = opt[Int]("tasks", default = Some(10))
      val symbol : scallop.ScallopOption[String] = opt[String]("symbol")
      val obsFile: scallop.ScallopOption[String] = trailArg[String](required = true)
      val featureFile: scallop.ScallopOption[String] = trailArg[String](required = true)
    }

    val conf = new SparkConf()
      .setMaster(cmdline.master())
      .setAppName("MLTest")
      .set("spark.executor.memory", "8g")
      .set("spark.mesos.coarse", "true")
      //.set("spark.akka.threads", "16")
      .set("spark.cores.max", cmdline.cores())
      .set("spark.local.dir", cmdline.workdir())
      //.set("spark.scheduler.mode", "FAIR")


    val sc = new SparkContext(conf)
    val obs_data = DataFrame.load_csv(sc, cmdline.obsFile(), separator = '\t')(x => math.log(x + 1) + 0.01)
    val pred_data = DataFrame.load_csv(sc, cmdline.featureFile(), separator = '\t')

    //PropertyConfigurator.configure("log4j.config")
    Logger.getLogger("org.apache").setLevel(Level.WARN)

    //val name_array = Array("A1CF", "A4GALT", "A2M").toIndexedSeq.toParArray
    val name_array = if (cmdline.symbol.isSupplied) {
      Array(cmdline.symbol())
    } else {
      obs_data.index.toArray
    }

    val work_array = name_array.flatMap(gene_name => generateCrossFold(pred_data, obs_data, gene_name, cmdline.outdir()))

    val threadPool = Executors.newFixedThreadPool(cmdline.tasks())
    threadPool.invokeAll(work_array.toList.asJavaCollection)
    threadPool.shutdown()
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
