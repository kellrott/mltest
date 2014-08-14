package edu.ucsc.mltools

import org.rogach.scallop
import org.apache.spark.{SparkContext, SparkConf}
import scala.util.parsing.json.{JSONObject, JSON}
import java.io.File
import org.apache.spark.SparkContext._
import scala.io.Source.fromFile
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.{Vectors => MLVectors}
import com.fasterxml.jackson.databind.ObjectMapper
import scala.collection.JavaConverters._

object ModelAnalysis {

  def main(args:Array[String]) = {
    object cmdline extends scallop.ScallopConf(args) {
      val master: scallop.ScallopOption[String] = opt[String]("master", default = Some("local"))
      val cores: scallop.ScallopOption[String] = opt[String]("cores")
      val workdir: scallop.ScallopOption[String] = opt[String]("workdir", default = Some("/tmp"))
      val modeldir: scallop.ScallopOption[String] = opt[String]("modeldir", default = Some("weights"))
      val symbol: scallop.ScallopOption[String] = opt[String]("symbol")
      val symbolFile: scallop.ScallopOption[String] = opt[String]("symbolfile")
      val groupSize: scallop.ScallopOption[Int] = opt[Int]("groupsize", default = Some(10))
      val taskCount: scallop.ScallopOption[Int] = opt[Int]("taskcount", default = Some(1000))
      val traincycles: scallop.ScallopOption[Int] = opt[Int]("traincycles", default = Some(100))
      val obsFile: scallop.ScallopOption[String] = trailArg[String](required = true)
      val labelFile: scallop.ScallopOption[String] = trailArg[String](required = true)
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

    val sc = new SparkContext(conf)
    val obs_data = DataFrame.load_csv(sc, cmdline.obsFile(), separator = '\t')
    val label_data = DataFrame.load_csv(sc, cmdline.labelFile(), separator = '\t')

    val index_br = obs_data.index_br

    val fileset = sc.parallelize(new File(cmdline.modeldir()).listFiles().map( _.getAbsolutePath ), 50)
    val model_objects = fileset.map( x => {
      try {
        val jsontxt = fromFile(x).mkString
        new ObjectMapper().readValue(jsontxt, classOf[java.util.Map[String, AnyRef]])
      } catch {
        case e: Exception => (Map[String,AnyRef]().asJava)
      }
    } ).filter( _.containsKey("gene"))

    /*
    val stats = model_objects.map( x => {
      (x("gene").asInstanceOf[String], (1, x("stats").asInstanceOf[Map[String,Double]]("auPR")))
    } ).reduceByKey( (x,y) => {
      (x._1+y._1, x._2+y._2)
    }).map( x => {
      (x._1, x._2._2 / x._2._1)
    })
    stats.collect.sortBy( x => x._2 ).foreach( x => println("%s\t%s".format(x._1, x._2)))
    */

    val models = model_objects.map( x => {
      val weight_map = x.get("weights").asInstanceOf[java.util.Map[String,Double]].asScala
      val weights = index_br.value.map( y => weight_map.getOrElse(y, 0.0) ).toArray
      val model = new LogisticRegressionModel(MLVectors.dense(weights), x.get("intercept").asInstanceOf[Double])
      model.clearThreshold()
      ("%s_%s".format(x.get("gene").asInstanceOf[String], x.get("foldNumber")), model)
    } ).filter(_._1 != null).collect()

    println(models.toIterator.next()._2.weights.toArray.mkString(","))

    /*
    val predictions = models.cartesian( obs_data.rdd ).map( x => {
      (x._1._1, Map( (x._2._1,x._1._2.predict(x._2._2)) ))
    } ).reduceByKey( _ ++ _ )
    val predictions_frame = DataFrame.create(predictions)
    predictions_frame.write_csv("feature_matrix.tsv", seperator='\t')
    */

    val models_br = sc.broadcast(models)
    val predictions = obs_data.rdd.map( x => {
      val m = models_br.value.map( y => {
        (y._1, y._2.predict(x._2))
      } ).toMap
      (x._1, m)
    })
    val pred_matrix = DataFrame.create( predictions )

    pred_matrix.write_csv("feature_matrix.tsv", seperator='\t')


  }
}
