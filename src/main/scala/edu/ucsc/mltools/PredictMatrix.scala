package edu.ucsc.mltools

import org.rogach.scallop
import org.apache.spark.{SparkContext, SparkConf}
import java.io.File
import scala.util.parsing.json.JSON
import scala.io.Source
import scala.io.Source._
import scala.Some
import com.fasterxml.jackson.databind.ObjectMapper
import scala.collection.JavaConverters._
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.{Vectors => MLVectors}

/**
 * Created by kellrott on 6/29/14.
 */
object PredictMatrix {
  def main(args:Array[String]) = {

    object cmdline extends scallop.ScallopConf(args) {
      val master : scallop.ScallopOption[String] = opt[String]("master", default = Some("local"))
      val cores : scallop.ScallopOption[String] = opt[String]("cores", default = Some("32"))
      val workdir : scallop.ScallopOption[String] = opt[String]("workdir", default = Some("/tmp"))

      val modeldir: scallop.ScallopOption[String] = trailArg[String](required = true)
      val obsfile: scallop.ScallopOption[String] = trailArg[String](required = true)
      val outFile: scallop.ScallopOption[String] = trailArg[String](required = true)
    }

    val conf = new SparkConf()
      .setMaster(cmdline.master())
      .setAppName("MLScan")
      .set("spark.executor.memory", "8g")
      .set("spark.mesos.coarse", "true")
      .set("spark.cores.max", cmdline.cores())
      .set("spark.local.dir", cmdline.workdir())
      .set("spark.scheduler.mode", "FAIR")

    val sc = new SparkContext(conf)

    val fileset = sc.parallelize(new File(cmdline.modeldir()).listFiles().map( _.getAbsolutePath ), 50)
    val model_objects = fileset.map( x => {
      try {
        val jsontxt = fromFile(x).mkString
        new ObjectMapper().readValue(jsontxt, classOf[java.util.Map[String, AnyRef]])
      } catch {
        case e: Exception => (Map[String,AnyRef]().asJava)
      }
    } ).filter( _.containsKey("gene"))

    val obs_data = DataFrame.load_csv(sc, cmdline.obsfile(), separator = '\t')
    val index_br = obs_data.index_br

    val models = model_objects.map( x => {
      val weight_map = x.get("weights").asInstanceOf[java.util.Map[String,Double]].asScala
      if (weight_map.keys.filter(index_br.value.contains(_)).size == 0) {
        throw new RuntimeException("No symbol overlap in model")
      }
      val weights = index_br.value.map( y => weight_map.getOrElse(y, 0.0) ).toArray
      val model = new LogisticRegressionModel(MLVectors.dense(weights), x.get("intercept").asInstanceOf[Double])
      model.clearThreshold()
      if (x.containsKey("foldNumber")) {
        ("%s_%s".format(x.get("gene").asInstanceOf[String], x.get("foldNumber")), model)
      } else {
        (x.get("gene").asInstanceOf[String], model)
      }
    } ).filter(_._1 != null).collect()

    val models_br = sc.broadcast(models)

    val predictions = obs_data.rdd.map( x => {
      (x._1, models_br.value.map( y => (y._1, y._2.predict(x._2) ) ).toMap)
    })

    val out = DataFrame.create(predictions)
    out.write_csv(cmdline.outFile(), '\t')

  }
}
