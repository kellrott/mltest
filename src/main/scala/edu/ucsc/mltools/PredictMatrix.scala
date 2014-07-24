package edu.ucsc.mltools

import org.rogach.scallop
import org.apache.spark.{SparkContext, SparkConf}
import java.io.File
import scala.util.parsing.json.JSON
import scala.io.Source

/**
 * Created by kellrott on 6/29/14.
 */
object PredictMatrix {
  def main(args:Array[String]) = {

    object cmdline extends scallop.ScallopConf(args) {
      val master : scallop.ScallopOption[String] = opt[String]("master", default = Some("local"))
      val cores : scallop.ScallopOption[String] = opt[String]("cores", default = Some("32"))
      val workdir : scallop.ScallopOption[String] = opt[String]("workdir", default = Some("/tmp"))

      val vectorDir: scallop.ScallopOption[String] = trailArg[String](required = true)
      val matrixFile: scallop.ScallopOption[String] = trailArg[String](required = true)
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

    val files = new File(cmdline.vectorDir()).listFiles().filter( f => """.vec""".r.findFirstIn(f.getName).isDefined)
    val files_rdd = sc.parallelize(files.slice(0,5))
    val data = files_rdd.map( x => JSON.parseFull(Source.fromFile(x).getLines().mkString("")).get.asInstanceOf[Map[String,Any]] )
    val mf = LinearModelFrame.create(data.map(x => {
      val label = x("gene").asInstanceOf[String]
      val intercept = x("intercept").asInstanceOf[Double]
      val weights = x("weights").asInstanceOf[Map[String,Double]]
      (label, intercept, weights)
    }))

    val features = DataFrame.load_csv(sc, cmdline.matrixFile(), '\t')(x => math.log(x+1) + 0.01)

    val out = mf.predict(features)
    out.write_csv(cmdline.outFile(), '\t')

  }
}
