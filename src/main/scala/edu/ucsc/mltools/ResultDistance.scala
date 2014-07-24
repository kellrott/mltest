package edu.ucsc.mltools

import org.rogach.scallop
import org.apache.spark.{SparkContext, SparkConf}
import java.io.File
import scala.util.parsing.json.JSON
import scala.io.Source

/**
 * Created by kellrott on 6/29/14.
 */
object ResultDistance {
  def main(args:Array[String]) = {

    object cmdline extends scallop.ScallopConf(args) {
      val master : scallop.ScallopOption[String] = opt[String]("master", default = Some("local"))
      val cores : scallop.ScallopOption[String] = opt[String]("cores", default = Some("32"))
      val workdir : scallop.ScallopOption[String] = opt[String]("workdir", default = Some("/tmp"))
      val outdir : scallop.ScallopOption[String] = opt[String]("outdir", default = Some("weights"))

      val vectorDir: scallop.ScallopOption[String] = trailArg[String](required = true)
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
    val files_rdd = sc.parallelize(files)
    val data = files_rdd.map( x => JSON.parseFull(Source.fromFile(x).getLines().mkString("")).get.asInstanceOf[Map[String,Any]] )
    val df = DataFrame.create(data.map(x => (x("gene").asInstanceOf[String], x("weights").asInstanceOf[Map[String,Double]])))

    val cor_df = df.pdist("euclidean")

    println(cor_df.index)
    println(cor_df.rdd.first())

    cor_df.write_csv(cmdline.outFile(), '\t')
  }
}
