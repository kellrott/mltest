package edu.ucsc.resultscan

import java.io.{FileReader, File}
import scala.util.parsing.json.{JSON, Parser}
import scala.io.Source
import org.apache.spark.SparkContext
import edu.ucsc.mltest.DataFrame

object Scan {
  def main(args:Array[String]) = {
    val sc = new SparkContext(args(0), "MLTest")

    val files = new File(args(1)).listFiles().filter( f => """.vec""".r.findFirstIn(f.getName).isDefined).slice(0,50)
    val files_rdd = sc.parallelize(files)
    val data = files_rdd.map( x => JSON.parseFull(Source.fromFile(x).getLines().mkString("")).get.asInstanceOf[Map[String,Any]] )
    val df = DataFrame.create(data.map(x => (x("gene").asInstanceOf[String], x("weights").asInstanceOf[Map[String,Double]])))

    val cor_df = df.pdist("euclidean")


    println(cor_df.index)
    println(cor_df.rdd.first())

    cor_df.write_csv(args(0) + ".matrix")

    //data.foreach( x => println(x.keys.mkString(",")))
  }
}