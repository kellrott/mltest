package edu.ucsc.resultscan

import java.io.{FileReader, File}
import scala.util.parsing.json.{JSON, Parser}
import scala.io.Source
import org.apache.spark.SparkContext

object Scan {
  def main(args:Array[String]) = {
    val sc = new SparkContext(args(0), "MLTest")

    val files = new File(args(1)).listFiles().filter( f => """.vec""".r.findFirstIn(f.getName).isDefined)
    val files_rdd = sc.parallelize(files)
    val data = files_rdd.map( x => JSON.parseFull(Source.fromFile(x).getLines().mkString("")).get.asInstanceOf[Map[String,Any]] )
    data.map(x => x("weights").asInstanceOf[Map[String,Double]]).take(10).foreach(println)
    //data.foreach( x => println(x.keys.mkString(",")))
  }
}