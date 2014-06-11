package edu.ucsc.mltest

import breeze.io.CSVReader
import java.io.{File, FileReader}
import org.apache.spark.mllib.{linalg => ml_linalg}
import breeze.{linalg => br_linalg}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._


object DataFrame {

  def load_csv(sc: SparkContext, path: String, separator : Char =  ',') : DataFrame = {
    val obs_data = CSVReader.iterator(new FileReader(new File(path)), separator = separator)
    val header = obs_data.next().seq

    val header_br = sc.broadcast(header)
    val obs_rdd = sc.textFile(path).flatMap( x => CSVReader.parse(x, separator = separator) )
    val obs_rdd_values = obs_rdd.filter( x=> x.zip(header_br.value).filter( y => y._1==y._2 ).size < header_br.value.size ).
      map( x =>
      (x(0), ml_linalg.Vectors.dense(x.slice(1, x.size).map(y =>y match { case "" => 0.0; case _ => y.toDouble }).toArray))
      )

    val out = new DataFrame(sc, header.slice(1,header.length), obs_rdd_values)
    return out
  }

  def create(data:RDD[(String,Map[String,Double])]) : DataFrame = {
    val s = data.map( x => x._2.keySet ).reduce( _ ++ _).toIndexedSeq
    val index_br = data.context.broadcast(s)
    val vec_data = data.map( x => {
      (x._1, ml_linalg.Vectors.dense(index_br.value.map( y => x._2(y)).toArray))
    })
    return new DataFrame(data.context, s, vec_data)
  }

}


class DataFrame(val sc : SparkContext, val index : IndexedSeq[String], val rdd:RDD[(String,ml_linalg.Vector)]) {

  val index_br = sc.broadcast(index)

  /*
  def transpose() : DataFrame = {
    val rdd_values = rdd.
      flatMap( x=> x._2.toArray.zip(index_br.value).map( y => (x._1, Map((y._2, y._1.toDouble))) ) ).
      reduceByKey( (x,y) => x ++ y)

  }
  */

  def slice(rowSlice : (String) => Boolean, colSlice : (String) => Boolean) : DataFrame = {
    val cols_filter = index.zipWithIndex.filter( x => colSlice(x._1) )
    val cols_pos = cols_filter.map(_._2)
    val cols_index = cols_filter.map(_._1)
    val out = rdd.filter( x => rowSlice(x._1) ).map( x => (x._1, ml_linalg.Vectors.dense(x._2.toArray.zipWithIndex.filter( x=> cols_pos.contains(x._2) ).map( x => x._1) )) )
    return new DataFrame(sc, cols_index, out)
  }

  def labelJoin(df:DataFrame, column:String) : LabeledDataFrame = {
    val i = index.indexOf(column)
    val out = rdd.join( df.rdd ).map( x => (x._1, LabeledPoint(x._2._1.toArray(i), x._2._2) ) )
    new LabeledDataFrame(sc, index, out)
  }

  def apply(f: Double => Double) : DataFrame = {
    return new DataFrame(sc, index, rdd.map( x => (x._1, ml_linalg.Vectors.dense(x._2.toArray.map(f))) ) )
  }

  def corr() : DataFrame = {
    rdd.cartesian(rdd).map( x => {

      val y = br_linalg.DenseVector(x._1._2)
      val z = br_linalg.DenseVector(x._2._2)

      MathUtils.corr(y,z)


    }
    )
    null
  }

}
