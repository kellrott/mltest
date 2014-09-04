package edu.ucsc.mltools

import breeze.io.CSVReader
import java.io.{FileWriter, File, FileReader}
import org.apache.spark.mllib.{linalg => ml_linalg}
import breeze.{linalg => br_linalg}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
//import edu.ucsc.mltools.MathUtils


object DataFrame {

  def load_csv(sc: SparkContext, path: String, separator : Char =  ',', minPartitions : Int=2) : DataFrame = {
    val obs_data = CSVReader.iterator(new FileReader(new File(path)), separator = separator)
    val header = obs_data.next().seq

    val header_br = sc.broadcast(header)
    val obs_rdd = sc.textFile(path, minPartitions).flatMap( x => CSVReader.parse(x, separator = separator) )
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
    val other_rdd = df.rdd
    val out = rdd.join( other_rdd ).map( x => (x._1, LabeledPoint(x._2._1.toArray(i), x._2._2) ) )
    new LabeledDataFrame(sc, column, index, out)
  }

  /*
  def labelJoin(df:DataFrame, columns:Array[String]) : LabeledDataFrame = {
    val i = columns.map( x => index.indexOf(x) )
    val other_rdd = df.rdd
    val out = rdd.join( other_rdd ).flatMap( x => {
      i.map( y => (x._1, LabeledPoint(x._2._1.toArray(y), x._2._2) ) )
    } )
    new LabeledDataFrame(sc, index, out)
  }
  */

  def reindex(newIndex : IndexedSeq[String], default:Double=0.0) : DataFrame = {
    val remap = newIndex.map( x => index.indexOf(x) )
    val remap_br = sc.broadcast(remap)
    val new_rdd = rdd.map( x => {
      val a = x._2.toArray
      val new_a = remap.map( y => {
        if (y == -1)
          default
        else
          a(y)
      }).toArray
      (x._1, ml_linalg.Vectors.dense(new_a))
    })
    new DataFrame(sc, newIndex, new_rdd)
  }

  def apply(f: Double => Double) : DataFrame = {
    return new DataFrame(sc, index, rdd.map( x => (x._1, ml_linalg.Vectors.dense(x._2.toArray.map(f))) ) )
  }

  def pdist(method:String) : DataFrame = {
    return method match {
      case "corr"  => pdist( (x,y) => MathUtils.corr(x,y))
      case "euclidean" => pdist( (x,y) => MathUtils.euclidean(x,y))
      case _ => throw new Exception("Bad distance method")
    }
  }

  def pdist( method : (br_linalg.Vector[Double], br_linalg.Vector[Double]) => Double) : DataFrame = {
    val n = rdd.cartesian(rdd).map( x => {
      val y = br_linalg.DenseVector(x._1._2.toArray)
      val z = br_linalg.DenseVector(x._2._2.toArray)
      val o = method(y,z)
      (x._1._1, Map((x._2._1, o)))
    }).reduceByKey( (x,y) => {
      x ++ y
    } )
    return DataFrame.create(n)
  }


  def write_csv(path:String, seperator : Char = ',') = {
    val out = new FileWriter(path)
    val header = Array("") ++ index
    out.write( header.mkString(seperator.toString) )
    out.write("\n")
    rdd.toLocalIterator.foreach(  x => {
      out.write(x._1)
      out.write(seperator.toString)
      out.write(x._2.toArray.mkString(seperator.toString))
      out.write("\n")
    })
  }

  def coalesce(numPartitions:Int = 1) : DataFrame = {
    return new DataFrame(sc, index, rdd.coalesce(numPartitions))
  }

}
