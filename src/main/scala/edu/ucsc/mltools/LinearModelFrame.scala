package edu.ucsc.mltools

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.{linalg => ml_linalg}
import org.apache.spark.mllib.regression.GeneralizedLinearModel
import org.apache.spark.mllib.linalg.Vector
import breeze.{linalg => br_linalg}




object LinearModelFrame {
  def create(data:RDD[(String,Double,Map[String,Double])]) : LinearModelFrame = {
    val s = data.map( x => x._3.keySet ).reduce( _ ++ _).toIndexedSeq
    val index_br = data.context.broadcast(s)
    val vec_data = data.map( x => {
      val model = new MyLogisticRegressionModel(ml_linalg.Vectors.dense(index_br.value.map( y => x._3(y)).toArray), x._2)
      (x._1, model.asInstanceOf[GeneralizedLinearModel])
    })
    return new LinearModelFrame(data.context, s, vec_data)
  }
}


class LinearModelFrame(val sc : SparkContext,
                       val index : IndexedSeq[String],
                       val rdd:RDD[(String,GeneralizedLinearModel)]) {

  def predict(data:DataFrame) : DataFrame = {
    //Build a map to change the vector order of the data into the vector order of the model
    val remap = index.map( x => if (data.index.contains(x)) data.index.indexOf(x) else -1 )
    val remap_br = sc.broadcast(remap)

    //Remap the data
    val remap_data = data.rdd.map( x => {
      (x._1, ml_linalg.Vectors.dense(remap_br.value.map( y => if (y == -1) 0 else x._2.toArray(y) ).toArray ) )
    })

    //Do the prediction of every model against every sample
    val out = rdd.cartesian( remap_data ).map( x => {
      (x._1._1, Map( (x._2._1, x._1._2.predict(x._2._2)) ))
    } ).reduceByKey( (x,y) => {x++y})

    //return a dataframe that is model x samples
    return DataFrame.create(out)
  }

}