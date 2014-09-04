package edu.ucsc.mltools

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext


class LabeledDataFrame(val sc:SparkContext, val labelName:String, val index : IndexedSeq[String], val rdd:RDD[(String,LabeledPoint)]) {

  def coalesce(numPartitions:Int = 1) : LabeledDataFrame = {
    return new LabeledDataFrame(sc, labelName, index, rdd.coalesce(numPartitions))
  }

}

