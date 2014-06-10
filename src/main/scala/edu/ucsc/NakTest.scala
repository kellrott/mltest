package edu.ucsc

/**
 * Created by kellrott on 6/6/14.
 */


import nak.classify.LogisticClassifier
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.io.CSVReader
import breeze.linalg.csvread
import java.io.FileReader
import java.io.File

class LocalDataFrame {

  def load(path:String, seperator:Char) = {

    val matrix = csvread(new File(path), separator='\t')

    println(matrix)
  }

}

object NakTest {

  def main(args:Array[String]) = {

    val df = new LocalDataFrame()
    df.load(args(0), '\t')
    /*

    val classifier = new LogisticClassifier.Trainer[Int,DenseVector[Double]].train(vectors)
    for( ex <- vectors) {
      val guessed = classifier.classify(ex.features)
      println(guessed,ex.label)
    }
    */

  }


}
