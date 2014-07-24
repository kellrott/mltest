package edu.ucsc.mltools

import breeze.linalg._
import java.io.File

/**
 * Created by kellrott on 6/29/14.
 */
class LocalDataFrame {

  def load(path:String, seperator:Char) = {

    val matrix = csvread(new File(path), separator='\t')

    println(matrix)
  }

}
