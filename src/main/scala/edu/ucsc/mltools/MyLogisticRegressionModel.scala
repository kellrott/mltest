package edu.ucsc.mltools

import org.apache.spark.mllib.regression.GeneralizedLinearModel
import org.apache.spark.mllib.{linalg => ml_linalg}
import breeze.{linalg => br_linalg}

/**
 * Created by kellrott on 6/29/14.
 */
class MyLogisticRegressionModel(
                                                override val weights: ml_linalg.Vector,
                                                override val intercept: Double)
  extends GeneralizedLinearModel(weights, intercept) {
     override protected def predictPoint(dataMatrix: ml_linalg.Vector, weightMatrix: ml_linalg.Vector, intercept: Double): Double = {
       val margin =
         br_linalg.Vector(weightMatrix.toArray).dot(br_linalg.Vector(dataMatrix.toArray)) + intercept
       val score = 1.0/ (1.0 + math.exp(-margin))
       return score
     }
   }
