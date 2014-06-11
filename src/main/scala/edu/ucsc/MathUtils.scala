import edu.ucsc.mltest

import breeze.{linalg => br_linalg}
import breeze.stats

object MathUtils {


  def corr(a: br_linalg.DenseVector[Double], b: br_linalg.DenseVector[Double]): Double = {
    if (a.length != b.length)
      return Double.NaN

    val n = a.length

    val (amean, avar) = stats.meanAndVariance(a)
    val (bmean, bvar) = stats.meanAndVariance(b)
    val astddev = math.sqrt(avar)
    val bstddev = math.sqrt(bvar)

    1.0 / (n - 1.0) * br_linalg.sum( ((a - amean) / astddev) :* ((b - bmean) / bstddev) )
  }


}