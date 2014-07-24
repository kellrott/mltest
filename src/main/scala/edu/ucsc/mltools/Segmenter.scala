package edu.ucsc.mltools

import scala.collection.mutable.ArrayBuffer

/**
 * Created by kellrott on 6/29/14.
 */
class Segmenter[T] (in: Iterator[T], blockSize:Int) extends Iterator[Seq[T]] {

  var block : Seq[T] = null

  def queue_block() {
    if (block == null) {
      var i = 0
      val o = new ArrayBuffer[T]()
      while (i < blockSize && in.hasNext) {
        o += in.next()
        i += 1
      }
      block = o.toSeq
    }
  }
  override def hasNext: Boolean = {
    queue_block()
    return block != null
  }

  override def next(): Seq[T] = {
    queue_block()
    if (block == null) {
      return null
    }
    val o = block
    block = null
    return o
  }

}
