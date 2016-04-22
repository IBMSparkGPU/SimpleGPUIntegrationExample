/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ibm.spark.gpu.simple

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.Row


/**
 * @author bherta
 */
object JNITest {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple GPU application to test JNI")
    val sparkContext = new SparkContext(conf)

    val useGPU = conf.getBoolean("spark.useGPU", true) // configuration flag example
    
    val rddInput = sparkContext.parallelize(Seq.range(0,10000))
    
    val cpuresult = rddInput.map(v=>v+1).take(3) // add one to each value, so now they run from 1 to 10000, in the CPU
    println("Values should be: "+cpuresult(0)+","+cpuresult(1)+","+cpuresult(2)+"...")
    // Note that the take() above forces the creation of the "rddInput" RDD
    
    val gpuresult = rddInput.mapPartitionsWithIndex((partitionId, iterator) => {
      if (useGPU) {
        println("Using GPU")
        System.loadLibrary("JNIInterface") // load the JNI interface, as this is the first time we touch it
        val jni = new JNIInterface
        jni.incrementArrayValues(partitionId, iterator.toArray).toIterator
      }
      else {
        println("Not using GPU")
        iterator.map{v=>v+1}
      }
    }).take(3)
    
    println("RDD values: "+gpuresult(0)+","+gpuresult(1)+","+gpuresult(2)+"...")
    
    // Data frames
    val sqlContext = new SQLContext(sparkContext);
    val dfInput = sqlContext.createDataFrame(rddInput.map(v=>Row(v)), StructType(List(StructField("value", IntegerType, true))))
    
    // TODO - call the GPU to produce gpuDfResult
    val gpuDfResult = dfInput.take(3)
    
    println("DataFrame values: "+gpuDfResult(0).get(0)+","+gpuDfResult(1).get(0)+","+gpuDfResult(2).get(0)+"...")
  }
}