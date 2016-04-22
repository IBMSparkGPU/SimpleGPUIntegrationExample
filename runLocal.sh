#bherta: Runs the program in local mode
SPARK_HOME=$HOME/Spark/spark-1.6.0-bin-hadoop2.6

$SPARK_HOME/bin/spark-submit --master local[*] --class com.ibm.spark.gpu.simple.JNITest --driver-library-path `pwd` JNITest.jar
