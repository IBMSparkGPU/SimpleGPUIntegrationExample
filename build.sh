#bherta: This compiles the example program, including JNI-scala trickery
#apologies for not making a proper makefile (yet)

# Modify these to point to your scala and spark folders
SCALA_HOME=/opt/scala-2.10.5
SPARK_HOME=$HOME/Spark/spark-1.6.0-bin-hadoop2.6


SCALA_CP=$SCALA_HOME/lib/scala-library.jar:$SCALA_HOME/lib/scala-reflect.jar
mkdir -p bin
cd src
# compile the Scala code
$SCALA_HOME/bin/scalac -classpath $SPARK_HOME/lib/spark-assembly-1.6.0-hadoop2.6.0.jar -d ../bin com/ibm/spark/gpu/simple/*.scala
# generate the JNI header from the scala class
javah -cp ../bin:$SCALA_CP com.ibm.spark.gpu.simple.JNIInterface
# create the jar file
jar -cf ../JNITest.jar -C ../bin .

# compile the native & cuda code
nvcc -c cuda/gpu_test.cu -o cuda/gpu_test.o -I/usr/local/cuda/include -gencode arch=compute_35,code=sm_35 -gencode arch=compute_20,code=sm_21 -use_fast_math -g -m64 -maxrregcount=32 -ftz=true -prec-div=false -prec-sqrt=false -Xcompiler "-fPIC -c -O2 -g "
nvcc -c cuda/utilities.cu -o cuda/utilities.o -I/usr/local/cuda/include -gencode arch=compute_35,code=sm_35 -gencode arch=compute_20,code=sm_21 -use_fast_math -g -m64 -maxrregcount=32 -ftz=true -prec-div=false -prec-sqrt=false -Xcompiler "-fPIC -c -O2 -g "
g++ -c -fPIC -I$JAVA_HOME/include -I$JAVA_HOME/include/linux -o JNIInterface.o JNIInterface.cpp
# compile the C++ implementation into a shared library
cd ..
g++ -o libJNIInterface.so src/JNIInterface.o src/cuda/gpu_test.o src/cuda/utilities.o -fPIC -shared -Wall -O3 -lcudart -lm -lcuda -L/usr/local/cuda/lib64
