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

#include "com_ibm_spark_gpu_simple_JNIInterface.h"
#include "cuda/gpu_test.h"
#include <ctype.h>

JNIEXPORT jintArray JNICALL Java_com_ibm_spark_gpu_simple_JNIInterface_incrementArrayValues
    (JNIEnv *env, jobject obj, jint partitionId, jintArray input) {

	// get length of the array
	jsize len = env->GetArrayLength(input);
	// create a new buffer for output data (RDDs are immutable, we don't mutate the input)
	jintArray output = env->NewIntArray(len);

	// get a pointer to the raw input & output arrays, pinning them in memory
	jint* inputBuffer = (jint*) env->GetPrimitiveArrayCritical(input, 0);
	jint* outputBuffer = (jint*) env->GetPrimitiveArrayCritical(output, 0);

	fprintf(stderr, "Processing partition %d, len=%d \n", partitionId, len);

	// Demonstration of bringing the input over to CUDA code.  Doesn't actually compute anything yet
	int success = gpu_increment(partitionId, len, inputBuffer, outputBuffer);

	if (success != 0) { // GPU computation failed.  Perform it on the CPU
		fprintf(stderr, "Unable to process partition %d on GPU.... falling back to the CPU\n", partitionId);
		for (jsize i=0; i<len; i++)
			outputBuffer[i] = inputBuffer[i] + 1;
	}
	else
		fprintf(stderr, "Partition %d successfully completed on GPU\n", partitionId);

	// tell the JVM that we're done with the buffers, so they can be unpinned
	env->ReleasePrimitiveArrayCritical(output, outputBuffer, 0);
	env->ReleasePrimitiveArrayCritical(input, inputBuffer, 0);

	return output;
}
