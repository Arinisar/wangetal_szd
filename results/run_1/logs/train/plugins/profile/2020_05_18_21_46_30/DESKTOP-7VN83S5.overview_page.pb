�	��ᔹ�$@��ᔹ�$@!��ᔹ�$@      ��!       "e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��ᔹ�$@��lu9��?1|E�^�@I�r���6@:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"@20.9 % of the total step time sampled is spent on Kernel Launch.*moderate2A7.2 % of the total step time sampled is spent on All Others time.#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��lu9��?��lu9��?!��lu9��?      ��!       "	|E�^�@|E�^�@!|E�^�@*      ��!       2      ��!       :	�r���6@�r���6@!�r���6@B      ��!       J      ��!       R      ��!       Z      ��!       JGPU�"j
@gradients/model_1/conv1d_3/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter���A��?!���A��?"j
@gradients/model_1/conv1d_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter.!&>R�?!��x0��?"8
model_1/conv1d_3/convolutionConv2D���?�?!p����#�?"8
model_1/conv1d_2/convolutionConv2D��U��?!�Hp����?"j
@gradients/model_1/conv1d_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter n��ΰ?!�>ţ
�?"/
mul_2/yUnsortedSegmentSum��sͳ��?!EO��?"*
gradients/AddN_1AddNX���Cu�?!�p@?���?"8
model_1/conv1d_1/convolutionConv2D��et��?!�͆�]>�?"@
embedding_1/embedding_lookupResourceGather<�E�?!�v����?"�
pgradients/model_1/conv1d_3/convolution_grad/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:TransposeUnknown�b�i���?!����>��?2blackQ      Y@"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high@20.9 % of the total step time sampled is spent on Kernel Launch.moderate"A7.2 % of the total step time sampled is spent on All Others time.*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).:
Refer to the TF2 Profiler FAQ2"GPU(: 