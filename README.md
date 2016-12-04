# Benchmark TensorFlow

TensorFlow is a powerful machine learning library that has been used to build various learning models. As it is gaining popularity, this project provides different benchmarks aiming for deeper understanding of tensorflow computation.  

There might be different ways of implementation for the same task. However, due to the complexity of tensorflow computational model, it is often hard to analyze the computation without actual comparison. This project compares computation time of different but equivalent implementations and show differences of computation time. 

The first example compares computation time with tensor operations and that with the unpacked list of tensors. It shows that working with unpacked tensors costs much more time than working with the tensor directly. Therefore, it is a good practice to vectorize computation when possible. 

