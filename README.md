# Introduction
In this assignment, you will investigate the performance difference between a GPU and a CPU.

![Matmul](img/matmul.png)

Your task is to measure the time it takes to multiply two square matrices and examine the relationship between matrix size and computation time. Your objective is to create a graph similar to the one shown below. 

![CPU Exectime](img/cpu_exectime.png)

The graph should include both GPU and CPU time measurements. Additionally, you should attempt to explore matrix multiplication with larger matrix sizes. What is the largest matrix size that is supported by the underlying hardware? 

## Matrix multiplication using Tensforflow
TensorFlow is an open-source machine learning framework developed by Google for building and training neural networks. Matrix multiplication is a foundational operation in many machine learning models such as neural networks, which are are composed of layers of neurons that perform computations. In a feedforward neural network, both the neural network itself and the input data is typically represented as matrices. Matrix multiplication is vital for training neural networks - a process that can take days or even weeks, depending on the complexity of the model and the computational resources available.

To use TensorFlow to perform matrix multiplications, you first need to import the TensorFlow library into your Python environment. You can do this by opening a new cell in your Jupyter Notebook and paste the following code:

```python
import tensorflow as tf
import time
```

We are going to generate two matrices containing random data. In the example below, the size of the matrices is 5 X 5, containing 25 *float32* values.

```python
size = 5
A = tf.random.normal([size, size])
B = tf.random.normal([size, size])
```

To p
```python
with tf.device('/CPU:0'):
    start_time = time.time()
    C_cpu = tf.matmul(A, B)
    C_cpu.numpy()
    cpu_time = time.time() - start_time
print(f"CPU time: {cpu_time}")
```

The C_cpu.numpy() is a Python Numpy array, for example containing the following data after the multiplication.
```python
array([[-1.0926545 ,  3.1091492 , -4.4844866 , -1.9047767 ,  0.39488536],
       [-0.91220856,  1.1700673 , -1.4075694 , -1.5006504 ,  0.8585447 ],
       [-1.1020844 ,  1.5990988 , -3.7829857 , -3.1916823 ,  2.8985267 ],
       [ 3.3098688 , -5.0233254 ,  5.299614  ,  0.36826795,  4.2648416 ],
       [-0.9915411 ,  1.6925802 , -0.9410507 ,  2.2324817 , -3.5573423 ]],
      dtype=float32)
```

The following code peforms matrix multiplication using a GPU.
## GPU 
```python
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        start_time = time.time()
        C_gpu = tf.matmul(A, B)
        C_gpu.numpy()
        gpu_time = time.time() - start_time
    print(f"GPU time: {gpu_time}")
else:
    print("No GPU found. Please ensure TensorFlow is setup with GPU support.")
```

## Hint 1: Python code structure 
Develop a Python function that perform the multiplication

```python
def measure_time(size, device):
    ...
    return time.time() - start_time
```

Construct a for-loop to iteratively execute the measure_time function over a specified range of intervals.
```python
cpu_times = []
gpu_times = []
sizes = range(5, 5001, 20) 

for size in sizes:
    gpu_time = measure_time(size, "/GPU:0")
    gpu_times.append(gpu_time)
```

Use the code below to plot the collected measurements.
```python
import matplotlib

plt.figure(figsize=(10, 5))
plt.plot(sizes, cpu_times, label='CPU Time')
plt.plot(sizes, gpu_times, label='GPU Time')
plt.xlabel('Matrix Size')
plt.ylabel('Computation Time (seconds)')
plt.title('Computation Time for Matrix Multiplication')
plt.legend()
plt.show()
```

## Hint 2: Matrix memory footprint
Let assume the matrix size is 20000 elements, then 
20000 × 20000 × 4 = 16000000 bytes = 1.49 GiB
is needed to store one matrix.

For all matrices, 1.49 X 3 = 4.47 GiB is needed.

The Nvidia 2080 Ti GPU has exactly 11 GiB. What is the maximum matrix dimension we can calculate using a Nvidia 2080 Ti GPU?
