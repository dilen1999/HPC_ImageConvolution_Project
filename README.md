## High-Performance Parallel Image Convolution Filters Using Serial, OpenMP, MPI, CUDA, and Hybrid Techniques.

<img width="902" alt="Image" src="https://github.com/user-attachments/assets/af55a5d8-93dd-4a11-9b1e-b0af86134eae" />

## 🧩 Serial and OpenMP Implementations

### 🔧 Overview

This section of the project focuses on implementing **image convolution filters** using:

- ✅ A **Serial C++** implementation (for baseline performance)
- 🚀 A **Parallel OpenMP** implementation (for shared-memory acceleration)

Both versions apply common convolution filters such as **sharpen**, **blur**, and **edge detection** on grayscale images using the OpenCV library.

---

### 🖼️ Input & Output

- **Input**: Grayscale `.png` image
- **Output**: Filtered image saved in the `Results/` directory
- **Example Kernels**:
  - **Sharpen**:  
    ```
    [  0  -1   0 ]
    [ -1   5  -1 ]
    [  0  -1   0 ]
    ```
  - **Edge Detection**:  
    ```
    [ -1  -1  -1 ]
    [ -1   8  -1 ]
    [ -1  -1  -1 ]
    ```

---


### ⚙️ Compilation Instructions

> 💡 Ensure `OpenCV` is installed. On Ubuntu:  
> `sudo apt install libopencv-dev`

#### ✅ Compile Serial Version
```bash
g++ Serial/convolution_serial.cpp -o serial_convolution `pkg-config --cflags --libs opencv4`
````

#### ▶️ Running the Program
```bash
./serial_convolution
````

#### 🚀 Compile OpenMP Version
```bash
g++ -fopenmp Kernels/convolution_openmp.cpp -o openmp_convolution `pkg-config --cflags --libs opencv4`
````

#### ▶️ Running the Program
```bash
./openmp_convolution
````
###  Final results different between serial and openmp
![Image](https://github.com/user-attachments/assets/3c5c72b2-3ee8-405d-9e41-9159a69fab45)
