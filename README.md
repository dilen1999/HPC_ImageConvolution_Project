# 📸 High-Performance Parallel Image Convolution Filters  
**Using Serial, OpenMP, MPI, CUDA, and Hybrid (MPI + CUDA) Techniques**

<img width="902" alt="Image" src="https://github.com/user-attachments/assets/af55a5d8-93dd-4a11-9b1e-b0af86134eae" />

---

## 🧩 Overview

This project demonstrates **image convolution filters** implemented using various **high-performance computing** techniques.

You can:
- 🟢 Run a **Serial C++ baseline**
- 🟢 Use **OpenMP** for shared memory parallelism
- 🟢 Use **MPI** for distributed memory parallelism
- 🟢 Use **CUDA** for GPU acceleration
- 🟢 Combine **MPI + CUDA** for a hybrid approach

✅ Each implementation applies common filters like **Sharpen**, **Blur**, and **Edge Detection** to a grayscale image using **OpenCV**.

## ✅ Speedup

| Method  | Serial Time | Method Time | Approx. Speedup |
| ------- | ----------- | ----------- | --------------- |
| Serial  | 0.39 s      | 0.39 s      | 1×              |
| OpenMP  | 0.39 s      | 0.12 s      | 3×              |
| MPI (4) | 0.39 s      | 0.098 s     | 4×              |
| CUDA    | 0.39 s      | 0.003 s     | 118×            |
| Hybrid  | 0.39 s      | 0.0037 s    | 107×            |


---

## 🖼️ Input & Output

- **Input**: Grayscale `.png` image (`Images/input.png`)
- **Output**: Filtered images saved to the `Results/` folder
- Each method produces:
  - `output_<method>_sharpen.png`
  - `output_<method>_blur.png`
  - `output_<method>_edge.png`

---

## 🔍 Example Kernels

| Filter | Kernel |
|--------|--------|
| **Sharpen** | `[[ 0 -1 0 ], [-1 5 -1], [ 0 -1 0 ]]` |
| **Blur**    | `[[ 1/9 1/9 1/9 ], [1/9 1/9 1/9], [1/9 1/9 1/9 ]]` |
| **Edge**    | `[[ -1 -1 -1 ], [-1 8 -1], [-1 -1 -1 ]]` |

---

## ⚙️ Build & Run Instructions

### ✅ Requirements

- OpenCV installed (`sudo apt install libopencv-dev`)
- `mpic++` for MPI, `nvcc` for CUDA
- NVIDIA GPU + CUDA drivers for CUDA/Hybrid

---




#### ✅ 1️⃣ Serial Version
```bash
g++ Serial/convolution_serial_opencv.cpp -o convolution_serial_opencv `pkg-config --cflags --libs opencv4`
````

#### ▶️ Running the Program
```bash
./convolution_serial_opencv
````
#### ✅ 2️⃣ OpenMP Version
```bash
g++ -fopenmp Openmp/convolution_openmp.cpp -o convolution_openmp `pkg-config --cflags --libs opencv4`
````

#### ▶️ Running the Program
```bash
export OMP_NUM_THREADS=4
./convolution_openmp
````
```bash
chmod +x run_openmp_threads.sh
./run_openmp_threads.sh
````

#### ✅ 3️⃣ MPI Version
```bash
mpic++ MPI/convolution_mpi.cpp -o convolution_mpi `pkg-config --cflags --libs opencv4`
````

#### ▶️ Running the Program
```bash
mpirun -np 4 ./convolution_mpi
````

#### ✅ 4️⃣ CUDA Version
```bash
nvcc CUDA/convolution_cuda.cu -o convolution_cuda `pkg-config --cflags --libs opencv4` -Xcudafe "--diag_suppress=611"
````

#### ▶️ Running the Program
```bash
./convolution_cuda
````

#### ✅ 5️⃣ Hybrid Version (MPI + CUDA)
```bash
nvcc -ccbin mpic++ Hybrid/convolution_mpi_cuda.cu -o convolution_mpi_cuda \
  -I/usr/lib/x86_64-linux-gnu/openmpi/include \
  -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi \
  `pkg-config --cflags --libs opencv4` -lmpi
````

#### ▶️ Running the Program
```bash
mpirun -np 2 ./convolution_mpi_cuda
````

#### ✅ 6️⃣ Compare Accuracy (RMSE)
```bash
g++ Compare/compare_rmse.cpp -o compare_rmse `pkg-config --cflags --libs opencv4`
````

#### ▶️ Running the Program
```bash
#!/bin/bash
echo "Comparing Serial vs MPI..."
./compare_rmse ../Results/output_serial.png ../Results/output_mpi.png
````
```bash
#!/bin/bash
echo "Comparing Serial vs CUDA..."
./compare_rmse ../Results/output_serial.png ../Results/output_cuda.png
````
```bash
#!/bin/bash
echo "Comparing Serial vs Hybrid..."
./compare_rmse ../Results/output_serial.png ../Results/output_hybrid.png
````
```bash
chmod +x compare_serial_mpi.sh compare_serial_cuda.sh compare_serial_hybrid.sh

# Or run all at once using a wrapper:
chmod +x compare_all.sh
./compare_all.sh
````
````
