#!/bin/bash

echo " Serial vs OpenMP:"
./compare_rmse ../Results/output_serial_sharpen.png ../Results/output_openmp_sharpen.png
./compare_rmse ../Results/output_serial_blur.png ../Results/output_openmp_blur.png
./compare_rmse ../Results/output_serial_edge.png ../Results/output_openmp_edge.png

echo " Serial vs MPI:"
./compare_rmse ../Results/output_serial_sharpen.png ../Results/output_mpi_sharpen.png
./compare_rmse ../Results/output_serial_blur.png ../Results/output_mpi_blur.png
./compare_rmse ../Results/output_serial_edge.png ../Results/output_mpi_edge.png

echo " Serial vs CUDA:"
./compare_rmse ../Results/output_serial_sharpen.png ../Results/output_cuda_sharpen.png
./compare_rmse ../Results/output_serial_blur.png ../Results/output_cuda_blur.png
./compare_rmse ../Results/output_serial_edge.png ../Results/output_cuda_edge.png

echo " Serial vs Hybrid:"
./compare_rmse ../Results/output_serial_sharpen.png ../Results/output_hybrid_sharpen.png
./compare_rmse ../Results/output_serial_blur.png ../Results/output_hybrid_blur.png
./compare_rmse ../Results/output_serial_edge.png ../Results/output_hybrid_edge.png

echo " All RMSE done!"
