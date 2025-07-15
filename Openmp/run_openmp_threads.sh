
echo "🔹 Running OpenMP with thread counts 1 to 20"

for threads in {1..20}
do
    echo ""
    echo "==============================="
    echo " OMP_NUM_THREADS=$threads"
    export OMP_NUM_THREADS=$threads
    ./convolution_openmp   
done
