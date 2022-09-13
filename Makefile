all:
	gcc sparseMatrixVectorMultiplicationCPU.c -o mainCPU -fopenmp -O3 compute.c ./IO/utils.c ./IO/io.c ./IO/mmio.c
	nvcc sparseMatrixVectorMultiplicationGPU.cu -o mainGPU -O3 ./computeGPU.cu ./compute.c ./IO/io.c ./IO/mmio.c ./IO/utils.c
CPU:
	gcc sparseMatrixVectorMultiplicationCPU.c -o mainCPU -fopenmp -O3 compute.c ./IO/utils.c ./IO/io.c ./IO/mmio.c
GPU:
	nvcc sparseMatrixVectorMultiplicationGPU.cu -o mainGPU -O3 ./computeGPU.cu ./compute.c ./IO/io.c ./IO/mmio.c ./IO/utils.c
cleanCPU:
	rm mainCPU
cleanGPU:
	rm mainGPU
clean:
	rm mainCPU
	rm mainGPU