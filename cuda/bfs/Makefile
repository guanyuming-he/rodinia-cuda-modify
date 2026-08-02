include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = bfs.cu

EXE = bfs

release: $(SRC)
	$(CC) -lineinfo $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

enum: $(SRC)
	$(CC) -lineinfo -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CC) -lineinfo -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CC) -lineinfo -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
