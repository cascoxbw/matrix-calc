
#BUILD_DIR := build

#MODULE_SRCS :=\
#        coma.cpp 

CC := gcc

#TARGET_PROCESSOR := -xCORE-AVX512

CC_FLAGS := -march=native

#MATRIX_SRCS := $(MODULE_SRCS)
#MATRIX_OBJS := $(foreach file,$(MATRIX_SRCS),$(BUILD_DIR)/$(file:.cpp=.o))

MATRIX_APP := app

#all : $(MATRIX_APP)
#	@echo "matrix built."
	dpcpp -O0 -g -march=sapphirerapids coma.cpp -o coma
	icx -O0 -g -march=native coma.cpp -o coma 
	gcc -O2 -march=native coma.cpp -o coma 
	gcc -O0 -march=native coma.cpp -o coma	
	gcc -O0 -g -march=native coma.cpp -o coma
	
#$(MATRIX_APP) : coma.o
#	$(CC) -o $@ $^ -O0 $(CC_FLAGS)
#.c.o:
#	$(CC) -c $< 

.PHONY: clean prepare xclean

prepare:
	#-mkdir -p $(BUILD_DIR)

clean :
	-rm -rf *.o coma

xclean : clean

