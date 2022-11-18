# matrix-calc
calc coma&amp;kronecker


dpcpp -O0 -g -march=sapphirerapids coma.cpp -o coma

dpcpp -O3 -march=core-avx2 coma-avx256.cpp -o coma-avx256

dpcpp -O3 -xCORE-AVX512 coma.cpp -o coma

dpcpp -O3 -xCORE-AVX2 coma-avx256.cpp -o coma-avx256

dpcpp -O0 -xCORE-AVX512 coma.cpp -o coma

dpcpp -O0 -xCORE-AVX2 coma-avx256.cpp -o coma-avx256

