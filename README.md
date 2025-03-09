# cuda
## Day 1 : Addng two vectors to a result vector
1. Learnt few nuances of pointers. float* A equivalent to float *A
2. But float *A, *B; not equivalent to float* A, B
3. Because, * is binded to variable name and not variable type. 
4. So, float* A,B; is float *A, float B. 
## Day 2 : Multiply two matrices 
## Day 3 : Apply softmax to an array
1. Max trick : Multiply divide a constant to softmax function but the ratio remains unchanged. express f(xi) = f(xi-xmax)
2. declaring shared variables
3. Sync threads is used if we want al threads to see the same state of the variable.
4. atomicAdd
## Day 4 : Very basic reduction pattern
## Day 5 : Transpose of matrix, naive approach