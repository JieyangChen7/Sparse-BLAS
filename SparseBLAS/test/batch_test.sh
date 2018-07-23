#!/bin/bash

NGPU=8
KERNEL=1

./spmv g 10000 $(NGPU) 10 $(KERNEL)
./spmv g 20000 $(NGPU) 10 $(KERNEL)
./spmv g 30000 $(NGPU) 10 $(KERNEL)
./spmv g 100000 $(NGPU) 10 $(KERNEL)
./spmv g 200000 $(NGPU) 10 $(KERNEL)
./spmv g 300000 $(NGPU) 10 $(KERNEL)