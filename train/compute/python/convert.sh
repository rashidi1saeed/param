#!/bin/bash

worker0=( test/data/dlrm_kineto/worker0* )
worker1=( test/data/dlrm_kineto/worker1* )
worker2=( test/data/dlrm_kineto/worker2* )
worker3=( test/data/dlrm_kineto/worker3* )
worker4=( test/data/dlrm_kineto/worker4* )
worker5=( test/data/dlrm_kineto/worker5* )
worker6=( test/data/dlrm_kineto/worker6* )
worker7=( test/data/dlrm_kineto/worker7* )

python3 ./tools/trace_link.py --et-file test/data/dlrm_pytorch_et/dlrm_eg_0.json --kineto-file "${worker0[0]}" --exact-match --annotation 'enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__'
python3 ./tools/trace_link.py --et-file test/data/dlrm_pytorch_et/dlrm_eg_1.json --kineto-file "${worker1[0]}" --exact-match --annotation 'enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__'
python3 ./tools/trace_link.py --et-file test/data/dlrm_pytorch_et/dlrm_eg_2.json --kineto-file "${worker2[0]}" --exact-match --annotation 'enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__'
python3 ./tools/trace_link.py --et-file test/data/dlrm_pytorch_et/dlrm_eg_3.json --kineto-file "${worker3[0]}" --exact-match --annotation 'enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__'
python3 ./tools/trace_link.py --et-file test/data/dlrm_pytorch_et/dlrm_eg_4.json --kineto-file "${worker4[0]}" --exact-match --annotation 'enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__'
python3 ./tools/trace_link.py --et-file test/data/dlrm_pytorch_et/dlrm_eg_5.json --kineto-file "${worker5[0]}" --exact-match --annotation 'enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__'
python3 ./tools/trace_link.py --et-file test/data/dlrm_pytorch_et/dlrm_eg_6.json --kineto-file "${worker6[0]}" --exact-match --annotation 'enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__'
python3 ./tools/trace_link.py --et-file test/data/dlrm_pytorch_et/dlrm_eg_7.json --kineto-file "${worker7[0]}" --exact-match --annotation 'enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__'

mkdir test/data/et_plus
mv ./test/data/dlrm_pytorch_et/*_plus.json ./test/data/et_plus/

