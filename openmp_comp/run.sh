
gcc -fopenmp main_mp.c -o mp_main

if [ "$1" == "mp" ]; then
  rm mp_main
 
elif [ "$1" == "tst" ]; then
  rm acctst
  $PGCC ${PGCC_FLAGS} -o acctst acctst.c
fi
