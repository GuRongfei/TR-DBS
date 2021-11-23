 swig -python gfg.i
 g++ -c -fpic gfg_wrap.c gfg.c -I/home/grf/anaconda3/include/python3.8
 g++ -shared gfg.o gfg_wrap.o -o _oscillator_cpp.so
