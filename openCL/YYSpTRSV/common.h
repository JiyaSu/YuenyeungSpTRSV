#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#ifndef BENCH_REPEAT
#define BENCH_REPEAT 20
#endif

#ifndef WARP_SIZE
#define WARP_SIZE   64
#endif

#ifndef WARP_PER_BLOCK
#define WARP_PER_BLOCK   4
#endif
