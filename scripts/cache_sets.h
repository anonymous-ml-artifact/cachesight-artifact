#ifndef CACHE_SETS_H
#define CACHE_SETS_H

// Number of monitored cache sets
#define NUM_CACHE_SETS 10

// Cache sets corresponding to hot functions in libonnxruntime.so (from compute_hot_functions_cache_sets.py)
static const int hot_cache_sets[NUM_CACHE_SETS] = {
    5658, 14681, 2122, 12639, 9878, 4978, 10993, 10992, 11006, 11004
};



#endif // CACHE_SETS_H
