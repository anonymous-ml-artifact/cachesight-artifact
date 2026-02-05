#ifndef CACHE_SETS_H
#define CACHE_SETS_H

// Number of monitored cache sets
#define NUM_CACHE_SETS 10

// Cache sets corresponding to hot functions in libonnxruntime.so (from compute_hot_functions_cache_sets.py)
static const int hot_cache_sets[NUM_CACHE_SETS] = {
    1676, 1684, 9845, 3147, 7845, 3144, 3146, 3179, 3183, 1685
};



#endif // CACHE_SETS_H
