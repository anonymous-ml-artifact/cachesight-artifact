import sys
import struct
import os
import re

PAGE_SIZE = 4096
PAGEMAP_ENTRY_SIZE = 8
CACHE_LINE_SIZE = 64
L3_NUM_SETS = 16384


# Your hot function offsets from simpleperf (without the lib prefix)
HOT_FUNCTION_OFFSETS = [
    0xc83318,  # Your top hot function

    0xC83530,
    0xC7DD54,
    0xC83304,
    0xC9C2F8,
    0xC8C958,
    0xC9C210,
    0xC9C214,
    0xC9C2B0,
    0xC9CAD8,
    0xC9CADC,
    0xC9CBC0,
    0xC83528,
    0xC9C218,
    0xC83564,
    0xC9C21C,
    0xC83554,
    0xC9C2B4,
    0xC9C20C,
    0xC9C2FC

    
    # Add more as needed
]




def read_maps_file(maps_file="maps.txt"):
    with open(maps_file, "r") as f:
        for line in f:
            if "base.apk" in line and "r-xp" in line and "mobilenetv2tfliteapp" in line:
                parts = line.split()
                addr_range = parts[0].split('-')
                start = int(addr_range[0], 16)
                print(f"Found base.apk executable at: {hex(start)}")
                return start
    raise Exception("No executable APK mapping found")

def parse_simpleperf_offset(offset_str):
    """Parse offset from simpleperf format like [+c83318] or just c83318"""
    # Remove brackets and + sign if present
    offset_str = offset_str.strip('[]').lstrip('+')
    # Parse as hex
    return int(offset_str, 16)

def virt_to_phys(pagemap_data, vaddr):
    """Convert virtual address to physical using pagemap data"""
    vpn = vaddr // PAGE_SIZE
    offset = vpn * PAGEMAP_ENTRY_SIZE
    
    if offset + PAGEMAP_ENTRY_SIZE > len(pagemap_data):
        return None
        
    entry = pagemap_data[offset:offset + PAGEMAP_ENTRY_SIZE]
    val = struct.unpack("Q", entry)[0]
    
    # Check if page is present (bit 63)
    if (val >> 63) == 0:
        return None
        
    # Extract PFN (bits 0-54)
    pfn = val & ((1 << 55) - 1)
    paddr = (pfn * PAGE_SIZE) + (vaddr % PAGE_SIZE)
    return paddr

def compute_cache_mapping(paddr):
    """Compute cache set from physical address"""
    # Cache set index (bits 6-19 for 16K sets)
    set_index = (paddr >> 6) & (L3_NUM_SETS - 1)
    
    # 4KB page offset for prime+probe
    page_offset = paddr & (PAGE_SIZE - 1)
    
    return {
        'set': set_index,
        'page_offset': page_offset,
        'paddr': paddr
    }

def main():
    # Check if files exist
    if not os.path.exists("maps.txt"):
        print("ERROR: maps.txt not found!")
        print("Get it with: adb shell su -c 'cat /proc/PID/maps' > maps.txt")
        return
        
    if not os.path.exists("pagemap.bin"):
        print("ERROR: pagemap.bin not found!")
        print("Get it with: adb shell su -c 'dd if=/proc/PID/pagemap of=/data/local/tmp/pagemap.bin bs=8'")
        print("Then: adb pull /data/local/tmp/pagemap.bin")
        return

    # Step 1: Read base address
    try:
        base_addr = read_maps_file("maps.txt")
        print(f"\nBase address found: 0x{base_addr:x}\n")
    except Exception as e:
        print(f"Error: {e}")
        print("\nDEBUG: First 10 lines of maps.txt containing 'apk' or '.so':")
        with open("maps.txt", "r") as f:
            count = 0
            for line in f:
                if ('apk' in line.lower() or '.so' in line) and count < 10:
                    print(f"  {line.strip()[:120]}...")
                    count += 1
        return

    # Step 2: Load pagemap
    with open("pagemap.bin", "rb") as f:
        pagemap_data = f.read()
    print(f"Loaded {len(pagemap_data)} bytes of pagemap data\n")

    # Step 3: Process hot function offsets
    results = []
    print("Processing hot function offsets:")
    print("-" * 60)
    
    for offset in HOT_FUNCTION_OFFSETS:
        vaddr = base_addr + offset
        paddr = virt_to_phys(pagemap_data, vaddr)
        
        if paddr:
            cache_info = compute_cache_mapping(paddr)
            results.append({
                'offset': offset,
                'vaddr': vaddr,
                'cache_set': cache_info['set'],
                'page_offset': cache_info['page_offset'],
                'paddr': paddr
            })
            print(f"0x{offset:08x} -> Set {cache_info['set']:5d}, Page offset 0x{cache_info['page_offset']:03x}")
        else:
            print(f"0x{offset:08x} -> NOT MAPPED (page not present)")

    # Step 4: Generate eviction sets JSON
    if results:
        import json
        eviction_sets = []
        seen_offsets = set()
        
        for r in results:
            page_off = r['page_offset'] & ~0x3F  # Align to cache line
            if page_off not in seen_offsets:
                seen_offsets.add(page_off)
                eviction_sets.append({
                    "target_offset": f"0x{page_off:x}",
                    "cache_set": r['cache_set'],
                    "source": f"function_0x{r['offset']:x}"
                })
        
        with open("eviction_sets.json", "w") as f:
            json.dump({"eviction_sets": eviction_sets}, f, indent=2)
        
        print(f"\nGenerated eviction_sets.json with {len(eviction_sets)} unique targets")

if __name__ == "__main__":
    main()
