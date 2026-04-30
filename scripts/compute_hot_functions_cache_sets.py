import sys
import struct
import os
import re

PAGE_SIZE = 4096
PAGEMAP_ENTRY_SIZE = 8
CACHE_LINE_SIZE = 64
L3_NUM_SETS = 16384


# Our hot function offsets from simpleperf (without the lib prefix)
HOT_FUNCTION_OFFSETS = [
    0xcc1f78,
    0xd05a14,
    0xd29c80,
    0xd2fc50,
    0xd190c0,
    0xd0b880,
    0xd259b8,
    0xd259fc,
    0xd25a44,
    0xd25acc,
    0xd25c0c,
    0xd25a98,
    0xd190f4,
    0xd25b94,
    
]




def read_maps_file(maps_file="maps.txt"):
    with open(maps_file, "r") as f:
        for line in f:

            if "r-xp" not in line:
                continue

            if "base.apk" not in line:
                continue

            parts = line.split()

            try:
                addr_range = parts[0].split('-')
                start = int(addr_range[0], 16)


                base_addr = start

            except Exception:
                continue

            print("FOUND APK EXECUTABLE MAPPING:")
            print(line.strip())
            print(f"Base addr: {hex(base_addr)}")

            return base_addr

    raise Exception("base.apk mapping not found")

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
    
  
    if (val >> 63) == 0:
        return None
        
    # Extract PFN (bits 0-54)
    pfn = val & ((1 << 55) - 1)
    paddr = (pfn * PAGE_SIZE) + (vaddr % PAGE_SIZE)
    return paddr

def compute_cache_mapping(paddr):
    line = paddr >> 6

    # more stable ARM-style approximation
    set_index = (
        line ^
        (line >> 15) ^
        (line >> 23)
    ) & (L3_NUM_SETS - 1)

    return {
        'set': set_index,
        'page_offset': paddr & (PAGE_SIZE - 1),
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
    from collections import defaultdict

    # group addresses by cache set
    groups = defaultdict(list)

    for r in results:
        groups[r['cache_set']].append(r)

    eviction_sets = []

    for set_id, items in groups.items():
        if len(items) >= 12:   # threshold

            addresses = []
            seen_pages = set()

            for x in items:
                page = x['paddr'] >> 12

                if page not in seen_pages:
                    seen_pages.add(page)

                    aligned_addr = x['paddr'] & ~(CACHE_LINE_SIZE - 1)
                    addresses.append(hex(aligned_addr))

                if len(addresses) >= 32:
                    break

            eviction_sets.append({
                "cache_set": set_id,
                "addresses": addresses
            })
        
    print(f"\nGenerated eviction_sets.json with {len(eviction_sets)} unique targets")

if __name__ == "__main__":
    main()
