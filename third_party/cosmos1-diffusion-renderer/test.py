def split_list_with_overlap(lst, chunk_size, overlap_size, chunk_mode="all"):
    """Splits a list into chunks with overlapping elements."""
    if overlap_size >= chunk_size:
        raise ValueError("Overlap size must be less than the chunk size.")

    chunks = []
    step = chunk_size - overlap_size

    for i in range(0, len(lst) - overlap_size, step):
        chunk = lst[i:i + chunk_size]
        chunks.append(chunk)
        if chunk_mode == "first":
            break

    if len(chunks) > 0 and chunk_mode == "drop_last" and len(chunks[-1]) < chunk_size:
        chunks = chunks[:-1]

    return chunks

lst = list(range(199))
chunk_size = 57
overlap_size = 50
chunk_mode = "all"
result = split_list_with_overlap(lst, chunk_size, overlap_size, chunk_mode)
print(f"Lentgh of resulting chunks: {len(result)}")
print("Resulting chunks:")
for chunk in result:
    print(f"From index {lst.index(chunk[0])} to {lst.index(chunk[-1])}")