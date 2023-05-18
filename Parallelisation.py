import multiprocessing as mp


def parallel_procces_graphs(file_name,vertices,players,func, max_processes=None):
    num_lines = sum(1 for line in open(file_name))
    n_lines = vertices+2
    # Maximum number of processes we can run at a time
    cpu_count = mp.cpu_count() if max_processes is None else max_processes
    chunk_size = ((num_lines//n_lines) // min(cpu_count, (num_lines//n_lines))) * n_lines
    # Arguments for each chunk (eg. [('input.txt', 0, 32), ('input.txt', 32, 64)])
    chunk_args = []
    with open(file_name, 'r') as f:
        chunk_start = 0
        # Iterate over all chunks and construct arguments for `process_chunk`
        while chunk_start < num_lines:
            chunk_end = min(num_lines, chunk_start + chunk_size)
            # print(chunk_start, chunk_end)
            # Save `process_chunk` arguments
            args = (file_name, chunk_start, chunk_end,vertices,players,func)
            chunk_args.append(args)
            # Move to the next chunk
            chunk_start = chunk_end

    with mp.Pool(cpu_count) as p:
        # Run chunks in parallel
        chunk_results = p.starmap(process_graphs, chunk_args)
    results = []
    # Combine chunk results into `results`
    for chunk_result in chunk_results:
        for result in chunk_result:
            results.append(result)
    return results

def process_graphs(file_name, chunk_start, chunk_end,vertices,players,func):
    chunk_results = []
    with open(file_name, 'r') as f:
        for index,line in enumerate(f):
            if index < chunk_start:
                continue
            chunk_start += 1
            if chunk_start > chunk_end:
                break
            if index%(vertices+2)==0:
                continue
            if index%(vertices+2)==1:
                cur_graph={}
            if vertices+1>=index%(vertices+2)>=2:
                head,tails = line.split(":")
                cur_graph[int(head.split()[0])] = list(map(int,tails[:-2].split()))
            if index%(vertices+2) == vertices+1:
                chunk_results+=func(cur_graph,vertices,players)
    return chunk_results