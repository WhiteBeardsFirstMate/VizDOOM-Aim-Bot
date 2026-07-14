def flush_nstep(buffer, gamma):
    R = 0.0
    n = len(buffer)
    for i, (_, _, r, _, _) in enumerate(buffer):
        R += (gamma ** i) * r
    s0, a0, _, _, _ = buffer[0]
    _, _, _, ns, d = buffer[-1]
    buffer.popleft()
    return R, s0, a0, ns, d, n
