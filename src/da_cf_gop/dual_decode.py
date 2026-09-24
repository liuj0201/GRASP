"""Exact joint edit-graph/CTC inference for anchored acceptable pronunciations.

The edit graph allows one OK/SUB/DEL decision per reference anchor and up to K
insertions per gap. Acoustic alignments, edit paths, and acceptable alternatives
are marginalized. Deletions are epsilon arcs, traversed only immediately before
a *new* CTC nonblank emission, or at termination. Traversing them before blanks
or repeated frame labels would count one (edit path, CTC path) multiple times.

Complete accepted surface strings are reserved for zero-error explanations:
the same string cannot acquire errors through a redundant DEL+INS edit path.
Acceptable alternatives here are one-to-one anchored phones. Length-changing
dictionary alternatives need a separate span graph and are not approximated by
normal deletions. All returned substitution/insertion posterior entries are
unconditional event probabilities (not probabilities conditioned on error).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numba import njit


def _weights(
    acceptable: Sequence[Sequence[int]],
    canonical_ids: Sequence[int],
    priors: Mapping[str, Any],
    vocab_size: int,
    max_insertions: int,
    edit_scale: float,
    variant_scale: float,
) -> tuple[np.ndarray, ...]:
    n = len(acceptable)
    if len(canonical_ids) != n or max_insertions < 0:
        raise ValueError("Anchor count mismatch or negative insertion limit")
    op = np.asarray(priors["op_probs"], dtype=np.float64)
    sub = np.asarray(priors["sub_probs"], dtype=np.float64)
    iph = np.asarray(priors["insert_phone_probs"], dtype=np.float64).copy()
    ip = np.broadcast_to(np.asarray(priors["insertion_prob"], dtype=np.float64), (n + 1,))
    if op.shape != (vocab_size, 3) or sub.shape != (vocab_size, vocab_size):
        raise ValueError("Expected op_probs[V,3] and sub_probs[V,V]")
    if iph.shape != (vocab_size,) or not np.all(np.isfinite(iph)):
        raise ValueError("Expected finite insert_phone_probs[V]")
    if not (np.all(np.isfinite(op)) and np.all(np.isfinite(sub)) and
            np.all(np.isfinite(ip)) and np.all((ip >= 0) & (ip <= 1))):
        raise ValueError("Invalid prior probabilities")
    if np.any(op[1:] < 0) or not np.allclose(op[1:].sum(axis=1), 1):
        raise ValueError("Operation probabilities must be nonnegative and normalized")
    if np.any(iph[1:] < 0) or iph[1:].sum() <= 0 or np.any(sub < 0):
        raise ValueError("Invalid phone probabilities")
    if edit_scale <= 0 or variant_scale <= 0:
        raise ValueError("Score scales must be positive")
    iph[0] = 0
    iph /= iph.sum()
    allowed = np.zeros((n, vocab_size), dtype=np.bool_)
    emit = np.zeros((n, vocab_size), dtype=np.float64)
    deletion = np.empty(n, dtype=np.float64)
    for i, (choices, canonical) in enumerate(zip(acceptable, canonical_ids)):
        choices = sorted(set(int(q) for q in choices))
        if not choices or min(choices) <= 0 or max(choices) >= vocab_size:
            raise ValueError("Acceptable realizations must be nonblank vocabulary IDs")
        if canonical not in choices:
            raise ValueError("The canonical realization must remain acceptable")
        allowed[i, choices] = True
        m = len(choices)
        # Sum the latent uniform variant prior for edit events shared by all
        # variants. At variant_scale=1 this factor is exactly one.
        variant_mass = m ** (1.0 - variant_scale)
        emit[i, choices] = op[canonical, 0] ** edit_scale * m ** (-variant_scale)
        row = sub[canonical].copy()
        row[0] = 0
        row[choices] = 0
        if np.any(~allowed[i, 1:]) and row.sum() <= 0:
            raise ValueError("Substitutions outside the acceptable set need prior support")
        if row.sum() > 0:
            row /= row.sum()
            emit[i] += variant_mass * op[canonical, 1] ** edit_scale * row ** edit_scale
        deletion[i] = variant_mass * op[canonical, 2] ** edit_scale
    states = (n + 1) * (max_insertions + 1)
    stop = np.empty(states, dtype=np.float64)
    ins = np.zeros((states, vocab_size), dtype=np.float64)
    dele = np.zeros(states, dtype=np.float64)
    for q in range(states):
        i, k = divmod(q, max_insertions + 1)
        stop[q] = (1 - ip[i]) ** edit_scale if k < max_insertions else 1.0
        if k < max_insertions:
            ins[q] = (ip[i] * iph) ** edit_scale
        if i < n:
            dele[q] = stop[q] * deletion[i]
    return allowed, emit, stop, ins, dele


@njit(cache=True, inline="always")
def _logadd(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    high, low = (a, b) if a >= b else (b, a)
    return high + np.log1p(np.exp(low - high))


@njit(cache=True)
def _logtotal_residual(row):
    largest = 0
    for a in range(1, row.size):
        if row[a] > row[largest]:
            largest = a
    second = -np.inf
    for a in range(row.size):
        if a != largest:
            second = max(second, row[a])
    if second == -np.inf:
        return row[largest], largest, -np.inf
    residual_sum = 0.0
    for a in range(row.size):
        if a != largest:
            residual_sum += np.exp(row[a] - second)
    residual = second + np.log(residual_sum)
    return _logadd(row[largest], residual), largest, residual


@njit(cache=True, inline="always")
def _logexclude(total, largest, residual, row, a):
    if a == largest:
        return residual
    if row[a] == -np.inf:
        return total
    return total + np.log1p(-np.exp(row[a] - total))


@njit(cache=True)
def _sum_inference(log_acoustic, emit, stop, ins, dele, allowed, kmax):
    tmax, vocab = log_acoustic.shape
    n = emit.shape[0]
    stride = kmax + 1
    states = (n + 1) * stride
    logemit, logstop, logins, logdele = np.log(emit), np.log(stop), np.log(ins), np.log(dele)
    forward = np.full((tmax + 1, states, vocab), -np.inf)
    forward[0, 0, 0] = 0.0
    # aclose[q,b] excludes previous frame label b, then applies deletions.
    aclose = np.full((states, vocab), -np.inf)
    for t in range(tmax):
        for q in range(states):
            total, largest, residual = _logtotal_residual(forward[t, q])
            forward[t + 1, q, 0] = total + log_acoustic[t, 0]
            for a in range(1, vocab):
                forward[t + 1, q, a] = forward[t, q, a] + log_acoustic[t, a]
                aclose[q, a] = _logexclude(total, largest, residual, forward[t, q], a)
        for q in range(states):
            i = q // stride
            nxt = (i + 1) * stride
            for a in range(1, vocab):
                mass = aclose[q, a]
                if i < n:
                    val = mass + logstop[q] + logemit[i, a] + log_acoustic[t, a]
                    forward[t + 1, nxt, a] = _logadd(forward[t + 1, nxt, a], val)
                    aclose[nxt, a] = _logadd(aclose[nxt, a], mass + logdele[q])
                if q % stride < kmax:
                    val = mass + logins[q, a] + log_acoustic[t, a]
                    forward[t + 1, q + 1, a] = _logadd(forward[t + 1, q + 1, a], val)
    # Only complete paths accept, including deletion of any remaining anchors.
    remaining = np.full(states, -np.inf)
    for q in range(states - 1, -1, -1):
        i = q // stride
        remaining[q] = logstop[q] if i == n else logdele[q] + remaining[(i + 1) * stride]
    logz = -np.inf
    terminal_prefix = np.full(states, -np.inf)
    for q in range(states):
        terminal_prefix[q] = _logtotal_residual(forward[tmax, q])[0]
        logz = _logadd(logz, terminal_prefix[q] + remaining[q])
    if not np.isfinite(logz):
        raise ValueError("No accepting graph/CTC path")
    token = np.zeros((n, 3))
    substitution = np.zeros((n, vocab))
    insertion = np.zeros((n + 1, kmax, vocab))
    for q in range(states):
        i = q // stride
        if i < n:
            nxt = (i + 1) * stride
            token[i, 2] += np.exp(terminal_prefix[q] + logdele[q] + remaining[nxt] - logz)
            terminal_prefix[nxt] = _logadd(terminal_prefix[nxt], terminal_prefix[q] + logdele[q])
    backward = np.empty((states, vocab))
    for q in range(states):
        backward[q, :] = remaining[q]
    suffix = np.full((states, vocab), -np.inf)
    previous = np.empty_like(backward)
    for t in range(tmax - 1, -1, -1):
        for q in range(states - 1, -1, -1):
            i = q // stride
            nxt = (i + 1) * stride
            for a in range(1, vocab):
                val = -np.inf
                if i < n:
                    val = logstop[q] + logemit[i, a] + log_acoustic[t, a] + backward[nxt, a]
                    val = _logadd(val, logdele[q] + suffix[nxt, a])
                if q % stride < kmax:
                    val = _logadd(val, logins[q, a] + log_acoustic[t, a] + backward[q + 1, a])
                suffix[q, a] = val
        for q in range(states):
            total, largest, residual = _logtotal_residual(forward[t, q])
            for a in range(1, vocab):
                aclose[q, a] = _logexclude(total, largest, residual, forward[t, q], a)
        for q in range(states):
            i = q // stride
            nxt = (i + 1) * stride
            for a in range(1, vocab):
                mass = aclose[q, a] - logz
                if i < n:
                    event = np.exp(mass + logstop[q] + logemit[i, a] + log_acoustic[t, a] + backward[nxt, a])
                    if allowed[i, a]:
                        token[i, 0] += event
                    else:
                        token[i, 1] += event
                        substitution[i, a] += event
                    token[i, 2] += np.exp(mass + logdele[q] + suffix[nxt, a])
                    aclose[nxt, a] = _logadd(aclose[nxt, a], aclose[q, a] + logdele[q])
                if q % stride < kmax:
                    insertion[i, q % stride, a] += np.exp(mass + logins[q, a] + log_acoustic[t, a] + backward[q + 1, a])
        for q in range(states):
            all_new, largest, residual = _logtotal_residual(suffix[q])
            blank = log_acoustic[t, 0] + backward[q, 0]
            previous[q, 0] = _logadd(blank, all_new)
            for a in range(1, vocab):
                other = _logexclude(all_new, largest, residual, suffix[q], a)
                previous[q, a] = _logadd(_logadd(blank, log_acoustic[t, a] + backward[q, a]), other)
        backward, previous = previous, backward
    return logz, token, substitution, insertion


@njit(cache=True)
def _max_event_scores(history, log_acoustic, logemit, logstop, logins, logdele, remaining, allowed, kmax):
    tmax, vocab = log_acoustic.shape
    n = logemit.shape[0]
    stride = kmax + 1
    states = (n + 1) * stride
    event_scores = np.full((n, 3), -np.inf)
    endprefix = np.full(states, -np.inf)
    for q in range(states):
        endprefix[q] = history[tmax, q].max()
    for q in range(states):
        i = q // stride
        if i < n:
            nxt = (i + 1) * stride
            event_scores[i, 2] = max(event_scores[i, 2], endprefix[q] + logdele[q] + remaining[nxt])
            endprefix[nxt] = max(endprefix[nxt], endprefix[q] + logdele[q])
    backward = np.empty((states, vocab))
    for q in range(states):
        backward[q, :] = remaining[q]
    previous = np.empty_like(backward)
    suffix = np.full((states, vocab), -np.inf)
    close = np.full((states, vocab), -np.inf)
    for t in range(tmax - 1, -1, -1):
        for q in range(states - 1, -1, -1):
            i = q // stride
            nxt = (i + 1) * stride
            for a in range(1, vocab):
                val = -np.inf
                if i < n:
                    val = logstop[q] + logemit[i, a] + log_acoustic[t, a] + backward[nxt, a]
                    val = max(val, logdele[q] + suffix[nxt, a])
                if q % stride < kmax:
                    val = max(val, logins[q, a] + log_acoustic[t, a] + backward[q + 1, a])
                suffix[q, a] = val
        for q in range(states):
            best, second, bestid = -np.inf, -np.inf, -1
            for a in range(vocab):
                val = history[t, q, a]
                if val > best:
                    second, best, bestid = best, val, a
                elif val > second:
                    second = val
            for a in range(1, vocab):
                close[q, a] = best if bestid != a else second
        for q in range(states):
            i = q // stride
            if i < n:
                nxt = (i + 1) * stride
                for a in range(1, vocab):
                    event = 0 if allowed[i, a] else 1
                    score = close[q, a] + logstop[q] + logemit[i, a] + log_acoustic[t, a] + backward[nxt, a]
                    event_scores[i, event] = max(event_scores[i, event], score)
                    score = close[q, a] + logdele[q] + suffix[nxt, a]
                    event_scores[i, 2] = max(event_scores[i, 2], score)
                    close[nxt, a] = max(close[nxt, a], close[q, a] + logdele[q])
        for q in range(states):
            best, second, bestid = -np.inf, -np.inf, -1
            for a in range(1, vocab):
                val = suffix[q, a]
                if val > best:
                    second, best, bestid = best, val, a
                elif val > second:
                    second = val
            blank = log_acoustic[t, 0] + backward[q, 0]
            previous[q, 0] = max(blank, best)
            for a in range(1, vocab):
                other = best if a != bestid else second
                previous[q, a] = max(blank, log_acoustic[t, a] + backward[q, a], other)
        backward, previous = previous, backward
    return event_scores


@njit(cache=True)
def _max_inference(log_acoustic, emit, stop, ins, dele, allowed, kmax):
    tmax, vocab = log_acoustic.shape
    n = emit.shape[0]
    stride = kmax + 1
    states = (n + 1) * stride
    logemit = np.log(emit)
    logstop = np.log(stop)
    logins = np.log(ins)
    logdele = np.log(dele)
    value = np.full((states, vocab), -np.inf)
    value[0, 0] = 0.0
    history = np.full((tmax + 1, states, vocab), -np.inf)
    history[0] = value
    trace = np.full((tmax, states, vocab), -1, dtype=np.int32)
    arc_source = np.full((tmax, states, vocab), -1, dtype=np.int32)
    close = np.full((states, vocab), -np.inf)
    origin = np.full((states, vocab), -1, dtype=np.int32)
    for t in range(tmax):
        nextvalue = np.full((states, vocab), -np.inf)
        for q in range(states):
            best = -np.inf
            second = -np.inf
            bestid = -1
            secondid = -1
            for a in range(vocab):
                val = value[q, a]
                if val > best:
                    second, secondid = best, bestid
                    best, bestid = val, a
                elif val > second:
                    second, secondid = val, a
            nextvalue[q, 0] = best + log_acoustic[t, 0]
            if bestid >= 0:
                trace[t, q, 0] = q * vocab + bestid
            for a in range(1, vocab):
                nextvalue[q, a] = value[q, a] + log_acoustic[t, a]
                trace[t, q, a] = q * vocab + a
                close[q, a] = best if bestid != a else second
                idx = bestid if bestid != a else secondid
                origin[q, a] = q * vocab + idx if idx >= 0 else -1
        for q in range(states):
            i = q // stride
            nxt = (i + 1) * stride
            for a in range(1, vocab):
                mass = close[q, a]
                if i < n:
                    score = mass + logstop[q] + logemit[i, a] + log_acoustic[t, a]
                    if score > nextvalue[nxt, a]:
                        nextvalue[nxt, a] = score
                        trace[t, nxt, a] = origin[q, a]
                        arc_source[t, nxt, a] = q
                    score = mass + logdele[q]
                    if score > close[nxt, a]:
                        close[nxt, a] = score
                        origin[nxt, a] = origin[q, a]
                if q % stride < kmax:
                    score = mass + logins[q, a] + log_acoustic[t, a]
                    if score > nextvalue[q + 1, a]:
                        nextvalue[q + 1, a] = score
                        trace[t, q + 1, a] = origin[q, a]
                        arc_source[t, q + 1, a] = q
        value = nextvalue
        history[t + 1] = value
    remaining = np.full(states, -np.inf)
    for q in range(states - 1, -1, -1):
        i = q // stride
        remaining[q] = logstop[q] if i == n else logdele[q] + remaining[(i + 1) * stride]
    best = -np.inf
    state = -1
    for q in range(states):
        for a in range(vocab):
            score = value[q, a] + remaining[q]
            if score > best:
                best = score
                state = q * vocab + a
    if state < 0:
        raise ValueError("No accepting Viterbi path")
    events = np.full(n, 2, dtype=np.int64)
    phones = np.full(n, -1, dtype=np.int64)
    inserted = np.full((n + 1, kmax), -1, dtype=np.int64)
    frame_labels = np.empty(tmax, dtype=np.int64)
    for t in range(tmax - 1, -1, -1):
        q, a = state // vocab, state % vocab
        frame_labels[t] = a
        source = arc_source[t, q, a]
        if source >= 0:
            i = source // stride
            if q == source + 1 and source % stride < kmax:
                inserted[i, source % stride] = a
            else:
                events[i] = 0 if allowed[i, a] else 1
                phones[i] = a
        state = trace[t, q, a]
    maxevents = _max_event_scores(
        history, log_acoustic, logemit, logstop, logins, logdele, remaining, allowed, kmax,
    )
    return best, events, phones, inserted, frame_labels, maxevents


def _compile_surface_exclusion(allowed, emit, stop, ins, dele, kmax):
    """Compose edit graph with complement of the accepted surface language.

    The generic component records how many output phones still match G_acc;
    divergence is absorbing. It rejects every complete accepted surface string.
    A separate clean component reinstates that string with its all-OK weight.
    Consequently DEL+INS and other redundant edit histories can never turn a
    fully acceptable pronunciation into an error explanation.
    """
    n, vocab = allowed.shape
    stride = kmax + 1
    qcount = (n + 1) * stride
    # A matched prefix contains exactly k emitted phones. At edit position
    # (i,j), j <= k <= i*(K+1)+j; unreachable combinations need no states.
    coordinates = []
    for q in range(qcount):
        i, j = divmod(q, stride)
        max_emitted = i * stride + j
        coordinates.extend((q, k) for k in range(j, min(n, max_emitted) + 1))
        if max_emitted > 0:
            coordinates.append((q, n + 1))  # diverged, absorbing
    node_index = {pair: node for node, pair in enumerate(coordinates)}
    clean_start = len(coordinates)
    nodes = clean_start + n + 1
    offsets = [0]
    labels = []
    prev_index = np.full((nodes, vocab), -1, dtype=np.int64)
    for node in range(nodes):
        k = coordinates[node][1] if node < clean_start else node - clean_start
        previous = range(vocab) if node < clean_start and k == n + 1 else (
            [0] if k == 0 else [0, *np.flatnonzero(allowed[k - 1]).tolist()]
        )
        for a in previous:
            prev_index[node, a] = len(labels)
            labels.append(a)
        offsets.append(len(labels))
    tok_dest = np.full((nodes, vocab), -1, dtype=np.int64)
    ins_dest = np.full_like(tok_dest, -1)
    logtok = np.full((nodes, vocab), -np.inf)
    logins = np.full_like(logtok, -np.inf)
    del_node = np.full(nodes, -1, dtype=np.int64)
    logdel = np.full(nodes, -np.inf)
    final = np.full(nodes, -np.inf)
    anchor = np.full(nodes, -1, dtype=np.int64)
    gap = np.empty(nodes, dtype=np.int64)
    slot = np.empty(nodes, dtype=np.int64)
    with np.errstate(divide="ignore"):
        for node in range(clean_start):
            q, k = coordinates[node]
            i, j = divmod(q, stride)
            gap[node], slot[node] = i, j
            if i < n:
                anchor[node] = i
                del_node[node] = node_index[((i + 1) * stride, k)]
                logdel[node] = np.log(dele[q])
            elif k != n:
                final[node] = np.log(stop[q])
            for a in range(1, vocab):
                after = k + 1 if k < n and allowed[k, a] else n + 1
                if i < n:
                    child = node_index[((i + 1) * stride, after)]
                    tok_dest[node, a] = prev_index[child, a]
                    logtok[node, a] = np.log(stop[q] * emit[i, a])
                if j < kmax:
                    child = node_index[(q + 1, after)]
                    ins_dest[node, a] = prev_index[child, a]
                    logins[node, a] = np.log(ins[q, a])
        for i in range(n + 1):
            node = clean_start + i
            gap[node], slot[node] = i, 0
            if i == n:
                final[node] = np.log(stop[i * stride])
            else:
                anchor[node] = i
                for a in np.flatnonzero(allowed[i]):
                    tok_dest[node, a] = prev_index[node + 1, a]
                    logtok[node, a] = np.log(stop[i * stride] * emit[i, a])
    return (
        np.asarray(offsets, dtype=np.int64), np.asarray(labels, dtype=np.int64), prev_index,
        tok_dest, ins_dest, logtok, logins, del_node, logdel, final, anchor, gap, slot,
        np.asarray([prev_index[0, 0], prev_index[clean_start, 0]], dtype=np.int64),
    )


def _candidate_mask(candidates, count, vocab, name):
    """None retains all phones; an empty row removes that error branch."""
    mask = np.ones((count, vocab), dtype=np.bool_)
    mask[:, 0] = False
    if candidates is None:
        return mask
    if len(candidates) != count:
        raise ValueError(f"{name} needs {count} rows")
    for i, row in enumerate(candidates):
        if row is None:
            continue
        ids = np.asarray(row)
        if ids.size and (not np.issubdtype(ids.dtype, np.integer) or
                         np.any(ids <= 0) or np.any(ids >= vocab)):
            raise ValueError(f"{name} must contain nonblank vocabulary IDs")
        mask[i] = False
        mask[i, ids.astype(np.int64)] = True
    return mask


def _compact_graph(graph):
    """Remove zero-weight arcs and graph/CTC states they cannot reach.

    Node order is topological. Epsilon deletions propagate pre-emission
    closures rather than previous-frame labels, so a node only needs previous
    labels supplied by its incoming nonblank arcs (plus its blank state).
    """
    (offsets, labels, prev_index, tok_dest, ins_dest, logtok, logins, del_node,
     logdel, final, anchor, gap, slot, initial) = graph
    nodes, vocab = prev_index.shape
    state_node = np.repeat(np.arange(nodes), np.diff(offsets))
    td = np.where(np.isfinite(logtok), tok_dest, -1)
    ind = np.where(np.isfinite(logins), ins_dest, -1)
    dn = np.where(np.isfinite(logdel), del_node, -1)
    tok_node = np.where(td >= 0, state_node[np.maximum(td, 0)], -1)
    ins_node = np.where(ind >= 0, state_node[np.maximum(ind, 0)], -1)
    reachable = np.zeros(nodes, dtype=np.bool_)
    reachable[state_node[initial]] = True
    for node in range(nodes):
        if reachable[node]:
            reachable[tok_node[node, tok_node[node] >= 0]] = True
            reachable[ins_node[node, ins_node[node] >= 0]] = True
            if dn[node] >= 0:
                reachable[dn[node]] = True
    accepting = np.isfinite(final)
    for node in range(nodes - 1, -1, -1):
        children = np.concatenate((tok_node[node], ins_node[node], [dn[node]]))
        accepting[node] |= np.any(accepting[children[children >= 0]])
    kept = np.flatnonzero(reachable & accepting)
    if not len(kept):
        raise ValueError("No accepting surface-constrained graph/CTC path")
    node_map = np.full(nodes, -1, dtype=np.int64)
    node_map[kept] = np.arange(len(kept))
    previous = np.zeros((len(kept), vocab), dtype=np.bool_)
    previous[:, 0] = True
    for node in kept:
        for destinations in (tok_node, ins_node):
            for a in np.flatnonzero(destinations[node] >= 0):
                child = node_map[destinations[node, a]]
                if child >= 0:
                    previous[child, a] = True
    new_offsets = np.r_[0, np.cumsum(previous.sum(axis=1))].astype(np.int64)
    new_labels = np.nonzero(previous)[1].astype(np.int64)
    new_prev = np.full(previous.shape, -1, dtype=np.int64)
    new_prev[previous] = np.arange(len(new_labels))
    destinations = []
    weights = []
    for target_nodes, logweights in ((tok_node, logtok), (ins_node, logins)):
        target = np.full(previous.shape, -1, dtype=np.int64)
        weight = np.full(previous.shape, -np.inf)
        for new_node, node in enumerate(kept):
            for a in np.flatnonzero(target_nodes[node] >= 0):
                child = node_map[target_nodes[node, a]]
                if child >= 0:
                    target[new_node, a] = new_prev[child, a]
                    weight[new_node, a] = logweights[node, a]
        destinations.append(target)
        weights.append(weight)
    new_del = np.full(len(kept), -1, dtype=np.int64)
    valid_del = dn[kept] >= 0
    new_del[valid_del] = node_map[dn[kept][valid_del]]
    new_logdel = np.where(new_del >= 0, logdel[kept], -np.inf)
    initial_nodes = node_map[state_node[initial]]
    new_initial = new_prev[initial_nodes[initial_nodes >= 0], 0]
    return (
        new_offsets, new_labels, new_prev, *destinations, *weights,
        new_del, new_logdel, final[kept], anchor[kept], gap[kept], slot[kept], new_initial,
    )


@njit(cache=True)
def _graph_forward(lp, offsets, labels, prev_index, tok_dest, ins_dest, logtok, logins,
                   del_node, logdel, initial, maximize):
    tmax, vocab = lp.shape
    nodes, states = len(offsets) - 1, len(labels)
    hist = np.full((tmax + 1, states), -np.inf)
    for s in initial:
        hist[0, s] = 0.0
    trace = np.full((tmax, states), -1, dtype=np.int32) if maximize else np.empty((0, 0), dtype=np.int32)
    source = np.full_like(trace, -1)
    arc = np.zeros_like(trace)
    close = np.full((nodes, vocab), -np.inf)
    origin = np.full((nodes, vocab), -1, dtype=np.int32)
    for t in range(tmax):
        for node in range(nodes):
            lo, hi = offsets[node], offsets[node + 1]
            if maximize:
                best, second, bestid, secondid = -np.inf, -np.inf, -1, -1
                for s in range(lo, hi):
                    val = hist[t, s]
                    if val > best:
                        second, secondid, best, bestid = best, bestid, val, s
                    elif val > second:
                        second, secondid = val, s
                hist[t + 1, lo] = best + lp[t, 0]
                trace[t, lo] = bestid
                for a in range(1, vocab):
                    prev = prev_index[node, a]
                    close[node, a] = best if prev != bestid else second
                    origin[node, a] = bestid if prev != bestid else secondid
            else:
                total, largest, residual = _logtotal_residual(hist[t, lo:hi])
                hist[t + 1, lo] = total + lp[t, 0]
                for a in range(1, vocab):
                    prev = prev_index[node, a]
                    close[node, a] = total if prev < 0 else _logexclude(
                        total, largest, residual, hist[t, lo:hi], prev - lo,
                    )
            for s in range(lo + 1, hi):
                hist[t + 1, s] = hist[t, s] + lp[t, labels[s]]
                if maximize:
                    trace[t, s] = s
        for node in range(nodes):
            dn = del_node[node]
            for a in range(1, vocab):
                mass = close[node, a]
                td, ind = tok_dest[node, a], ins_dest[node, a]
                if td >= 0:
                    val = mass + logtok[node, a] + lp[t, a]
                    if maximize:
                        if val > hist[t + 1, td]:
                            hist[t + 1, td] = val
                            trace[t, td], source[t, td], arc[t, td] = origin[node, a], node, 1
                    else:
                        hist[t + 1, td] = _logadd(hist[t + 1, td], val)
                if ind >= 0:
                    val = mass + logins[node, a] + lp[t, a]
                    if maximize:
                        if val > hist[t + 1, ind]:
                            hist[t + 1, ind] = val
                            trace[t, ind], source[t, ind], arc[t, ind] = origin[node, a], node, 2
                    else:
                        hist[t + 1, ind] = _logadd(hist[t + 1, ind], val)
                if dn >= 0:
                    val = mass + logdel[node]
                    if maximize:
                        if val > close[dn, a]:
                            close[dn, a], origin[dn, a] = val, origin[node, a]
                    else:
                        close[dn, a] = _logadd(close[dn, a], val)
    return hist, trace, source, arc


@njit(cache=True)
def _graph_backward(lp, hist, offsets, labels, prev_index, tok_dest, ins_dest, logtok, logins,
                    del_node, logdel, final, anchor, gap, slot, allowed, kmax, maximize):
    tmax, vocab = lp.shape
    nodes, states, n = len(offsets) - 1, len(labels), allowed.shape[0]
    remaining = final.copy()
    for node in range(nodes - 1, -1, -1):
        dn = del_node[node]
        if dn >= 0:
            candidate = logdel[node] + remaining[dn]
            remaining[node] = max(remaining[node], candidate) if maximize else _logadd(remaining[node], candidate)
    total_score, last = -np.inf, -1
    endprefix = np.full(nodes, -np.inf)
    backward = np.full(states, -np.inf)
    for node in range(nodes):
        lo, hi = offsets[node], offsets[node + 1]
        for s in range(lo, hi):
            val = hist[tmax, s] + remaining[node]
            backward[s] = remaining[node]
            if maximize:
                if val > total_score:
                    total_score, last = val, s
                endprefix[node] = max(endprefix[node], hist[tmax, s])
            else:
                total_score = _logadd(total_score, val)
                endprefix[node] = _logadd(endprefix[node], hist[tmax, s])
    if not np.isfinite(total_score):
        raise ValueError("No accepting surface-constrained graph/CTC path")
    token = np.full((n, 3), -np.inf)
    substitution = np.full((n, vocab), -np.inf)
    insertion = np.full((n + 1, kmax, vocab), -np.inf)
    for node in range(nodes):
        dn, i = del_node[node], anchor[node]
        if dn >= 0:
            val = endprefix[node] + logdel[node] + remaining[dn]
            if maximize:
                token[i, 2] = max(token[i, 2], val)
                endprefix[dn] = max(endprefix[dn], endprefix[node] + logdel[node])
            else:
                token[i, 2] = _logadd(token[i, 2], val)
                endprefix[dn] = _logadd(endprefix[dn], endprefix[node] + logdel[node])
    suffix = np.full((nodes, vocab), -np.inf)
    close = np.full_like(suffix, -np.inf)
    previous = np.full(states, -np.inf)
    for t in range(tmax - 1, -1, -1):
        for node in range(nodes - 1, -1, -1):
            dn = del_node[node]
            for a in range(1, vocab):
                val = -np.inf
                td, ind = tok_dest[node, a], ins_dest[node, a]
                if td >= 0:
                    val = logtok[node, a] + lp[t, a] + backward[td]
                if ind >= 0:
                    other = logins[node, a] + lp[t, a] + backward[ind]
                    val = max(val, other) if maximize else _logadd(val, other)
                if dn >= 0:
                    other = logdel[node] + suffix[dn, a]
                    val = max(val, other) if maximize else _logadd(val, other)
                suffix[node, a] = val
        for node in range(nodes):
            lo, hi = offsets[node], offsets[node + 1]
            if maximize:
                best, second, bestid = -np.inf, -np.inf, -1
                for s in range(lo, hi):
                    val = hist[t, s]
                    if val > best:
                        second, best, bestid = best, val, s
                    elif val > second:
                        second = val
                for a in range(1, vocab):
                    close[node, a] = best if prev_index[node, a] != bestid else second
            else:
                total, largest, residual = _logtotal_residual(hist[t, lo:hi])
                for a in range(1, vocab):
                    prev = prev_index[node, a]
                    close[node, a] = total if prev < 0 else _logexclude(total, largest, residual, hist[t, lo:hi], prev - lo)
        for node in range(nodes):
            dn, i = del_node[node], anchor[node]
            for a in range(1, vocab):
                mass = close[node, a]
                td, ind = tok_dest[node, a], ins_dest[node, a]
                if td >= 0:
                    val = mass + logtok[node, a] + lp[t, a] + backward[td]
                    event = 0 if allowed[i, a] else 1
                    if maximize:
                        token[i, event] = max(token[i, event], val)
                    else:
                        token[i, event] = _logadd(token[i, event], val)
                        if event == 1:
                            substitution[i, a] = _logadd(substitution[i, a], val)
                if ind >= 0:
                    val = mass + logins[node, a] + lp[t, a] + backward[ind]
                    gi, sj = gap[node], slot[node]
                    insertion[gi, sj, a] = max(insertion[gi, sj, a], val) if maximize else _logadd(insertion[gi, sj, a], val)
                if dn >= 0:
                    val = mass + logdel[node] + suffix[dn, a]
                    if maximize:
                        token[i, 2] = max(token[i, 2], val)
                        close[dn, a] = max(close[dn, a], mass + logdel[node])
                    else:
                        token[i, 2] = _logadd(token[i, 2], val)
                        close[dn, a] = _logadd(close[dn, a], mass + logdel[node])
        for node in range(nodes):
            lo, hi = offsets[node], offsets[node + 1]
            blank = lp[t, 0] + backward[lo]
            if maximize:
                best, second, bestid = -np.inf, -np.inf, -1
                for a in range(1, vocab):
                    val = suffix[node, a]
                    if val > best:
                        second, best, bestid = best, val, a
                    elif val > second:
                        second = val
                previous[lo] = max(blank, best)
                for s in range(lo + 1, hi):
                    a = labels[s]
                    previous[s] = max(blank, lp[t, a] + backward[s], best if a != bestid else second)
            else:
                total, largest, residual = _logtotal_residual(suffix[node])
                previous[lo] = _logadd(blank, total)
                for s in range(lo + 1, hi):
                    a = labels[s]
                    other = _logexclude(total, largest, residual, suffix[node], a)
                    previous[s] = _logadd(_logadd(blank, lp[t, a] + backward[s]), other)
        backward, previous = previous, backward
    return total_score, token, substitution, insertion, last


def _surface_inference(lp, graph, allowed, kmax, maximize):
    (offsets, labels, prev_index, tok_dest, ins_dest, logtok, logins, del_node, logdel,
     final, anchor, gap, slot, initial) = graph
    hist, trace, source, arc = _graph_forward(
        lp, offsets, labels, prev_index, tok_dest, ins_dest, logtok, logins, del_node, logdel, initial, maximize,
    )
    score, token, sub, insertion, last = _graph_backward(
        lp, hist, offsets, labels, prev_index, tok_dest, ins_dest, logtok, logins, del_node, logdel,
        final, anchor, gap, slot, allowed, kmax, maximize,
    )
    if not maximize:
        return score, np.exp(token - score), np.exp(sub - score), np.exp(insertion - score), token
    n = allowed.shape[0]
    events, phones = np.full(n, 2, dtype=np.int64), np.full(n, -1, dtype=np.int64)
    inserted = np.full((n + 1, kmax), -1, dtype=np.int64)
    frames = np.empty(len(lp), dtype=np.int64)
    for t in range(len(lp) - 1, -1, -1):
        a = labels[last]
        frames[t] = a
        node, event = source[t, last], arc[t, last]
        if event == 1:
            i = anchor[node]
            events[i], phones[i] = (0 if allowed[i, a] else 1), a
        elif event == 2:
            inserted[gap[node], slot[node]] = a
        last = trace[t, last]
    return score, events, phones, inserted, frames, token


def decode(
    log_probs: np.ndarray,
    acceptable: Sequence[Sequence[int]],
    priors: Mapping[str, Any],
    *,
    canonical_ids: Sequence[int] | None = None,
    max_insertions: int = 2,
    edit_scale: float = 1.0,
    variant_scale: float = 1.0,
    include_viterbi: bool = True,
    substitution_candidates: Sequence[Sequence[int] | None] | None = None,
    insertion_candidates: Sequence[Sequence[int] | None] | None = None,
) -> dict[str, Any]:
    """Decode fixed frame log probabilities; no PHN, boundaries, or labels used.

    ``priors`` contains normalized ``op_probs[V,3]`` (OK, SUB, DEL),
    ``sub_probs[V,V]``, ``insertion_prob`` (scalar or one per gap), and
    ``insert_phone_probs[V]``. Blank ID is zero. SUB is renormalized after
    removing every acceptable phone, not just the canonical phone. A truncated
    geometric insertion process uses insertion_prob at each slot and is forced
    to stop after max_insertions. At unit score scales the edit graph is a
    normalized distribution; powered research weights are not renormalized.

    Optional candidates give nonblank phone IDs per anchor (SUB) or gap (INS).
    None means unrestricted, including for an individual row; [] removes that
    row's error branch. Acceptable phones and deletion epsilon arcs remain.
    Pruning preserves each surviving arc's original weight, without
    renormalization. Its partition is therefore a subset of the full graph's
    partition; graph/CTC states and arcs are compacted before inference.
    """
    lp = np.ascontiguousarray(log_probs, dtype=np.float64)
    if lp.ndim != 2 or lp.shape[1] < 2 or np.any(np.isnan(lp)) or np.any(np.isposinf(lp)):
        raise ValueError("Expected finite-or-minus-infinity frame log probabilities")
    if canonical_ids is None:
        canonical_ids = [int(a[0]) for a in acceptable]
    allowed, emit, stop, ins, dele = _weights(
        acceptable, canonical_ids, priors, lp.shape[1], max_insertions, edit_scale, variant_scale,
    )
    restricted = substitution_candidates is not None or insertion_candidates is not None
    if restricted:
        sub_mask = _candidate_mask(
            substitution_candidates, len(acceptable), lp.shape[1], "substitution_candidates",
        )
        ins_mask = _candidate_mask(
            insertion_candidates, len(acceptable) + 1, lp.shape[1], "insertion_candidates",
        )
        emit *= sub_mask | allowed
        ins *= np.repeat(ins_mask, max_insertions + 1, axis=0)
    offsets = lp.max(axis=1)
    if not np.all(np.isfinite(offsets)):
        raise ValueError("Every frame needs at least one possible acoustic label")
    graph = _compile_surface_exclusion(allowed, emit, stop, ins, dele, max_insertions)
    if restricted:
        graph = _compact_graph(graph)
    logz, token, sub, insertion, token_mass = _surface_inference(
        lp - offsets[:, None], graph, allowed, max_insertions, False,
    )
    if not np.all(np.isfinite(token)) or not np.allclose(token.sum(axis=1), 1, atol=1e-7):
        raise ArithmeticError("Forward/backward token posteriors do not sum to one")
    # Only suppress accumulated floating-point error, not graph uncertainty.
    token = np.clip(token, 0.0, 1.0)
    result: dict[str, Any] = {
        "log_partition": float(logz + offsets.sum()),
        "token_posteriors": token,
        "substitution_posteriors": sub,
        "insertion_posteriors": insertion,
        "token_log_odds": np.log(token[:, 0] + 1e-12) - np.log(token[:, 1:].sum(axis=1) + 1e-12),
        "graph_state_count": int(len(graph[0]) - 1),
        "ctc_state_count": int(len(graph[1])),
        "graph_arc_count": int(
            np.isfinite(graph[5]).sum() + np.isfinite(graph[6]).sum() + np.isfinite(graph[8]).sum()
        ),
        "graph_array_bytes": int(sum(a.nbytes for a in graph)),
        "sum_forward_bytes": int((len(lp) + 1) * len(graph[1]) * 8),
        "max_history_trace_bytes": int((len(lp) + 1) * len(graph[1]) * 8 +
                                       len(lp) * len(graph[1]) * 12) if include_viterbi else 0,
        "accepted_surface_errors_excluded": True,
    }
    if include_viterbi:
        score, events, phones, inserted, frames, maxevents = _surface_inference(
            lp, graph, allowed, max_insertions, True,
        )
        result.update({
            "viterbi_score": float(score),
            "viterbi_token_events": events,
            "viterbi_token_phones": phones,
            "viterbi_insertions": inserted,
            "viterbi_frame_labels": frames,
            "max_event_log_scores": maxevents,
            "max_marginal_token_log_odds": maxevents[:, 0] - np.maximum(maxevents[:, 1], maxevents[:, 2]),
        })
    return result
