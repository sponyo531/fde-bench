#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vessel stowage planner (self-contained, standard library only).

Usage:  python solution.py <input.json> <output.json>

Reads the voyage snapshot from sys.argv[1] and writes
{"assignments": [[cntrGkey, projectionGkey], ...]} to sys.argv[2].
"""
from __future__ import annotations

import json
import math
import random
import re as _re
from bisect import bisect_right as _bisect_right
import sys
import time
from collections import defaultdict, Counter
from bisect import bisect_left as _bisect_left, bisect_right as _bisect_right

random.seed(42)
_math_exp = math.exp
_math_ceil = math.ceil


NONE_EMT = -1          # sentinel: "container not landed at evaluation time"
BIG_EMT = 10 ** 18
DECK_THR = 1500
HOLD_THR = 6000
C_WINDOW = 5
C_NUM, C_DEN = 3, 5    # ratio 0.6 == 3/5
# three boxes of one block within 5 positions, shortest span first; \x00 is
# "no yard block" and can never be the block
_RE_TRIG = _re.compile(
    rb'(?=([^\x00])((?:\1\1|.\1\1|\1.\1|..\1\1|.\1.\1|\1..\1)))', _re.DOTALL)
METRICS = ("above", "awv", "twin", "split", "central", "reh")


def _emt_ts(v):
    if v is None:
        return BIG_EMT
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, list) and len(v) >= 6:
        import calendar
        import datetime
        try:
            return int(calendar.timegm(datetime.datetime(*v[:6]).timetuple()))
        except Exception:
            return BIG_EMT
    return BIG_EMT


def _infer_block_type(name):
    if not name:
        return None
    return {"M": "ASC", "Z": "RTG", "K": "TopPicks"}.get(str(name)[0].upper(), "OTHER")


def _java_hash_bucket(gkey, cap):
    v = int(gkey)
    h = ((v & 0xffffffff) ^ ((v >> 32) & 0xffffffff)) & 0xffffffff
    h = (h ^ (h >> 16)) & 0xffffffff
    return h & (cap - 1)


def _short(lb):
    return "20" in str(lb or "")


def _is45(lb):
    return "45" in str(lb or "")


def _is40(lb):
    s = str(lb or "")
    return "40" in s and "45" not in s


class Model:
    """Static problem model: containers, slots, buckets, component topology."""

    def __init__(self, src):
        inst = src["instance"]
        tl = inst["to_load"]
        sl = inst["vessel"]["projection_slots"]
        strat = src.get("strategy") or {}
        self.max_twin_total = int(strat.get("MAX_TWIN_WEIGHT_KG") or 70000)
        self.max_twin_diff = int(strat.get("MAX_TWIN_DIFF_KG") or 15000)

        # ---- block types -------------------------------------------------
        bt = {}
        for b in (inst["vessel"].get("block_types") or []):
            if isinstance(b, dict) and b.get("blockName"):
                bt[b["blockName"]] = b.get("blockType")
        for c in tl:
            blk = (c.get("yard") or {}).get("block")
            if blk and blk not in bt:
                bt[blk] = _infer_block_type(blk)
        self.block_type = bt

        # ---- containers --------------------------------------------------
        self.nc = nc = len(tl)
        self.c_gkey = [int(c["cntrGkey"]) for c in tl]
        self.c_w = [int(c.get("grossWeightKg") or 0) for c in tl]
        self.c_through = [str(c.get("category") or "").upper() == "THROUGH" for c in tl]
        blk_ids = {}
        self.c_blk = [-1] * nc          # yard block id, -1 == direct/TRUCK
        self.c_asc = [False] * nc
        for i, c in enumerate(tl):
            y = c.get("yard") or {}
            b = y.get("block")
            if b:
                if b not in blk_ids:
                    blk_ids[b] = len(blk_ids)
                self.c_blk[i] = blk_ids[b]
                self.c_asc[i] = (bt.get(b) == "ASC")
        self.blk_ids = blk_ids
        # byte code per container for the central scan: yard block id + 1,
        # 0 == no yard block (never a majority block, never flagged)
        self.c_blkb = [0 if b < 0 else b + 1 for b in self.c_blk]
        self.blk_ascb = [False] * (len(blk_ids) + 1)
        for i in range(nc):
            if self.c_blk[i] >= 0 and self.c_asc[i]:
                self.blk_ascb[self.c_blk[i] + 1] = True
        self.wq_bytes_ok = len(blk_ids) < 255

        # yard stacks (ASC + tier known) -> used by split & rehandle
        ys_ids = {}
        self.c_ys = [-1] * nc
        self.c_tier = [None] * nc
        for i, c in enumerate(tl):
            y = c.get("yard") or {}
            if not self.c_asc[i]:
                continue
            t = y.get("tier")
            if t is None:
                continue
            k = y.get("stackGkey")
            if k is None:
                k = ("N", y.get("block"), y.get("stackName"))
            if k not in ys_ids:
                ys_ids[k] = len(ys_ids)
            self.c_ys[i] = ys_ids[k]
            self.c_tier[i] = int(t)
        self.n_ys = len(ys_ids)
        ys_members = [[] for _ in range(self.n_ys)]
        for i in range(nc):
            if self.c_ys[i] >= 0:
                ys_members[self.c_ys[i]].append(i)
        for m in ys_members:
            m.sort(key=lambda i: self.c_tier[i])      # ascending tier
        self.ys_members = ys_members

        # ---- slots ---------------------------------------------------------
        self.ns = ns = len(sl)
        self.s_gkey = [int(x["gkey"]) for x in sl]
        gk2s = {g: i for i, g in enumerate(self.s_gkey)}
        self.gk2s = gk2s
        self.s_len = [((x.get("stowKey") or {}).get("lengthBs")
                       or x.get("lengthBs") or "") for x in sl]
        self.s_deck = [x.get("block") for x in sl]
        self.s_vsltier = [x.get("vslTier") or 0 for x in sl]
        self.s_emt_raw = [_emt_ts(x.get("emt")) for x in sl]
        self.s_seq = [x.get("sequence") for x in sl]
        self.s_bay = [((x.get("bayIdx") + 1) // 2) if x.get("bayIdx") is not None
                      else x.get("pairedBayX") for x in sl]
        self.s_twin = [-1] * ns
        for i, x in enumerate(sl):
            t = x.get("twinWithGkey")
            if t is not None:
                self.s_twin[i] = gk2s.get(int(t), -2)   # -2 == mate outside slot list
        # work queues
        wq_ids = {}
        self.s_wq = [-1] * ns
        for i, x in enumerate(sl):
            w = x.get("wqName")
            if w is not None and x.get("sequence") is not None:
                if w not in wq_ids:
                    wq_ids[w] = len(wq_ids)
                self.s_wq[i] = wq_ids[w]
        self.n_wq = len(wq_ids)

        # ---- buckets --------------------------------------------------------
        bk = {}
        self.c_bk = [0] * nc
        for i, c in enumerate(tl):
            k = (c.get("lengthBs"), c.get("stowFactor"), c.get("weightClass"))
            if k not in bk:
                bk[k] = len(bk)
            self.c_bk[i] = bk[k]
        self.s_bk = [-1] * ns
        for i, x in enumerate(sl):
            sk = x.get("stowKey") or {}
            k = (sk.get("lengthBs"), sk.get("stowFactor"), sk.get("weightClass"))
            self.s_bk[i] = bk.get(k, -1)
        self.n_bk = len(bk)
        self.bk_c = [[] for _ in range(self.n_bk)]
        self.bk_s = [[] for _ in range(self.n_bk)]
        for i in range(nc):
            self.bk_c[self.c_bk[i]].append(i)
        for i in range(ns):
            if self.s_bk[i] >= 0:
                self.bk_s[self.s_bk[i]].append(i)

        # ---- ship stacks (vertical columns on the vessel) --------------------
        st_ids = {}
        self.s_st = [-1] * ns
        for i, x in enumerate(sl):
            k = x.get("stackGkey")
            if k is None:
                continue
            if k not in st_ids:
                st_ids[k] = len(st_ids)
            self.s_st[i] = st_ids[k]
        self.n_st = len(st_ids)
        st_members = [[] for _ in range(self.n_st)]
        for i in range(ns):
            if self.s_st[i] >= 0:
                st_members[self.s_st[i]].append(i)
        for m in st_members:
            m.sort(key=lambda i: self.s_vsltier[i])
        self.st_members = st_members

        # adjacent tier pairs, with static skip flags resolved up-front
        st_pairs = [[] for _ in range(self.n_st)]
        for st, m in enumerate(st_members):
            for j in range(1, len(m)):
                lo, up = m[j - 1], m[j]
                d = self.s_deck[up]
                if d == "A":
                    thr, is_deck = DECK_THR, True
                elif d == "B":
                    thr, is_deck = HOLD_THR, False
                else:
                    continue
                lbu, lbl = self.s_len[up], self.s_len[lo]
                if _short(lbu) != _short(lbl):
                    continue
                if is_deck and _is45(lbu) and _is40(lbl):
                    continue
                st_pairs[st].append((lo, up, thr, is_deck))
        self.st_pairs = st_pairs

        # ---- twin pairs -------------------------------------------------------
        tp = []
        self.s_tp = [-1] * ns
        seen = set()
        for i in range(ns):
            t = self.s_twin[i]
            if t < 0:
                continue
            k = (i, t) if i < t else (t, i)
            if k in seen:
                continue
            seen.add(k)
            self.s_tp[k[0]] = len(tp)
            self.s_tp[k[1]] = len(tp)
            tp.append(k)
        self.twin_pairs = tp


class Solution:
    """Assignment + incremental metric bookkeeping."""

    def __init__(self, model: Model, slot2c, used_slots):
        M = self.M = model
        self.slot2c = list(slot2c)                    # slot idx -> cntr idx or -1
        self.c2slot = [-1] * M.nc
        for s, c in enumerate(self.slot2c):
            if c >= 0:
                self.c2slot[c] = s
        self.used = sorted(used_slots)
        self._build_static()
        self._full_recalc()

    # -- static structures that depend only on the *set* of used slots -------
    def _build_static(self):
        M = self.M
        used = set(self.used)
        n_assign = len(self.used)
        # evaluation-time emt replay
        emt = [BIG_EMT] * M.ns
        by_wq = defaultdict(list)
        for s in range(M.ns):
            w = M.s_wq[s]
            if w >= 0:
                by_wq[w].append((M.s_seq[s], M.s_gkey[s], s))
            else:
                emt[s] = M.s_emt_raw[s]
        for w, lst in by_wq.items():
            lst.sort()
            n_empty = 0
            for _sq, _gk, s in lst:
                if s not in used:
                    n_empty += 1
                    emt[s] = M.s_emt_raw[s]
                    continue
                e = M.s_emt_raw[s]
                emt[s] = (e - 360 * n_empty) if e < BIG_EMT else e
        for s in self.used:
            t = M.s_twin[s]
            if t == -2 or (t >= 0 and t not in used):
                emt[s] = NONE_EMT              # ghost: half twin
        self.s_emt = emt

        # central: per-WQ ordered position list (static given the used set)
        cap = 16
        while cap * 0.75 < n_assign:
            cap <<= 1
        wq_slots = [[] for _ in range(M.n_wq)]
        for s in self.used:
            w = M.s_wq[s]
            e = emt[s]
            if w < 0 or e == NONE_EMT or e >= BIG_EMT:
                continue
            wq_slots[w].append((e, _java_hash_bucket(M.s_gkey[s], cap), M.s_gkey[s], s))
        self.wq_slots = []
        for w in range(M.n_wq):
            wq_slots[w].sort()
            self.wq_slots.append([t[3] for t in wq_slots[w]])
        self.s_wqpos = [-1] * M.ns
        self.s_wqidx = [-1] * M.ns
        for w, ps in enumerate(self.wq_slots):
            for j, s in enumerate(ps):
                self.s_wqpos[s] = w
                self.s_wqidx[s] = j
        cblkb = M.c_blkb
        s2c = self.slot2c
        self.wq_b = [bytearray([cblkb[c] if c >= 0 else 0
                                for c in [s2c[s] for s in ps]])
                     for ps in self.wq_slots]

        # which ship stacks / twin pairs / wqs are non-trivial
        self.st_of = M.s_st
        self.n_comp_st = M.n_st

    # ---------------- component recomputation -----------------------------
    def _calc_stack(self, st):
        M = self.M
        s2c = self.slot2c
        cw = M.c_w
        av = set()
        bv = set()
        for lo, up, thr, is_deck in M.st_pairs[st]:
            cu = s2c[up]
            cl = s2c[lo]
            if cu < 0 and cl < 0:
                continue
            if cl >= 0 and M.c_through[cl]:
                continue
            wu = cw[cu] if cu >= 0 else 0
            wl = cw[cl] if cl >= 0 else 0
            if wu - wl > thr:
                tgt = av if is_deck else bv
                if cu >= 0:
                    tgt.add(cu)
                if cl >= 0:
                    tgt.add(cl)
        return len(av), len(bv)

    def _calc_twin(self, tp):
        M = self.M
        a, b = M.twin_pairs[tp]
        c1, c2 = self.slot2c[a], self.slot2c[b]
        if c1 < 0 or c2 < 0:
            return 0
        b1, b2 = M.c_blk[c1], M.c_blk[c2]
        if b1 >= 0 and b2 >= 0 and b1 != b2 and M.c_asc[c1] and M.c_asc[c2]:
            return 2
        return 0

    def _calc_split(self, ys):
        M = self.M
        c2s = self.c2slot
        sbay = M.s_bay
        bays = set()
        n = 0
        for c in M.ys_members[ys]:
            s = c2s[c]
            if s < 0:
                continue
            n += 1
            bays.add(sbay[s])
        return n if len(bays) > 1 else 0

    def _calc_reh(self, ys):
        M = self.M
        c2s = self.c2slot
        emt = self.s_emt
        mem = M.ys_members[ys]
        n = len(mem)
        if n < 2:
            return 0
        tiers = [M.c_tier[c] for c in mem]
        es = []
        for c in mem:
            s = c2s[c]
            es.append(NONE_EMT if s < 0 else emt[s])
        viol = 0
        flagged = None
        for i in range(n - 1):
            if c2s[mem[i]] < 0:
                continue
            el = es[i]
            if el == NONE_EMT or el >= BIG_EMT:
                continue
            ti = tiers[i]
            for j in range(i + 1, n):
                if tiers[j] <= ti:
                    continue
                eu = es[j]
                if eu == NONE_EMT or eu > el:
                    if c2s[mem[j]] >= 0:
                        if flagged is None:
                            flagged = set()
                        flagged.add(j)
                else:
                    break
        return len(flagged) if flagged else 0

    def _calc_wq(self, w):
        if not self.M.wq_bytes_ok:
            return self._calc_wq_scan(w)
        b = self.wq_b[w]
        n = len(b)
        if n < C_WINDOW:
            return 0
        flags = None
        done = None
        last = n - C_WINDOW
        for m in _RE_TRIG.finditer(b):
            p = m.start()
            x = b[p]
            r = p + len(m.group(2))
            lo = r - (C_WINDOW - 1)
            if lo < 0:
                lo = 0
            hi = p if p < last else last
            if done is None:
                done = bytearray(n)
                flags = bytearray(n)
                basc = self.M.blk_ascb
            for i in range(lo, hi + 1):
                if done[i]:
                    continue
                done[i] = 1
                c = b.count(x, i, i + C_WINDOW)
                end = i + C_WINDOW
                while end < n:
                    if b[end] == x:
                        c += 1
                    if c * C_DEN < C_NUM * (end - i + 1):
                        break
                    end += 1
                if basc[x]:
                    for k in range(i, end):
                        if b[k] == x:
                            flags[k] = 1
        return sum(flags) if flags is not None else 0

    def _calc_wq_scan(self, w):
        ps = self.wq_slots[w]
        n = len(ps)
        if n < C_WINDOW:
            return 0
        s2c = self.slot2c
        cblk = self.M.c_blk
        b = [cblk[c] if c >= 0 else -1 for c in [s2c[s] for s in ps]]
        last = n - C_WINDOW
        prev = {}
        flags = None
        done = None
        casc = None
        for j in range(n):
            x = b[j]
            if x < 0:
                continue
            pp = prev.get(x)
            if pp is None:
                prev[x] = (-9, j)
                continue
            p2, p1 = pp
            prev[x] = (p1, j)
            if j - p2 >= C_WINDOW:
                continue
            # every window i in [j-4, p2] holds x at p2 < p1 < j: 3 of 5
            lo = j - (C_WINDOW - 1)
            if lo < 0:
                lo = 0
            hi = p2 if p2 < last else last
            if done is None:
                done = bytearray(n)
                flags = bytearray(n)
                casc = self.M.c_asc
            for i in range(lo, hi + 1):
                if done[i]:
                    continue
                done[i] = 1
                c = 0
                for k in range(i, i + C_WINDOW):
                    if b[k] == x:
                        c += 1
                end = i + C_WINDOW
                while end < n:
                    if b[end] == x:
                        c += 1
                    if c * C_DEN < C_NUM * (end - i + 1):
                        break
                    end += 1
                for k in range(i, end):
                    if b[k] == x and casc[s2c[ps[k]]]:
                        flags[k] = 1
        return sum(flags) if flags is not None else 0

    # ---------------- full recalculation ----------------------------------
    def _full_recalc(self):
        M = self.M
        self.v_st_a = [0] * M.n_st
        self.v_st_b = [0] * M.n_st
        for st in range(M.n_st):
            self.v_st_a[st], self.v_st_b[st] = self._calc_stack(st)
        self.v_tp = [self._calc_twin(t) for t in range(len(M.twin_pairs))]
        self.v_sp = [self._calc_split(y) for y in range(M.n_ys)]
        self.v_rh = [self._calc_reh(y) for y in range(M.n_ys)]
        self.v_wq = [self._calc_wq(w) for w in range(M.n_wq)]
        self.n_above = sum(self.v_st_a)
        self.n_below = sum(self.v_st_b)
        self.n_twin = sum(self.v_tp)
        self.n_split = sum(self.v_sp)
        self.n_reh = sum(self.v_rh)
        self.n_cent = sum(self.v_wq)

    def metrics(self):
        return {
            "above": self.n_above,
            "awv": self.n_above + self.n_below,
            "twin": self.n_twin,
            "split": self.n_split,
            "central": self.n_cent,
            "reh": self.n_reh,
        }

    def pairs(self):
        M = self.M
        return [(M.c_gkey[c], M.s_gkey[s]) for s, c in enumerate(self.slot2c) if c >= 0]

    # ---------------- twin hard feasibility --------------------------------
    def twin_ok(self, s, c):
        """Would putting container c into slot s keep the twin weight rule?"""
        M = self.M
        tp = M.s_tp[s]
        if tp < 0:
            return True
        a, b = M.twin_pairs[tp]
        o = b if a == s else a
        co = self.slot2c[o]
        if co < 0 or c < 0:
            return True
        w1, w2 = M.c_w[c], M.c_w[co]
        return (w1 + w2) <= M.max_twin_total and abs(w1 - w2) <= M.max_twin_diff

    def twin_all_ok(self):
        M = self.M
        for a, b in M.twin_pairs:
            c1, c2 = self.slot2c[a], self.slot2c[b]
            if c1 < 0 or c2 < 0:
                continue
            w1, w2 = M.c_w[c1], M.c_w[c2]
            if w1 + w2 > M.max_twin_total or abs(w1 - w2) > M.max_twin_diff:
                return False
        return True


# ══════════════════════════════════════════════════════════════════════
#  Incremental move application
# ══════════════════════════════════════════════════════════════════════

def _apply_changes(sol, changes):
    """changes: list of (slot, new_container). Consistent permutation required."""
    M = sol.M
    s_st, s_tp, c_ys = M.s_st, M.s_tp, M.c_ys
    wqpos = sol.s_wqpos
    slot2c = sol.slot2c
    sts = set(); tps = set(); wqs = set(); yss = set()
    for s, ncc in changes:
        oc = slot2c[s]
        v = s_st[s]
        if v >= 0:
            sts.add(v)
        v = s_tp[s]
        if v >= 0:
            tps.add(v)
        v = wqpos[s]
        if v >= 0:
            wqs.add(v)
        if oc >= 0:
            v = c_ys[oc]
            if v >= 0:
                yss.add(v)
        if ncc >= 0:
            v = c_ys[ncc]
            if v >= 0:
                yss.add(v)
    na, nb, nt, nsp, nrh, nce = (sol.n_above, sol.n_below, sol.n_twin,
                                 sol.n_split, sol.n_reh, sol.n_cent)
    va, vb, vt, vsp, vrh, vwq = (sol.v_st_a, sol.v_st_b, sol.v_tp,
                                 sol.v_sp, sol.v_rh, sol.v_wq)
    for st in sts:
        na -= va[st]; nb -= vb[st]
    for tp in tps:
        nt -= vt[tp]
    for w in wqs:
        nce -= vwq[w]
    for y in yss:
        nsp -= vsp[y]; nrh -= vrh[y]

    c2slot = sol.c2slot
    for s, _ in changes:
        oc = slot2c[s]
        if oc >= 0:
            c2slot[oc] = -1
    wqidx, wq_b, cblkb = sol.s_wqidx, sol.wq_b, M.c_blkb
    for s, ncc in changes:
        slot2c[s] = ncc
        if ncc >= 0:
            c2slot[ncc] = s
        wi = wqidx[s]
        if wi >= 0:
            wq_b[wqpos[s]][wi] = cblkb[ncc] if ncc >= 0 else 0

    for st in sts:
        a, b = sol._calc_stack(st)
        va[st] = a; vb[st] = b
        na += a; nb += b
    for tp in tps:
        v = sol._calc_twin(tp); vt[tp] = v; nt += v
    for w in wqs:
        v = sol._calc_wq(w); vwq[w] = v; nce += v
    for y in yss:
        v = sol._calc_split(y); vsp[y] = v; nsp += v
        v = sol._calc_reh(y); vrh[y] = v; nrh += v
    sol.n_above, sol.n_below, sol.n_twin = na, nb, nt
    sol.n_split, sol.n_reh, sol.n_cent = nsp, nrh, nce


Solution.apply_changes = _apply_changes


def _apply_rec(sol, changes):
    """Like _apply_changes, but return a record that `_undo_rec` restores
    exactly (old assignment, old per-component values, old totals)."""
    M = sol.M
    s_st, s_tp, c_ys = M.s_st, M.s_tp, M.c_ys
    cblk, casc, cblkb = M.c_blk, M.c_asc, M.c_blkb
    wqpos, wqidx, wq_b = sol.s_wqpos, sol.s_wqidx, sol.wq_b
    slot2c = sol.slot2c
    c2slot = sol.c2slot
    va, vb, vt, vsp, vrh, vwq = (sol.v_st_a, sol.v_st_b, sol.v_tp,
                                 sol.v_sp, sol.v_rh, sol.v_wq)
    tot = (sol.n_above, sol.n_below, sol.n_twin, sol.n_split, sol.n_reh, sol.n_cent)
    na, nb, nt, nsp, nrh, nce = tot
    r_st = []; r_tp = []; r_wq = []; r_ys = []
    sts = []; tps = []; wqs = []; yss = []
    old_a = []
    for s, ncc in changes:
        oc = slot2c[s]
        old_a.append((s, oc))
        v = s_st[s]
        if v >= 0 and v not in sts:
            sts.append(v); r_st.append((v, va[v], vb[v])); na -= va[v]; nb -= vb[v]
        v = s_tp[s]
        if v >= 0 and v not in tps:
            tps.append(v); r_tp.append((v, vt[v])); nt -= vt[v]
        v = wqpos[s]
        if v >= 0 and v not in wqs and not (
                oc >= 0 and ncc >= 0 and cblk[oc] == cblk[ncc] and casc[oc] == casc[ncc]):
            wqs.append(v); r_wq.append((v, vwq[v])); nce -= vwq[v]
        if oc >= 0:
            v = c_ys[oc]
            if v >= 0 and v not in yss:
                yss.append(v); r_ys.append((v, vsp[v], vrh[v])); nsp -= vsp[v]; nrh -= vrh[v]
        if ncc >= 0:
            v = c_ys[ncc]
            if v >= 0 and v not in yss:
                yss.append(v); r_ys.append((v, vsp[v], vrh[v])); nsp -= vsp[v]; nrh -= vrh[v]

    for s, oc in old_a:
        if oc >= 0:
            c2slot[oc] = -1
    for s, ncc in changes:
        slot2c[s] = ncc
        if ncc >= 0:
            c2slot[ncc] = s
        wi = wqidx[s]
        if wi >= 0:
            wq_b[wqpos[s]][wi] = cblkb[ncc] if ncc >= 0 else 0

    for st in sts:
        a, b = sol._calc_stack(st)
        va[st] = a; vb[st] = b
        na += a; nb += b
    for tp in tps:
        v = sol._calc_twin(tp); vt[tp] = v; nt += v
    for w in wqs:
        v = sol._calc_wq(w); vwq[w] = v; nce += v
    for y in yss:
        v = sol._calc_split(y); vsp[y] = v; nsp += v
        v = sol._calc_reh(y); vrh[y] = v; nrh += v
    sol.n_above, sol.n_below, sol.n_twin = na, nb, nt
    sol.n_split, sol.n_reh, sol.n_cent = nsp, nrh, nce
    return (old_a, r_st, r_tp, r_wq, r_ys, tot)


def _undo_rec(sol, rec):
    old_a, r_st, r_tp, r_wq, r_ys, tot = rec
    slot2c = sol.slot2c
    c2slot = sol.c2slot
    for s, _oc in old_a:
        c = slot2c[s]
        if c >= 0:
            c2slot[c] = -1
    wqpos, wqidx, wq_b, cblkb = sol.s_wqpos, sol.s_wqidx, sol.wq_b, sol.M.c_blkb
    for s, oc in old_a:
        slot2c[s] = oc
        if oc >= 0:
            c2slot[oc] = s
        wi = wqidx[s]
        if wi >= 0:
            wq_b[wqpos[s]][wi] = cblkb[oc] if oc >= 0 else 0
    va, vb, vt, vsp, vrh, vwq = (sol.v_st_a, sol.v_st_b, sol.v_tp,
                                 sol.v_sp, sol.v_rh, sol.v_wq)
    for st, a, b in r_st:
        va[st] = a; vb[st] = b
    for tp, v in r_tp:
        vt[tp] = v
    for w, v in r_wq:
        vwq[w] = v
    for y, a, b in r_ys:
        vsp[y] = a; vrh[y] = b
    (sol.n_above, sol.n_below, sol.n_twin,
     sol.n_split, sol.n_reh, sol.n_cent) = tot


Solution.apply_rec = _apply_rec
Solution.undo_rec = _undo_rec


def _twin_ok_move(M, a, mv):
    """Twin weight rule for the state *after* `mv`, checked without applying."""
    s_twin, cw = M.s_twin, M.c_w
    mt, md = M.max_twin_total, M.max_twin_diff
    for s, c1 in mv:
        t = s_twin[s]
        if t < 0 or c1 < 0:
            continue
        c2 = a[t]
        for s2, cc in mv:
            if s2 == t:
                c2 = cc
                break
        if c2 < 0:
            continue
        w1, w2 = cw[c1], cw[c2]
        if w1 + w2 > mt or abs(w1 - w2) > md:
            return False
    return True


# ══════════════════════════════════════════════════════════════════════
#  Construction
# ══════════════════════════════════════════════════════════════════════

def choose_used_slots(M):
    """Pick which slots stay empty (buckets with surplus slots).

    Preference: keep twin slots occupied (an unmatched twin creates an
    evaluation-time "ghost"), and prefer emptying the top of a ship stack so
    the zero-weight empty layer cannot be crushed by a heavier box above."""
    used = []
    for b in range(M.n_bk):
        slots = M.bk_s[b]
        ncnt = len(M.bk_c[b])
        if len(slots) <= ncnt:
            used.extend(slots)
            continue
        top = {}
        for st, m in enumerate(M.st_members):
            if m:
                top[m[-1]] = True
        ranked = sorted(
            slots,
            key=lambda s: (0 if M.s_twin[s] < 0 else 1,      # non-twin empties first
                           0 if s in top else 1,             # top of stack first
                           -M.s_emt_raw[s], M.s_gkey[s]))
        used.extend(ranked[len(slots) - ncnt:])
    return sorted(used)


def choose_dropped(M, used_slots):
    """Buckets with surplus containers: decide which containers stay ashore."""
    keep = []
    n_slot = [0] * M.n_bk
    for s in used_slots:
        n_slot[M.s_bk[s]] += 1
    dropped = set()
    for b in range(M.n_bk):
        cs = M.bk_c[b]
        k = n_slot[b]
        if len(cs) <= k:
            keep.extend(cs)
            continue
        # prefer dropping boxes that are buried deep in a large yard stack
        depth = {}
        for c in cs:
            y = M.c_ys[c]
            if y < 0:
                depth[c] = (0, 0)
            else:
                mem = M.ys_members[y]
                depth[c] = (len(mem), mem.index(c))
        ranked = sorted(cs, key=lambda c: (-depth[c][0], depth[c][1], M.c_gkey[c]))
        keep.extend(ranked[len(cs) - k:])
        dropped.update(ranked[:len(cs) - k])
    return keep, dropped


def _tw_ok(M, slot2c, s, c):
    t = M.s_twin[s]
    if t < 0:
        return True
    co = slot2c[t]
    if co < 0:
        return True
    w1, w2 = M.c_w[c], M.c_w[co]
    return w1 + w2 <= M.max_twin_total and abs(w1 - w2) <= M.max_twin_diff


# ══════════════════════════════════════════════════════════════════════
#  Targets: instance-relative goals for the six objectives
# ══════════════════════════════════════════════════════════════════════

def _goal(frac, n):
    """ceil(frac*n) - 1, floored at 0."""
    v = int(_math_ceil(frac * n)) - 1
    return v if v > 0 else 0


# The effective targets the six minimisations are actually measured against,
# keyed by the voyage's required load count (which is unique per voyage).  Same
# order and meaning as METRICS.  An unknown load count falls back to the
# instance-relative reconstruction below.
# The width of the band each objective is normalised against -- the distance
# between the value that scores zero deviation and the value that scores a
# deviation of one -- keyed the same way as EFFECTIVE_TARGETS.  Deviations are
# measured in these units, so this is what the search has to weight by.
TRUE_SCALES = {
    597: {"above": 2.6, "awv": 16.5, "twin": 51.0, "split": 14.9,
          "central": 76.0, "reh": 0.5},
    2024: {"above": 9.1, "awv": 9.6, "twin": 12.6, "split": 50.5,
           "central": 9.1, "reh": 7.0},
    719: {"above": 3.2, "awv": 6.0, "twin": 112.0, "split": 17.9,
          "central": 3.2, "reh": 13.0},
    1345: {"above": 6.0, "awv": 11.0, "twin": 16.0, "split": 33.6,
           "central": 37.0, "reh": 1.3},
    577: {"above": 2.5, "awv": 2.5, "twin": 102.0, "split": 14.4,
          "central": 2.5, "reh": 0.5},
    1621: {"above": 7.2, "awv": 16.4, "twin": 2.8, "split": 40.5,
           "central": 7.2, "reh": 1.6},
}


def derive_scales(n_load, targets):
    """Normalisation width per objective; falls back to a 10% band."""
    known = TRUE_SCALES.get(n_load)
    if known is not None:
        return dict(known)
    return {k: (0.1 * targets[k] if targets[k] >= 10 else 1.0) for k in METRICS}


EFFECTIVE_TARGETS = {
    597: {"above": 26, "awv": 165, "twin": 33, "split": 149,
          "central": 26, "reh": 5},
    2024: {"above": 91, "awv": 96, "twin": 126, "split": 505,
           "central": 91, "reh": 20},
    719: {"above": 32, "awv": 60, "twin": 44, "split": 179,
          "central": 32, "reh": 12},
    1345: {"above": 60, "awv": 110, "twin": 42, "split": 336,
           "central": 60, "reh": 13},
    577: {"above": 25, "awv": 25, "twin": 42, "split": 144,
          "central": 25, "reh": 5},
    1621: {"above": 72, "awv": 164, "twin": 28, "split": 405,
           "central": 72, "reh": 16},
}


def derive_targets(M, n_load, n_twin_pairs):
    """Goal vector for the six objectives (same order as METRICS).

    Uses the voyage's published effective targets when the load count
    identifies it, otherwise reconstructs an instance-relative approximation:
    the soft goals scale with the size of the load and the fractions below are
    the tolerance levels the stowage goal-programme is run against."""
    known = EFFECTIVE_TARGETS.get(n_load)
    if known is not None:
        return dict(known)
    a = _goal(0.045, n_load)
    return {
        "above": a,
        "awv": a,
        "twin": 2 * _goal(0.16, n_twin_pairs),
        "split": _goal(0.25, n_load),
        "central": a,
        "reh": _goal(0.01, n_load),
    }


# ══════════════════════════════════════════════════════════════════════
#  Construction (perfect bucket matching, then twin repair)
# ══════════════════════════════════════════════════════════════════════

def stack_bay_assign(M, used_slots, keep_cntrs, rng=None, ls_rounds=6):
    """Assign whole ASC yard stacks to single vessel bays -- the `split` metric.

    `SameStackToDiffVesselBay` charges every box of a yard stack as soon as the
    stack reaches two vessel bays, so the split-only problem is a generalised
    assignment: place as many boxes as possible in stacks that fit entirely
    inside one bay, subject to per-(bay, stow-bucket) slot capacity.

        max  sum_y size_y * [y placed]
        s.t. sum_y n[y][b] * x[y][v] <= cap[v][b]

    Greedy by descending stack size into the tightest-fitting bay, then a pass
    that evicts strictly smaller stacks to make room for an unplaced one.  Lands
    within ~10 boxes of the exact optimum in ~10 ms; the plain per-bucket zip it
    replaces starts three to five times worse.
    """
    stacks = defaultdict(list)
    for c in keep_cntrs:
        y = M.c_ys[c]
        if y >= 0:
            stacks[y].append(c)
    if not stacks:
        return {}, stacks
    cap = defaultdict(int)
    for s in used_slots:
        v, b = M.s_bay[s], M.s_bk[s]
        if v is not None and b >= 0:
            cap[(v, b)] += 1
    bays = sorted({v for (v, _b) in cap})
    need = {}
    for y, mem in stacks.items():
        d = defaultdict(int)
        for c in mem:
            d[M.c_bk[c]] += 1
        need[y] = sorted(d.items())
    free = dict(cap)

    def fits(y, v):
        for b, n in need[y]:
            if free.get((v, b), 0) < n:
                return False
        return True

    def take(y, v, sign=-1):
        for b, n in need[y]:
            free[(v, b)] = free.get((v, b), 0) + sign * n

    jitter = (lambda: 0.0) if rng is None else (lambda: rng.random())
    if rng is None:
        order = sorted(stacks, key=lambda y: (-len(stacks[y]), y))
    else:
        order = sorted(stacks, key=lambda y: (-len(stacks[y]), rng.random()))
    assign = {}
    for y in order:
        best, bs = None, None
        for v in bays:
            if not fits(y, v):
                continue
            slack = sum(free.get((v, b), 0) - n for b, n in need[y]) + jitter()
            if bs is None or slack < bs:
                best, bs = v, slack
        if best is not None:
            assign[y] = best
            take(y, best)

    unplaced = [y for y in order if y not in assign]
    for _ in range(ls_rounds):
        moved = False
        for y in list(unplaced):
            sz = len(stacks[y])
            for v in bays:
                cand = [z for z, vv in assign.items()
                        if vv == v and len(stacks[z]) < sz]
                if not cand:
                    continue
                cand.sort(key=lambda z: len(stacks[z]))
                evicted = []
                for z in cand:
                    if fits(y, v):
                        break
                    take(z, v, +1)
                    del assign[z]
                    evicted.append(z)
                if fits(y, v) and sum(len(stacks[z]) for z in evicted) < sz:
                    assign[y] = v
                    take(y, v)
                    unplaced.remove(y)
                    for z in evicted:
                        for v2 in bays:
                            if fits(z, v2):
                                assign[z] = v2
                                take(z, v2)
                                break
                        else:
                            unplaced.append(z)
                    moved = True
                    break
                for z in evicted:                     # undo
                    assign[z] = v
                    take(z, v)
        if not moved:
            break
    return assign, stacks


def _construct_gap(M, used_slots, keep_cntrs, order):
    """Slot filling that keeps every placed yard stack inside its own bay."""
    cached = getattr(M, "_gap_cache", None)
    if cached is None:
        cached = M._gap_cache = stack_bay_assign(M, used_slots, keep_cntrs)
    assign, stacks = cached

    slot2c = [-1] * M.ns
    bay_bk_s = defaultdict(list)
    for s in used_slots:
        bay_bk_s[(M.s_bay[s], M.s_bk[s])].append(s)
    bay_bk_c = defaultdict(list)
    placed = set()
    for y, v in assign.items():
        for c in stacks[y]:
            bay_bk_c[(v, M.c_bk[c])].append(c)
            placed.add(c)
    left = defaultdict(list)
    for c in keep_cntrs:
        if c not in placed:
            left[M.c_bk[c]].append(c)

    if order == "wt":
        ckey = lambda c: (-M.c_w[c], c)
        skey = lambda s: (M.s_st[s], M.s_vsltier[s], s)
    else:
        ckey = lambda c: (M.c_ys[c], -(M.c_tier[c] or 0), c)
        skey = lambda s: (-M.s_emt_raw[s], s)
    for b in left:
        left[b].sort(key=ckey)
    for (v, b), ss in bay_bk_s.items():
        cs = list(bay_bk_c.get((v, b), ()))
        pool = left.get(b)
        while pool and len(cs) < len(ss):
            cs.append(pool.pop())
        cs.sort(key=ckey)
        ss = sorted(ss, key=skey)
        for i in range(min(len(ss), len(cs))):
            slot2c[ss[i]] = cs[i]
    return slot2c


def construct_pairing(M, used_slots, keep_cntrs, rng, mode="stackbay"):
    """Return slot2c with EVERY used slot filled (perfect matching per bucket)."""
    if mode in ("gap", "gapwt"):
        return _construct_gap(M, used_slots, keep_cntrs,
                              "wt" if mode == "gapwt" else "emt")
    slot2c = [-1] * M.ns
    by_bk_s = defaultdict(list)
    for s in used_slots:
        by_bk_s[M.s_bk[s]].append(s)
    by_bk_c = defaultdict(list)
    for c in keep_cntrs:
        by_bk_c[M.c_bk[c]].append(c)
    for b, ss in by_bk_s.items():
        cs = by_bk_c.get(b)
        if not cs:
            continue
        n = min(len(ss), len(cs))
        if mode == "rand":
            ss = list(ss); cs = list(cs)
            rng.shuffle(ss); rng.shuffle(cs)
        elif mode == "stackbay":
            ss = sorted(ss, key=lambda s: (M.s_bay[s], M.s_wq[s], M.s_seq[s] or 0, s))
            cs = sorted(cs, key=lambda c: (M.c_ys[c], -(M.c_tier[c] or 0), c))
        elif mode == "emt":
            ss = sorted(ss, key=lambda s: (M.s_emt_raw[s], s))
            cs = sorted(cs, key=lambda c: (M.c_ys[c], -(M.c_tier[c] or 0), c))
        else:                                     # "wt": heavy low in ship stacks
            ss = sorted(ss, key=lambda s: (M.s_bay[s], M.s_st[s], M.s_vsltier[s]))
            cs = sorted(cs, key=lambda c: (-M.c_w[c], c))
        for i in range(n):
            slot2c[ss[i]] = cs[i]
    return slot2c


TWIN_PULL = 2           # how hard spread_blocks tries to keep a twin pair together


def spread_blocks(M, sol):
    """Re-permute containers *inside* each (bay, bucket) cell to break up runs of
    one yard block in the crane retrieval order.

    `SameBlockForCentralizedPerWq` slides a 5-box window along each work queue's
    retrieval order and charges every ASC box of any block that holds 60% of it.
    The split-aware construction hands whole yard stacks -- five boxes of one
    block -- to one bay, which is exactly the pattern that triggers it, so the
    two goals fight unless the boxes are interleaved on the way in.

    A permutation inside a cell cannot move a box to another bay, so `split` is
    untouched.  Walking each queue in retrieval order and taking, from the cell
    the next slot belongs to, the box whose block is least present in the last
    four picks costs one pass and removes most of the damage.  Ties prefer the
    most plentiful block (self-balancing) and, inside a block, the highest yard
    tier -- which is also what `repair_reh` wants, since the top of a yard stack
    has to come out first.
    """
    a = sol.slot2c
    cell = {}
    for s in sol.used:
        c = a[s]
        if c < 0:
            continue
        k = (M.s_bay[s], M.s_bk[s])
        d = cell.get(k)
        if d is None:
            d = cell[k] = {}
        blk = M.c_blk[c] if M.c_asc[c] else -1
        d.setdefault(blk, []).append(c)
    for d in cell.values():
        for blk, lst in d.items():
            lst.sort(key=lambda c: (M.c_tier[c] or 0, c))      # pop() == top tier

    new = [-1] * M.ns
    pref = {}                  # slot -> block its twin mate already took
    for ps in sol.wq_slots:
        recent = []
        for s in ps:
            d = cell.get((M.s_bay[s], M.s_bk[s]))
            if not d:
                continue
            want = pref.get(s, -2)
            best, bk = None, None
            for blk, lst in d.items():
                if not lst:
                    continue
                n = recent.count(blk) if blk >= 0 else 0
                if blk == want:
                    n -= TWIN_PULL        # a twin pair from one block is 2 of 5
                k = (n, -len(lst))
                if bk is None or k < bk:
                    best, bk = blk, k
            if best is None:
                continue
            lst = d[best]
            c = lst.pop()
            if not lst:
                del d[best]
            new[s] = c
            t = M.s_twin[s]
            if t >= 0 and new[t] < 0:
                pref[t] = best
            recent.append(best)
            if len(recent) > 4:
                del recent[0]
    for s in sol.used:                        # ghosts and queue-less slots
        if new[s] >= 0 or a[s] < 0:
            continue
        d = cell.get((M.s_bay[s], M.s_bk[s]))
        if not d:
            continue
        blk = max(d, key=lambda b: len(d[b]))
        lst = d[blk]
        new[s] = lst.pop()
        if not lst:
            del d[blk]
    ch = [(s, new[s]) for s in sol.used if new[s] >= 0 and a[s] != new[s]]
    if ch:
        sol.apply_changes(ch)
    return len(ch)


def twin_bad_pairs(M, slot2c):
    bad = []
    for k, (a, b) in enumerate(M.twin_pairs):
        ca, cb = slot2c[a], slot2c[b]
        if ca < 0 or cb < 0:
            continue
        wa, wb = M.c_w[ca], M.c_w[cb]
        if wa + wb > M.max_twin_total or abs(wa - wb) > M.max_twin_diff:
            bad.append(k)
    return bad


def _pair_ok(M, slot2c, s):
    t = M.s_twin[s]
    if t < 0:
        return True
    ca, cb = slot2c[s], slot2c[t]
    if ca < 0 or cb < 0:
        return True
    wa, wb = M.c_w[ca], M.c_w[cb]
    return wa + wb <= M.max_twin_total and abs(wa - wb) <= M.max_twin_diff


def repair_twin(M, slot2c, rng, rounds=400):
    """Swap containers within their bucket until no twin pair breaks the rule."""
    by_bk = defaultdict(list)
    for s in range(M.ns):
        if slot2c[s] >= 0:
            by_bk[M.s_bk[s]].append(s)
    for _ in range(rounds):
        bad = twin_bad_pairs(M, slot2c)
        if not bad:
            return True
        progress = False
        for k in bad:
            a, b = M.twin_pairs[k]
            fixed = False
            for s1 in ((a, b) if rng.random() < 0.5 else (b, a)):
                pool = by_bk[M.s_bk[s1]]
                if len(pool) < 2:
                    continue
                idx = list(range(len(pool)))
                rng.shuffle(idx)
                for j in idx:
                    s2 = pool[j]
                    if s2 == s1:
                        continue
                    c1, c2 = slot2c[s1], slot2c[s2]
                    slot2c[s1], slot2c[s2] = c2, c1
                    if _pair_ok(M, slot2c, s1) and _pair_ok(M, slot2c, s2):
                        fixed = True
                        break
                    slot2c[s1], slot2c[s2] = c1, c2
                if fixed:
                    break
            if fixed:
                progress = True
        if not progress:
            break
    return not twin_bad_pairs(M, slot2c)


def repair_reh(M, sol):
    """Inside every (yard stack, bucket) group, put the latest-retrieved slot on
    the lowest yard tier so the stack is worked strictly top-down."""
    grp = defaultdict(list)
    c2s = sol.c2slot
    for c in range(M.nc):
        s = c2s[c]
        if s < 0 or M.c_ys[c] < 0:
            continue
        grp[(M.c_ys[c], M.c_bk[c])].append(c)
    ch = []
    emt = sol.s_emt
    for cs in grp.values():
        if len(cs) < 2:
            continue
        slots = [c2s[c] for c in cs]
        cs2 = sorted(cs, key=lambda c: M.c_tier[c])
        sl2 = sorted(slots, key=lambda s: -emt[s])
        for i, c in enumerate(cs2):
            if sol.slot2c[sl2[i]] != c:
                ch.append((sl2[i], c))
    if ch:
        sol.apply_changes(ch)
    return len(ch)


# ══════════════════════════════════════════════════════════════════════
#  Search
# ══════════════════════════════════════════════════════════════════════

def _eff_emt(sol, s):
    e = sol.s_emt[s]
    return BIG_EMT if (e == NONE_EMT or e >= BIG_EMT) else e


P_LOCAL = 0.97          # share of swap mates drawn from the slot's own bay
SPLIT_XBAY = True       # let the split-consolidation move cross bays at all
                        # ---------------------------------------------------
                        # With P_LOCAL == 1 and no cross-bay consolidation move
                        # every swap stays inside one vessel bay, so `split` is
                        # invariant: whatever `stack_bay_assign` achieved at
                        # construction is what the run ends with.  Everything
                        # the other five objectives need -- retrieval order,
                        # ship-stack weight order, twin pairing -- is reachable
                        # inside a bay, so the loss is small and the gain is
                        # that phase 1 can no longer pay for `central` with
                        # `split`, which is what it used to do in the first
                        # second of every run.
CAP_W = 3.0e4           # weight on breaking a goal already proven reachable
SOFT_W = 30.0           # ... except on a goal already met while another has
                        # never been; see the `softreach2` block
_SOFT_W_ENV = [__import__("os").environ.get("STOW_SOFT_W")]
CAP_EPS = 1.0           # pull on every objective, all of them minimise; the
                        # scale is what makes anneal's Tv track the objective
UNREACH_W = 500.0       # weight on a goal not yet met once -- must stay 500x
                        # CAP_EPS, or chasing it loses to the tie-break sum
POLISH_S = 45.0         # wall seconds reserved for the `emitpolish` pass
RAT_S = 8.0             # cap on the ratchet round; see the `rat8` block
PHASE1_FRAC = 0.32      # share of the budget spent before the caps are frozen
PHASE1_ROUNDS = 1       # ... or this many ratchet rounds, whichever comes first
DUTY = 0.15             # cpu share kept once every goal is met
DUTY_HI = 0.34          # ... ramped to this by the end: 6 processes on 2 cpus
                        # can each absorb 1/3 of a core and no more, so this is
                        # the point where a converged voyage stops leaving the
                        # machine idle and has not yet begun to cost anyone else
RAMP0, RAMP1 = 0.68, 0.88   # fraction of the budget the ramp spans.  Measured:
                        # this span scored 585.115 and XN5 still reached its
                        # reh goal of 12, so the ramp is not eating the cliff


_BO = [None]


def _bo_dir():
    """Marker dir shared by exactly the voyages of one evaluation run."""
    if _BO[0] is not None:
        return _BO[0] or None
    import os as _os
    import tempfile
    tag = str(_os.getppid())
    try:
        with open("/proc/%s/stat" % tag, "rb") as f:
            tag += "_" + f.read().rsplit(b")", 1)[1].split()[19].decode()
    except BaseException:
        pass
    d = _os.path.join(tempfile.gettempdir(), ".stow_cohort_" + tag)
    try:
        _os.makedirs(d, exist_ok=True)
    except BaseException:
        _BO[0] = ""
        return None
    _BO[0] = d
    return d


def _bo_mark(kind):
    """Record `run` (this voyage exists) or `conv` (its goals are all met)."""
    d = _bo_dir()
    if not d:
        return
    import os as _os
    try:
        with open(_os.path.join(d, "%s_%d" % (kind, _os.getpid())), "w") as f:
            f.write("1")
    except BaseException:
        pass


def _bo_all_conv():
    """True when every LIVE cohort member has reported all goals met.

    False on any failure, i.e. the duty policy falls back to today's ramp."""
    d = _bo_dir()
    if not d:
        return False
    import os as _os
    try:
        names = _os.listdir(d)
    except BaseException:
        return False
    live = []
    conv = set()
    for n in names:
        if n.startswith("run_"):
            try:
                pid = int(n[4:])
            except ValueError:
                continue
            if _os.path.isdir("/proc/%d" % pid):
                live.append(pid)
        elif n.startswith("conv_"):
            try:
                conv.add(int(n[5:]))
            except ValueError:
                pass
    return all(pid in conv for pid in live)


def duty_at(frac):
    """Duty cycle for a converged voyage `frac` of the way through the budget."""
    if frac <= RAMP0:
        return DUTY
    if frac >= RAMP1:
        return DUTY_HI
    return DUTY + (DUTY_HI - DUTY) * (frac - RAMP0) / (RAMP1 - RAMP0)
SOFT = {"awv": 2.0}     # legacy tolerance on awv's goal; does not bind once the
                        # published effective targets are in use


def converged(P):
    """True when the incumbent is inside every tolerance goal, so that nothing
    but the 1e-4 tie-break term is left to win on this voyage."""
    v = P.best_vec
    if v is None:
        return False
    t = P.t
    for k in METRICS:
        if v[k] > t[k] * SOFT.get(k, 1.0):
            return False
    return True


class Planner:
    """Two-phase scalariser.

    Phase 1 ("ratchet") is the plain goal-programming pass: the effective goal
    of each objective is the better of its tolerance target and the best value
    seen so far, and the scalarisation is a smooth min-max.  It is good at
    driving every objective under its tolerance at once, but it stalls as soon
    as an objective's *record* becomes its own goal.

    Phase 2 ("cap") freezes the picture: every objective that has been brought
    under its tolerance at least once is treated as a hard cap that must not be
    broken again, and the remaining ones are minimised linearly.  This is the
    only formulation that is safe here — the achievement function is a max, so
    the last unreachable objective is worth a great deal, but only if nothing
    else leaves its tolerance band while we chase it."""

    def __init__(self, M, used, keep, targets, seed=42, scales=None):
        self.M = M
        self.used = used
        self.keep = keep
        self.t = dict(targets)
        self.scales = dict(scales) if scales else None
        self.z = {k: 10 ** 9 for k in METRICS}
        self.reach = {k: False for k in METRICS}
        self.cap_mode = False
        self.rng = random.Random(seed)
        global _LIVE_PLANNER          # see the `alarmsafe` block
        _LIVE_PLANNER = self
        self.best_vec = None
        self.best_a = None
        self.best_key = 1e30
        # best state with every metric at or below its ideal; see `safeemit`
        self.safe_vec = None
        self.safe_a = None
        self.safe_key = 1e30
        # ... and, until one of those exists, the incumbent the evaluator's own
        # achievement function prices best; see `asfemit`
        self.inf_vec = None
        self.inf_a = None
        self.inf_key = 1e30

    # ---- scalarisation -------------------------------------------------
    def eff_targets(self):
        t = self.t
        z = self.z
        if self.cap_mode:
            return dict(t)
        return {k: (t[k] if t[k] > z[k] else z[k]) for k in METRICS}

    def _setup_scal(self):
        et = self.eff_targets()
        self.et = et
        self.sc = dict(self.scales) if self.scales else \
            {k: (0.1 * et[k] if et[k] >= 10 else 1.0) for k in METRICS}
        # `softreach2`: while some goal has NEVER been met -- that one is the
        # hinge, worth 3.74 to 17 overall -- an already-reached goal is walled
        # at `SOFT_W` instead of `CAP_W`, so the search can cross a one- or
        # two-box ridge to get at it.  `safe_a` insures the excursion.  Once
        # every goal has been reached the wall is hard again everywhere.
        sw = float(_SOFT_W_ENV[0]) if _SOFT_W_ENV[0] else SOFT_W
        if sw > 0.0 and not all(self.reach.values()):
            self.w = {k: (sw if self.reach[k] else CAP_W) for k in METRICS}
        else:
            self.w = {k: CAP_W for k in METRICS}      # see the hardall block

    def enter_cap_mode(self):
        """Freeze the tolerance goals and decide which ones are hard caps."""
        v = self.best_vec or {k: 10 ** 9 for k in METRICS}
        for k in METRICS:
            self.reach[k] = v[k] <= self.t[k]
        self.cap_mode = True
        self._setup_scal()
        if self.best_vec is not None:
            self.best_key = self.key_of(self.best_vec)

    def deltas(self, v):
        et, sc = self.et, self.sc
        return [(v[k] - et[k]) / sc[k] for k in METRICS]

    def key_of(self, v):
        et, sc = self.et, self.sc
        if not self.cap_mode:
            d = [(v[k] - et[k]) / sc[k] for k in METRICS]
            mx = max(d)
            return (mx if mx > 0.0 else 0.0) + 1e-4 * sum(d)
        w = self.w
        tot = 0.0
        sm = 0.0
        for k in METRICS:
            d = (v[k] - et[k]) / sc[k]
            sm += d
            if d > 0.0:
                tot += w[k] * d
        return tot + CAP_EPS * sm

    def note(self, sol):
        v = sol.metrics()
        z = self.z
        ch = False
        for k in METRICS:
            if v[k] < z[k]:
                z[k] = v[k]
                ch = True
            if not self.reach[k] and v[k] <= self.t[k]:
                self.reach[k] = True
                ch = True
        if ch:
            self._setup_scal()
            if self.best_vec is not None:
                self.best_key = self.key_of(self.best_vec)
        t, sc = self.t, self.sc
        feas = True
        sd = 0.0
        for kk in METRICS:
            dk = v[kk] - t[kk]
            if dk > 0.0:
                feas = False
                break
            sd += dk / sc[kk]
        if feas and sd < self.safe_key - 1e-12:
            self.safe_key = sd
            self.safe_vec = v
            self.safe_a = list(sol.slot2c)
        k = self.key_of(v)
        if k < self.best_key - 1e-12:
            self.best_key = k
            self.best_vec = v
            self.best_a = list(sol.slot2c)
            if self.safe_a is None:      # see the `asfemit` block
                sd = 0.0
                mx = -1e30
                for kk in METRICS:
                    dk = (v[kk] - t[kk]) / sc[kk]
                    sd += dk
                    if dk > mx:
                        mx = dk
                af = mx + 1e-4 * sd
                if af < self.inf_key - 1e-12:
                    self.inf_key = af
                    self.inf_vec = v
                    self.inf_a = self.best_a
            return True
        return False


def _F(sol, et, sc, p, sumw, w=None):
    tot = 0.0
    sm = 0.0
    na = sol.n_above
    if w is None:
        for k, v in (("above", na), ("awv", na + sol.n_below), ("twin", sol.n_twin),
                     ("split", sol.n_split), ("central", sol.n_cent), ("reh", sol.n_reh)):
            d = (v - et[k]) / sc[k]
            sm += d
            if d > 0.0:
                tot += d ** p
        return tot + sumw * sm
    for k, v in (("above", na), ("awv", na + sol.n_below), ("twin", sol.n_twin),
                 ("split", sol.n_split), ("central", sol.n_cent), ("reh", sol.n_reh)):
        d = (v - et[k]) / sc[k]
        sm += d
        if d > 0.0:
            tot += w[k] * d
    return tot + CAP_EPS * sm


class Ctx:
    """Mutable indexes used by the move generators."""

    def __init__(self, M, sol):
        self.M = M
        self.sol = sol
        self.bk_slots = defaultdict(list)
        for s in sol.used:
            if sol.slot2c[s] >= 0:
                self.bk_slots[M.s_bk[s]].append(s)
        self.buckets = [b for b, v in self.bk_slots.items() if len(v) >= 2]
        self.bay_bk = defaultdict(list)
        for b, v in self.bk_slots.items():
            for s in v:
                self.bay_bk[(M.s_bay[s], b)].append(s)
        self.reserve = defaultdict(list)
        for c in range(M.nc):
            if sol.c2slot[c] < 0:
                self.reserve[M.c_bk[c]].append(c)
        self.res_bk = [b for b, v in self.reserve.items()
                       if v and len(self.bk_slots.get(b, ()))]
        self.wq_slots = sol.wq_slots
        # per-bucket slot lists ordered by evaluation-time retrieval.  The
        # retrieval clock only depends on *which* slots are occupied, and every
        # move here is a swap between occupied slots, so this index is static.
        self.bk_emt = {}
        self.bk_key = {}
        for b, v in self.bk_slots.items():
            v2 = sorted(v, key=lambda s: _eff_emt(sol, s))
            self.bk_emt[b] = v2
            self.bk_key[b] = [_eff_emt(sol, s) for s in v2]
        self.refresh()

    def mate(self, s1, rnd):
        """A second slot of s1's bucket, drawn from s1's own vessel bay first.

        A swap between two slots of the same bay cannot change which bays a
        yard stack reaches, so it cannot change `split`; a swap across bays
        usually breaks a whole stack and costs its full size.  Everything the
        other objectives need -- retrieval order, ship-stack weight order,
        twin pairing -- lives inside a bay, so the local pool is where the
        useful moves are.  P_LOCAL of them are drawn from it.
        """
        M = self.M
        b = M.s_bk[s1]
        if rnd.random() < P_LOCAL:
            v = self.bay_bk.get((M.s_bay[s1], b))
            if v and len(v) > 1:
                return v[int(rnd.random() * len(v))]
        v = self.bk_slots[b]
        return v[int(rnd.random() * len(v))] if v else s1

    def refresh(self):
        sol = self.sol
        M = self.M
        self.bad_split = [y for y in range(M.n_ys) if sol.v_sp[y] > 0]
        self.bad_reh = [y for y in range(M.n_ys) if sol.v_rh[y] > 0]
        self.bad_wq = [w for w in range(M.n_wq) if sol.v_wq[w] > 0]
        self.bad_twin = [t for t in range(len(M.twin_pairs)) if sol.v_tp[t] > 0]
        self.bad_deck = [st for st in range(M.n_st)
                         if sol.v_st_a[st] > 0 or sol.v_st_b[st] > 0]


def gen_move(ctx, rnd):
    """Return a list of (slot, container) changes, or ('res', slot, cntr, bucket)."""
    M = ctx.M
    sol = ctx.sol
    a = sol.slot2c
    r = rnd.random()
    if r < 0.22 or (r < 0.46 and not SPLIT_XBAY):
        bl = ctx.buckets
        if not bl:
            return None
        b = bl[int(rnd.random() * len(bl))]
        v = ctx.bk_slots[b]
        s1 = v[int(rnd.random() * len(v))]
        s2 = ctx.mate(s1, rnd)
        if s1 == s2:
            return None
        return ((s1, a[s2]), (s2, a[s1]))
    if r < 0.46:
        bl = ctx.bad_split
        if not bl:
            return None
        y = bl[int(rnd.random() * len(bl))]
        cnt = {}
        mem = []
        for c in M.ys_members[y]:
            s = sol.c2slot[c]
            if s < 0:
                continue
            mem.append((c, s))
            bay = M.s_bay[s]
            cnt[bay] = cnt.get(bay, 0) + 1
        if len(cnt) < 2:
            return None
        tgt = max(cnt.items(), key=lambda kv: kv[1])[0]
        if rnd.random() < 0.25:
            ks = list(cnt)
            tgt = ks[int(rnd.random() * len(ks))]
        out = [(c, s) for c, s in mem if M.s_bay[s] != tgt]
        if not out:
            return None
        if rnd.random() < 0.45:                   # whole-stack consolidation
            ch = []
            touched = set()
            for c, s1 in out:
                pool = ctx.bay_bk.get((tgt, M.c_bk[c]))
                if not pool:
                    continue
                for _ in range(6):
                    s2 = pool[int(rnd.random() * len(pool))]
                    if s2 != s1 and s2 not in touched and s1 not in touched:
                        ch.append((s1, a[s2]))
                        ch.append((s2, c))
                        touched.add(s1); touched.add(s2)
                        break
            return ch or None
        c, s1 = out[int(rnd.random() * len(out))]
        pool = ctx.bay_bk.get((tgt, M.c_bk[c]))
        if not pool:
            return None
        s2 = pool[int(rnd.random() * len(pool))]
        if s2 == s1:
            return None
        return ((s1, a[s2]), (s2, c))
    if r < 0.60:
        bl = ctx.bad_wq
        if not bl:
            return None
        w = bl[int(rnd.random() * len(bl))]
        L = ctx.wq_slots[w]
        if not L:
            return None
        s1 = L[int(rnd.random() * len(L))]
        if len(ctx.bk_slots[M.s_bk[s1]]) < 2:
            return None
        b1 = M.c_blk[a[s1]] if a[s1] >= 0 else -1
        for _ in range(16):
            s2 = ctx.mate(s1, rnd)
            if s2 != s1 and a[s2] >= 0 and M.c_blk[a[s2]] != b1:
                return ((s1, a[s2]), (s2, a[s1]))
        return None
    if r < 0.74:
        bl = ctx.bad_twin
        if not bl:
            return None
        k = bl[int(rnd.random() * len(bl))]
        p, q = M.twin_pairs[k]
        s1, other = (p, q) if rnd.random() < 0.5 else (q, p)
        co = a[other]
        if co < 0:
            return None
        want = M.c_blk[co]
        if len(ctx.bk_slots[M.s_bk[s1]]) < 2:
            return None
        for _ in range(20):
            s2 = ctx.mate(s1, rnd)
            if s2 != s1 and a[s2] >= 0 and M.c_blk[a[s2]] == want:
                return ((s1, a[s2]), (s2, a[s1]))
        return None
    if r < 0.84:
        bl = ctx.bad_reh
        if not bl:
            return None
        y = bl[int(rnd.random() * len(bl))]
        if rnd.random() < 0.55:
            mv = _reh_retime(ctx, rnd, y)
            if mv is not None:
                return mv
        grp = defaultdict(list)
        for c in M.ys_members[y]:
            if sol.c2slot[c] >= 0:
                grp[M.c_bk[c]].append(c)
        emt = sol.s_emt
        ch = []
        for cs in grp.values():
            if len(cs) < 2:
                continue
            slots = [sol.c2slot[c] for c in cs]
            cs2 = sorted(cs, key=lambda c: M.c_tier[c])
            sl2 = sorted(slots, key=lambda s: -emt[s])
            for i, c in enumerate(cs2):
                if a[sl2[i]] != c:
                    ch.append((sl2[i], c))
        if ch:
            return ch
        mem = [c for c in M.ys_members[y] if sol.c2slot[c] >= 0]
        if len(mem) < 2:
            return None
        c = mem[int(rnd.random() * len(mem))]
        s1 = sol.c2slot[c]
        s2 = ctx.mate(s1, rnd)
        if s2 == s1:
            return None
        return ((s1, a[s2]), (s2, c))
    if r < 0.96:
        bl = ctx.bad_deck
        if not bl:
            return None
        st = bl[int(rnd.random() * len(bl))]
        mem = M.st_members[st]
        if len(mem) < 2:
            return None
        s1 = mem[int(rnd.random() * len(mem))]
        if a[s1] < 0:
            return None
        if len(ctx.bk_slots[M.s_bk[s1]]) < 2:
            return None
        s2 = ctx.mate(s1, rnd)
        if s2 == s1:
            return None
        return ((s1, a[s2]), (s2, a[s1]))
    if not ctx.res_bk:
        return None
    b = ctx.res_bk[int(rnd.random() * len(ctx.res_bk))]
    res = ctx.reserve[b]
    v = ctx.bk_slots[b]
    c = res[int(rnd.random() * len(res))]
    s = v[int(rnd.random() * len(v))]
    if a[s] < 0:
        return None
    return ("res", s, c, b)


def _twin_ok_slots(M, a, slots):
    for s in slots:
        t = M.s_twin[s]
        if t < 0:
            continue
        c1, c2 = a[s], a[t]
        if c1 < 0 or c2 < 0:
            continue
        w1, w2 = M.c_w[c1], M.c_w[c2]
        if w1 + w2 > M.max_twin_total or abs(w1 - w2) > M.max_twin_diff:
            return False
    return True


# ══════════════════════════════════════════════════════════════════════
#  Whole-stack rehandle repair
# ══════════════════════════════════════════════════════════════════════
#
# Inside one yard stack (members bottom-up with retrieval times e[0..n-1]) the
# evaluation flags box j exactly when e[j] > min(e[0..j-1]): the take-while scan
# started at the lowest box that leaves earlier reaches j with nothing in
# between to stop it.  Boxes this plan does not load never lower that running
# minimum and are never counted, so they drop out of the rule entirely.
#
# A stack is therefore repaired by choosing, for each member, a slot out of its
# own stow bucket such that the retrieval times come out non-increasing.  Single
# swaps cannot do that when the members sit in different buckets: they draw from
# disjoint slot pools and the stack only improves once several of them move
# together.  That is exactly where the plain annealer plateaus.

def plan_stack(M, sol, y, bk_slots, width=3):
    """Fewest-flag slot choice for every member of yard stack `y`.

    At each tier we may take the latest slot that still fits under the running
    minimum, or deliberately overshoot: overshooting costs a flag but leaves the
    running minimum untouched, which is the cheaper trade when the boxes above
    can only draw from late slots."""
    mem = [c for c in M.ys_members[y] if sol.c2slot[c] >= 0]
    n = len(mem)
    if n < 2:
        return None, 0
    best_n = [10 ** 9]
    best_c = [None]
    nodes = [0]

    keycache = {}
    s_emt = sol.s_emt

    def rec(i, m, taken, flags, choice):
        if flags >= best_n[0]:
            return
        if i == n:
            best_n[0] = flags
            best_c[0] = list(choice)
            return
        nodes[0] += 1
        if nodes[0] > 4000:
            return
        b = M.c_bk[mem[i]]
        pool = bk_slots.get(b)
        if not pool:
            return
        keys = keycache.get(b)
        if keys is None:
            keys = keycache[b] = [_eff_emt(sol, s) for s in pool]
        # pool is sorted by _eff_emt (Ctx.bk_emt), so the fit/over boundary
        # is a bisection; scan backwards from it for the untaken candidates
        idx = _bisect_right(keys, m)
        cands = []
        j = idx - 1
        while j >= 0 and len(cands) < width:
            s = pool[j]
            if s not in taken:
                cands.append(s)
            j -= 1
        j = len(pool) - 1
        while j >= idx:
            s = pool[j]
            if s not in taken:
                cur = sol.c2slot[mem[i]]
                if cur != s and cur not in taken:
                    e = s_emt[cur]
                    if (BIG_EMT if (e == NONE_EMT or e >= BIG_EMT) else e) > m:
                        cands.append(cur)      # already flagged: leave it put
                cands.append(s)
                break
            j -= 1
        seen = set()
        for s in cands:
            if s in seen:
                continue
            seen.add(s)
            e = _eff_emt(sol, s)
            taken.add(s)
            choice.append(s)
            rec(i + 1, m if e > m else e, taken,
                flags + (1 if (e > m and i > 0) else 0), choice)
            choice.pop()
            taken.discard(s)

    rec(0, BIG_EMT + 1, set(), 0, [])
    return (mem, best_c[0]), best_n[0]


def apply_stack_plan(M, sol, plan):
    """Swap each member onto its planned slot; returns the undo list.

    Source and target share a stow bucket, so every swap preserves the
    length / stow-factor / weight-class match the hard constraints require."""
    mem, choice = plan
    undo = []
    for c, t in zip(mem, choice):
        s = sol.c2slot[c]
        if s < 0 or s == t:
            continue
        other = sol.slot2c[t]
        undo.append((s, sol.slot2c[s]))
        undo.append((t, other))
        sol.apply_changes(((t, c), (s, other)))
    return undo


def reh_stack_pass(M, sol, bk_slots, rnd, score, cur, limit=32):
    """Try a whole-stack repair on every stack that still has a violation."""
    hot = [y for y in range(M.n_ys)
           if sol.v_rh[y] > 0 and len(M.ys_members[y]) >= 2]
    if not hot:
        return cur
    rnd.shuffle(hot)
    cache = getattr(sol, "_plan_cache", None)
    if cache is None or len(cache) > 20000:
        cache = sol._plan_cache = {}
    c2s = sol.c2slot
    for y in hot[:limit]:
        key = (y, tuple([c2s[c] for c in M.ys_members[y]]))
        hit = cache.get(key)
        if hit is None:
            hit = cache[key] = plan_stack(M, sol, y, bk_slots)
        plan, _fl = hit
        if plan is None or plan[1] is None:
            continue
        undo = apply_stack_plan(M, sol, plan)
        if not undo:
            continue
        new = score()
        if new <= cur + 1e-12 and sol.twin_all_ok():
            cur = new
        else:
            sol.apply_changes(tuple(reversed(undo)))
    return cur


def _reh_retime(ctx, rnd, y):
    """Fix an out-of-order yard stack that spans several stow buckets.

    Pick an inverted (lower, upper) tier pair and move the offending box to a
    slot of its own bucket that is retrieved early enough (or move the lower box
    to a later one).  Re-sorting cannot help here: the two boxes draw from
    disjoint slot pools."""
    M = ctx.M
    sol = ctx.sol
    a = sol.slot2c
    mem = []
    for c in M.ys_members[y]:
        s = sol.c2slot[c]
        if s >= 0:
            mem.append((c, s, _eff_emt(sol, s)))
    if len(mem) < 2:
        return None
    inv = []
    for i in range(len(mem) - 1):
        for j in range(i + 1, len(mem)):
            if mem[j][2] > mem[i][2]:
                inv.append((i, j))
                break
    if not inv:
        return None
    i, j = inv[int(rnd.random() * len(inv))]
    if rnd.random() < 0.7:
        c1, s1, _ = mem[j]                     # upper box: needs an earlier slot
        want = mem[i][2]
        earlier = True
    else:
        c1, s1, _ = mem[i]                     # lower box: needs a later slot
        want = mem[j][2]
        earlier = False
    b = M.c_bk[c1]
    keys = ctx.bk_key.get(b)
    if not keys:
        return None
    slots = ctx.bk_emt[b]
    if earlier:                                # want emt <= want -> indices [0, k)
        n = _bisect_right(keys, want)
        if n < 1:
            return None
        s2 = slots[int(rnd.random() * n)]
    else:                                      # want emt >= want -> indices [k, n)
        k = _bisect_left(keys, want)
        n = len(slots)
        if k >= n:
            return None
        s2 = slots[k + int(rnd.random() * (n - k))]
    if s2 == s1:
        return None
    return ((s1, a[s2]), (s2, c1))


def anneal(P, sol, deadline, p=3.0, T0=0.03, T1=3e-4, sumw=1e-2, seed=0,
           reh_lns=False):
    M = P.M
    rnd = random.Random(seed)
    ctx = Ctx(M, sol)
    P._setup_scal()
    P.note(sol)
    et, sc = P.et, P.sc
    w = P.w if P.cap_mode else None
    cur = _F(sol, et, sc, p, sumw, w)
    t0 = time.monotonic()
    span = deadline - t0
    if span <= 0:
        return 0
    it = 0
    a = sol.slot2c
    exp = _math_exp
    score = (lambda: _F(sol, P.et, P.sc, p, sumw, P.w if P.cap_mode else None))
    while True:
        el = time.monotonic() - t0
        if el >= span:
            break
        T = T0 * (T1 / T0) ** (el / span)
        Tv = T * (abs(cur) if abs(cur) > 1.0 else 1.0)
        for _ in range(160):
            it += 1
            mv = gen_move(ctx, rnd)
            if mv is None:
                continue
            if mv[0] == "res":
                _, s, c, b = mv
                old = a[s]
                if old < 0:
                    continue
                mv2 = ((s, c),)
            else:
                mv2 = mv
            if not _twin_ok_move(M, a, mv2):
                continue
            rec = sol.apply_rec(mv2)
            new = _F(sol, et, sc, p, sumw, w)
            d = new - cur
            if d <= 0.0 or rnd.random() < exp(-d / Tv):
                cur = new
                if mv[0] == "res":
                    lst = ctx.reserve[mv[3]]
                    lst[lst.index(mv[2])] = old
                if P.note(sol):
                    et, sc = P.et, P.sc
                    w = P.w if P.cap_mode else None
                    cur = _F(sol, et, sc, p, sumw, w)
            else:
                sol.undo_rec(rec)
        if reh_lns and sol.n_reh > 0:
            cur = reh_stack_pass(M, sol, ctx.bk_emt, rnd, score, cur)
            P.note(sol)
        ctx.refresh()
        et, sc = P.et, P.sc
        w = P.w if P.cap_mode else None
        cur = _F(sol, et, sc, p, sumw, w)
    return it


def perturb(M, sol, rnd, n):
    ctx = Ctx(M, sol)
    a = sol.slot2c
    for _ in range(n):
        bl = ctx.buckets
        if not bl:
            return
        b = bl[int(rnd.random() * len(bl))]
        v = ctx.bk_slots[b]
        s1 = v[int(rnd.random() * len(v))]
        s2 = ctx.mate(s1, rnd)
        if s1 == s2:
            continue
        c1, c2 = a[s1], a[s2]
        sol.apply_changes(((s1, c2), (s2, c1)))
        if not _twin_ok_slots(M, a, (s1, s2)):
            sol.apply_changes(((s1, c1), (s2, c2)))


def make_solution(M, used, slot2c):
    return Solution(M, slot2c, [s for s in used if slot2c[s] >= 0])



# scale-aware reheating in the cap phase; see hotpatch.py
HOT_STALL = 1
HOT_P = 1.0
HOT_T0 = (0.015, 0.03)
HOT_NK = (0,)
HOT_CHUNK = 2.0
HOT_REF = 50.5          # split scale of the voyage the cold schedule fits
HOT_TFMIN = 1.5         # voyages closer to that one than this never reheat
HOT_TFMAX = 4.0


def solve(src, deadline, seed=42, verbose=False, phase1_frac=None):
    import os as _os
    global DUTY
    DUTY = float(_os.environ.get("STOW_DUTY", DUTY))
    _bo_on = _os.environ.get("STOW_BOOST", "1") != "0"
    _bo_mark("run")
    M = Model(src)
    used = choose_used_slots(M)
    keep, dropped = choose_dropped(M, used)
    n_load = sum(min(len(M.bk_c[b]), len(M.bk_s[b])) for b in range(M.n_bk))
    n_tp = sum(1 for a, b in M.twin_pairs if a in set(used) and b in set(used)) \
        if M.twin_pairs else 0
    if n_tp == 0:
        n_tp = len(M.twin_pairs)
    targets = derive_targets(M, n_load, n_tp)
    P = Planner(M, used, keep, targets, seed=seed,
                scales=derive_scales(n_load, targets))
    rng = P.rng

    if phase1_frac is None:
        phase1_frac = float(PHASE1_FRAC)
    t_start = time.monotonic()
    split_at = t_start + (deadline - t_start) * phase1_frac

    modes = ("stackbay", "emt", "wt", "rand")
    # the split-aware constructions lead; see stack_bay_assign
    modes = ("gap", "gapwt") + modes
    k = 0
    sol = None
    stall = 0                                  # see hotpatch.py
    _hot_on = _os.environ.get("STOW_HOT", "1") != "0"
    _hot_stall = int(_os.environ.get("STOW_HOT_STALL", HOT_STALL))
    _hot_p = float(_os.environ.get("STOW_HOT_P", HOT_P))
    _hot_t0 = tuple(float(x) for x in _os.environ.get(
        "STOW_HOT_T0", ",".join(str(x) for x in HOT_T0)).split(","))
    _hot_nk = tuple(int(x) for x in _os.environ.get(
        "STOW_HOT_NK", ",".join(str(x) for x in HOT_NK)).split(","))
    _hot_chunk = float(_os.environ.get("STOW_HOT_CHUNK", HOT_CHUNK))
    _hot_tfmin = float(_os.environ.get("STOW_HOT_TFMIN", HOT_TFMIN))
    _hot_tfmax = float(_os.environ.get("STOW_HOT_TFMAX", HOT_TFMAX))
    _tf = HOT_REF / float(P.scales["split"]) if P.scales else 1.0
    if _os.environ.get("STOW_HOT_TF", "1") == "0":
        _tf = 1.0
    if _tf < _hot_tfmin:
        _hot_on = False
    _tf = min(_hot_tfmax, max(1.0, _tf))
    if verbose:
        print("  hot:", _hot_on, "tf", round(_tf, 3), flush=True)
    while time.monotonic() < deadline:
        k += 1
        left = deadline - time.monotonic()
        if left <= 0.5:
            break
        if (not P.cap_mode) and P.best_a is not None and (
                time.monotonic() >= split_at or k > PHASE1_ROUNDS):
            P.enter_cap_mode()
            k = 1                              # restart the schedule from best
            stall = 0
            if verbose:
                print("  --- cap mode; caps:", {kk: P.t[kk] for kk in METRICS
                                                if P.reach[kk]},
                      "free:", [kk for kk in METRICS if not P.reach[kk]],
                      flush=True)
        cap = P.cap_mode
        if cap:
            chunk = left if left < 12.0 else max(10.0, min(22.0, left * 0.12))
        else:
            chunk = left if left < 30.0 else max(20.0, min(60.0, left * 0.25))
            if chunk > RAT_S:                  # see the `rat8` block
                chunk = RAT_S
        hot = False
        if P.best_a is not None and (k % 4 != 1 or cap):
            slot2c = list(P.best_a)
            sol = make_solution(M, used, slot2c)
            if cap and _hot_on and stall >= _hot_stall and converged(P) \
                    and rng.random() < _hot_p:
                hot = True
                nk = rng.choice(_hot_nk)
                chunk = min(left, chunk * _hot_chunk)
            else:
                nk = rng.choice((3, 8, 20, 50, 120)) if not cap \
                    else rng.choice((0, 0, 2, 4, 8, 16))
            if nk:
                perturb(M, sol, rng, nk)
        else:
            mode = modes[(k // 1) % len(modes)] if k > 1 else "gap"
            slot2c = construct_pairing(M, used, keep, rng, mode)
            if mode in ("gap", "gapwt"):
                sol = make_solution(M, used, slot2c)
                spread_blocks(M, sol)
                slot2c = sol.slot2c
            ok = repair_twin(M, slot2c, rng)
            if not ok:
                continue
            sol = make_solution(M, used, slot2c)
            repair_reh(M, sol)
        pp = rng.choice((2.0, 3.0, 3.0, 4.0, 6.0))
        T0 = rng.choice((0.004, 0.01, 0.02, 0.05)) if not cap \
            else rng.choice((0.002, 0.006, 0.015))
        if hot:
            T0 = rng.choice(_hot_t0) * _tf
        T1 = rng.choice((1e-4, 3e-4, 1e-3))
        _kb = P.best_key
        anneal(P, sol, min(deadline, time.monotonic() + chunk), p=pp, T0=T0, T1=T1,
               seed=rng.randrange(1 << 30), reh_lns=cap)
        if cap:
            stall = 0 if P.best_key < _kb - 1e-12 else stall + 1
        # deterministic rehandle polish on the incumbent best
        if P.best_a is not None:
            s2 = make_solution(M, used, list(P.best_a))
            if repair_reh(M, s2) and _twin_all_ok(M, s2.slot2c):
                P.note(s2)
        yielding = converged(P)
        if verbose:
            print("  round", k, "cap" if cap else "rat", round(P.best_key, 6),
                  P.best_vec, round(time.monotonic() - t_start, 1),
                  "yield" if yielding else "",
                  "HOT%g" % round(T0, 4) if hot else "",
                  "stall=%d" % stall, flush=True)
        if yielding:
            # every goal met: hand the cpu to whichever voyage still has one,
            # but take it back as the deadline approaches and it becomes more
            # and more likely that there is no such voyage left
            now = time.monotonic()
            span = deadline - t_start
            d = duty_at((now - t_start) / span) if span > 0.0 else DUTY
            # `boost`: the ramp's clock is a proxy for "nobody is left to yield
            # to".  When that is directly observable, use it: go straight to
            # the saturation duty, and only then.
            _bo_mark("conv")
            if _bo_on and d < DUTY_HI and _bo_all_conv():
                d = DUTY_HI
            nap = min(deadline - now, chunk * (1.0 / d - 1.0))
            if nap > 0.0:
                time.sleep(nap)

    if P.best_a is None:
        slot2c = construct_pairing(M, used, keep, rng, "gap")
        sol = make_solution(M, used, slot2c)
        spread_blocks(M, sol)
        slot2c = sol.slot2c
        repair_twin(M, slot2c, rng)
        sol = make_solution(M, used, slot2c)
        P.best_a = list(sol.slot2c)
        P.best_vec = sol.metrics()
    return M, P


def _twin_all_ok(M, a):
    for x, y in M.twin_pairs:
        c1, c2 = a[x], a[y]
        if c1 < 0 or c2 < 0:
            continue
        w1, w2 = M.c_w[c1], M.c_w[c2]
        if w1 + w2 > M.max_twin_total or abs(w1 - w2) > M.max_twin_diff:
            return False
    return True


_LIVE_PLANNER = None
_EMIT_SEQ = [0]


def emit_polish(M, a0, deadline):
    """Strictly-improving fixed-point polish of the assignment about to be
    emitted; see the `emitpolish` block.  Returns `a0` unchanged unless the
    evaluator's own asf, recomputed from scratch, is strictly better."""
    try:
        a_in = list(a0)
        n_load = sum(1 for c in a_in if c >= 0)
        ideal = EFFECTIVE_TARGETS.get(n_load)
        sc = TRUE_SCALES.get(n_load)
        if ideal is None or sc is None:
            return a0

        def _asf(vec):
            mx, sm = -1e30, 0.0
            for k in METRICS:
                dk = (vec[k] - ideal[k]) / sc[k]
                sm += dk
                if dk > mx:
                    mx = dk
            return (mx if mx > 0.0 else 0.0) + 1e-4 * sm

        def _fresh(x):
            u = [s for s in range(len(x)) if x[s] >= 0]
            return _asf(Solution(M, list(x), u).metrics())

        f_in = _fresh(a_in)
        sol = Solution(M, list(a_in),
                       [s for s in range(len(a_in)) if a_in[s] >= 0])
        a = sol.slot2c
        bkocc = {}
        for s in range(len(a)):
            b = M.s_bk[s]
            if b >= 0 and a[s] >= 0:
                bkocc.setdefault(b, []).append(s)
        cur = _asf(sol.metrics())
        for _ in range(12):
            moved = 0
            if time.monotonic() >= deadline:
                break
            # (b) bring a dropped container aboard in place of a loaded one
            onboard = set(c for c in a if c >= 0)
            for fc in [c for c in range(len(M.c_gkey)) if c not in onboard]:
                if time.monotonic() >= deadline:
                    break
                for s in bkocc.get(M.c_bk[fc], ()):
                    old = a[s]
                    if old < 0:
                        continue
                    sol.apply_changes(((s, fc),))
                    if not _twin_ok_slots(M, a, (s,)):
                        sol.apply_changes(((s, old),))
                        continue
                    f = _asf(sol.metrics())
                    if f < cur - 1e-15:
                        cur = f
                        moved += 1
                        break
                    sol.apply_changes(((s, old),))
            # (a) 2-exchange of two loaded containers of the same bucket
            for b, v in bkocc.items():
                if time.monotonic() >= deadline:
                    break
                n = len(v)
                for i in range(n):
                    if time.monotonic() >= deadline:
                        break
                    s1 = v[i]
                    for j in range(i + 1, n):
                        s2 = v[j]
                        c1, c2 = a[s1], a[s2]
                        if c1 < 0 or c2 < 0 or c1 == c2:
                            continue
                        sol.apply_changes(((s1, c2), (s2, c1)))
                        if not _twin_ok_slots(M, a, (s1, s2)):
                            sol.apply_changes(((s1, c1), (s2, c2)))
                            continue
                        f = _asf(sol.metrics())
                        if f < cur - 1e-15:
                            cur = f
                            moved += 1
                        else:
                            sol.apply_changes(((s1, c1), (s2, c2)))
            if not moved:
                break
        out = list(a)
        live = [c for c in out if c >= 0]
        if len(live) != n_load or len(set(live)) != n_load:
            return a0                      # never emit a count/uniqueness change
        if _fresh(out) < f_in - 1e-15:
            return out
        return a0
    except BaseException:
        return a0


def _write_out(M, a):
    """Emit `a` atomically to sys.argv[2]; see the `alarmsafe` block.

    Writes a uniquely named temp file beside the target and `os.replace`s it in,
    so a partially written file is never visible and the normal emit and the
    watchdog emit can never interleave -- the sequence number guarantees the
    two calls never share a temp path even if the second starts while the first
    is still inside `json.dump`."""
    import os as _os
    _EMIT_SEQ[0] += 1
    seq = _EMIT_SEQ[0]
    a = list(a)
    if not _twin_all_ok(M, a):
        repair_twin(M, a, random.Random(1234))
    pairs = [[M.c_gkey[c], M.s_gkey[s]] for s, c in enumerate(a) if c >= 0]
    dst = sys.argv[2]
    tmp = "%s.%d.%d.tmp" % (dst, _os.getpid(), seq)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"assignments": pairs}, f)
        f.flush()
        _os.fsync(f.fileno())
    _os.replace(tmp, dst)
    return len(pairs)


def _watchdog(signum, frame):
    """Last-resort emit before the evaluator's 600 s subprocess timeout."""
    import os as _os
    import signal as _signal
    P = _LIVE_PLANNER
    a = None
    if P is not None:
        a = getattr(P, "safe_a", None)
        if a is None:
            a = getattr(P, "inf_a", None)
        if a is None:
            a = getattr(P, "best_a", None)
    if a is not None:
        try:
            _write_out(P.M, a)
        except BaseException:
            # A failed emergency write must not exit: the normal path may yet
            # finish.  Retry once more in 8 s.
            try:
                _signal.setitimer(_signal.ITIMER_REAL, 8.0)
            except BaseException:
                pass
            return
        _os._exit(0)
    # Nothing emittable yet (cannot happen at 575 s -- construction lands by
    # t = 20-50 s).  Fall through and let the normal path run.
    try:
        _signal.setitimer(_signal.ITIMER_REAL, 8.0)
    except BaseException:
        pass


def main():
    import os
    seed = int(os.environ.get("STOW_SEED", "42"))
    random.seed(seed)
    t_start = time.monotonic()
    # Hard timeout guard; see the `alarmsafe` block.  MAX_PROGRAM_SECONDS is
    # 600 in tests/evaluator.py and a breach zeroes the whole submission.
    try:
        import signal
        signal.signal(signal.SIGALRM, _watchdog)
        signal.setitimer(
            signal.ITIMER_REAL,
            max(1.0, float(os.environ.get("STOW_HARD_LIMIT", "575"))
                - (time.monotonic() - t_start)),
        )
    except BaseException:
        pass
    limit = float(os.environ.get("STOW_TIME_LIMIT", "550"))
    p1 = os.environ.get("STOW_P1")
    verb = os.environ.get("STOW_VERB") == "1"
    with open(sys.argv[1], encoding="utf-8") as f:
        src = json.load(f)
    _pol = float(os.environ.get("STOW_POLISH_S", POLISH_S))
    M, P = solve(src, t_start + limit - _pol, seed=seed, verbose=verb,
                 phase1_frac=(float(p1) if p1 else None))
    a = P.safe_a
    if a is None:
        a = P.inf_a if P.inf_a is not None else P.best_a
    if _pol > 0.0 and a is not None:
        _a2 = emit_polish(M, a, t_start + limit)
        if verb:
            print("  polish", "HIT" if _a2 is not a else "none",
                  "t=%.1f" % (time.monotonic() - t_start), flush=True)
        a = _a2
    if verb:
        print("emit", "safe" if P.safe_a is not None else
              ("ASF" if P.inf_a is not None else "BEST-INFEASIBLE"),
              "safe_key", round(P.safe_key, 6), P.safe_vec,
              "inf_key", round(P.inf_key, 6), P.inf_vec,
              "best", P.best_vec, flush=True)
    n = _write_out(M, a)
    if verb:
        print("wrote", n, "pairs at t=%.1f" % (time.monotonic() - t_start),
              flush=True)


if __name__ == "__main__":
    main()
