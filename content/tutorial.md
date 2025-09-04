# Tensor Basics in PyTorch — Hands‑on Tutorial

This tutorial teaches everyday tensor skills: create tensors, index/slice, reshape, add/remove dims, permute, repeat vs expand, broadcasting, and a short intro to **einops** (not einsum).

## What is a Tensor?
A tensor is an N‑D array with a shape and dtype. Rank (`ndim`) is the number of axes.  
Common shapes: vectors `[N]`, matrices `[M,N]`, and images `[N, C, H, W]` (batch, channels, height, width).  
Tensors can be on CPU or GPU; floats default to `float32`.

**Practice pool ideas:** creating different shapes like `[3,2]`, inspecting `dtype`, `device`, `ndim`.

## Indexing & Slicing
Use `t[i, j]` for a single element. `:` selects all along that axis.  
Negative indices count from the end: `-1` is the last index.  
Slicing usually returns **views**, which are cheap—important for performance.

**Practice pool ideas:** top‑left blocks, last columns, ranges like `1:3`.

## Reshape vs View
`reshape` changes the view of the same storage when possible. `view` requires contiguous tensors.  
Use `-1` to infer a dimension. Flattening with `reshape(-1)` is common before an MLP.

**Practice pool ideas:** flatten a `[2,3,2]` to `[12]`, then reshape to `[3,4]`.

## Unsqueeze / Squeeze
`unsqueeze(dim)` inserts a size‑1 dimension; `squeeze(dim)` removes it.  
This is handy to add batch or channel dimensions or to prep for broadcasting.

**Practice pool ideas:** turn `[3]` into `[1,3]` or `[3,1]`, then remove again.

## Permute
`permute` reorders axes. In vision you may convert between `[N,C,H,W]` and `[N,H,W,C]`.  
Non‑contiguous results may need `.contiguous()` before `.view()`.

**Practice pool ideas:** convert a batch from NCHW to NHWC.

## Repeat vs Expand
`expand` creates broadcasted views along size‑1 axes (no memory copy).  
`repeat` tiles data (allocates memory) but works for more shapes.

**Practice pool ideas:** from `[3,1]` to `[3,4]` using `repeat` or `expand` appropriately.

## Broadcasting
PyTorch matches sizes from the right; `1` can expand to match the other tensor.  
Use `unsqueeze` to add size‑1 dims so shapes align cleanly.

**Practice pool ideas:** add a row/column vector to a 2‑D matrix by unsqueezing.

## A taste of einops
`einops.rearrange` expresses reshape/transpose as readable patterns.  
Great for complex pipelines where numbers get hard to track.

**Practice pool ideas:** flatten last two dims, swap positions, or create grouped channels.
