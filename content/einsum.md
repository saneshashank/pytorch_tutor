# Einsum (Einstein Summation) — Reference

Kept separate from the tutorial so the core flow stays lightweight.
Examples:

```python
# dot
torch.einsum('i,i->', a, b)

# matvec
torch.einsum('ij,j->i', A, x)

# matmul
torch.einsum('ik,kj->ij', A, B)

# outer
torch.einsum('i,j->ij', a, b)

# batch matmul
torch.einsum('bik,bkj->bij', A, B)
```
