To run part 1 bilateral filtering:

```bash
python bilateral_filtering.py
```

you can also set different parameters

```bash
--sigma_r 0.08 --sigma_s 50 --epsilon 0.02 --tau 0.8 --kernel 9
```



To run part 2 gradient domain processing:

```bash
python gradient_domain_processing.py
```

you can also set different parameters

```bash
--epsilon 0.001 --N 1000 --sigma 40 --tau_s 1.0 --initialization ambient --boundary ambient
```

