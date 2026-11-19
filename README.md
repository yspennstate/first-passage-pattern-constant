# The exact asymptotics of the waiting time for a first-passage pattern

Yitzchak Shmalo

Let T be the first time at which the last n steps of a simple symmetric random walk form a word that first reaches -h at its last step, and let p be the probability that a given window is such a word. Pitman and Tang asked for the exact asymptotics of the mean E T for h of order sqrt(n), and its order n log n was proved in earlier work of the author. We prove that E T ~ log L / (2 pi p) with L = min(h^2, n^2/h^2) whenever h -> infinity and h/n -> 0, and that T / E T is asymptotically exponential. For h ~ |lambda| sqrt(n) with lambda < 0 this gives E T ~ e^(lambda^2/2) (2 sqrt(2 pi) |lambda|)^(-1) n log n. Say that a walk X matches a simple symmetric walk Z at depth j if X is at -j when Z first reaches -j. For fixed h and for h/n -> delta > 0 the limit of p E T is the reciprocal of the probability that Z is never matched by an independent walk X, conditioned to stay above -h in the first case and with drift -delta in the second. For h = 2 the limit is 3 pi / 8.

- `main.tex`, `main.pdf`: the paper.
- `check.py`: checks the recursion of Proposition 8.1 against the exact mean for small n, the constants rho_h and G_1, and recomputes the derived columns of Table 1 from the values of E T quoted there (Python 3 with numpy and scipy; about a minute). Its output is `check_output.txt`.
