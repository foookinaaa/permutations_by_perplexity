# The best permutation by perplexity
kaggle competition https://www.kaggle.com/competitions/santa-2024/overview

In this competition you have samples with Christmas stories that have words in wrong order, and you need to rearrange this words to get proper story back. 

For get data you need to use this command: 
```commandline
kaggle competitions download -c santa-2024
```
We have 5 rows, each with words from Christmas stories, that we need to rearrange in correct order.
Stories from 10 to 100 words. 
More in EDA.ipynb

##### Baselines
* Brute force - just take all possible combinations of words and score each to find the lowest perplexity. The problem that this approach is too long because even in the first row where we have only 10 words there are 10! = 3628800 possible permutations for this and you need to score each. And the last row has 100! permutations. So, we need to find less time and resource consuming approach.
* Greedy Decoding - on each step choose the word that brings the lowest perplexity from words that we have not used before. It can work long by time because evaluate all candidates on each step, and also it can be local optimum decision, because you choose the optimal word on each step, but the combination of words from previous steps can not to be the most optimal choice.
* Beam Search - same as greedy, but instead of 1 best candidate it chooses k the best candidates on each step. It helps to do lower chances to be in the local optimove, because it's track k strategies on each step. But it is more computationally expensive than greedy decoding, as it evaluates and tracks multiple candidates at each step. Also, the results still can bring to the local optimum, instead global, if k was not enough.
* Simulated Annealing - start with random solution and make small changes from that to find better combination. We take two random words, swap it and hope to get better score. Repeat this some amount of steps. It's faster than brute force and trying to escape local optimum, but still cannot be global optimum. This method also sensitive to hyperparameters.

| Row N |   Raw   |  Random | Brute Force |  Greedy | Beam Search | Simulated | 
|:------|:-------:|--------:|------------:|--------:|------------:|----------:|
| 0     | 3887.90 | 2105.57 |     1375.48 | 1327.97 |      515.99 |    496.23 | 
| 1     | 6068.93 | 2833.36 |           - | 2017.02 |      598.56 |    577.89 |
| 2     | 1118.26 |       - |           - |  967.78 |      526.17 |    394.08 |
| 3     | 1287.11 |       - |           - |  762.59 |      386.46 |    309.32 |
| 4     | 353.25  |       - |           - |  283.85 |      204.45 |    202.86 | 
| 5     | 354.64  |       - |           - |  128.94 |       82.93 |     82.93 |
| Mean  | 2178.35 |       - |           - |  914.70 |      386.02 |    343.88 | 

The best mean perplexity from baselines is 343.88, that we can take as a baseline for experiments.

### Results
I take IBIS approach and modify it for my task, after that added simulated annealing approach, and repeated it for 3 iterations:

| Row N | IBIS (1) | IBIS + SA (2) | IBIS + SA (3) | 
|:------|:--------:|--------------:|--------------:|
| 0     |  496.23  |        496.23 |        496.23 |  
| 1     |  549.27  |        536.55 |        511.98 | 
| 2     |  308.11  |        308.11 |        303.34 |
| 3     |  266.65  |        236.24 |        242.79 |
| 4     |  126.45  |        126.45 |        121.13 |
| 5     |  50.89   |         49.62 |         49.62 |
| Mean  |  299.60  |        292.20 |        282.97 |

The best score that I can achieve for 3 iterations is 282.97, which is much better, than baseline 343.88. 
If try more iterations that it can be improved even more, because I get global optimum only for 0 row. For other rows it's still local optimums based on kaggle leaderboard results.

The IBIS approach works better and faster than beam search with combination of simulated annealing, but it still can stay in the local optimums.
