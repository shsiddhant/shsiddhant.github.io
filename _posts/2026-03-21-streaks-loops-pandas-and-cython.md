---

layout: post
title: "Streaks Detection: Loops, Pandas, and a Cython Detour"
categories: [Python, Pandas, Cython]

---

Back in October 2025, I was working on my project [**memory.fm**](https://www.github.com/shsiddhant/memory.fm),
and I needed to solve a simple problem: detect ***listening streaks***, i.e. consecutive listens of same artist, album, or track in a listening history.

It sounds like a natural fit for **pandas**. And to be fair, there is a clean, readable, and idiomatic solution.

## The Pandas Approach

A common solution would look like this:

```python
def streak_gen_pandas(ser: pd.Series, min_length: int = 10):
    data = pd.DataFrame(ser)
    data['start_of_streak'] = ser.ne(ser.shift())
    data['streak_id'] = data.start_of_streak.cumsum()
    data['streak_counter'] = data.groupby('streak_id').cumcount() + 1
    return data[data["streak_counter"] >= min_length]
```

This works by:

- Detecting value changes using shift

- Assigning a group to each streak using `cumsum`

- Counting within each group using `groupby` + `cumcount`

It’s expressive and easy to read.

On my own listening history,  which had around **40k** scrobbles at the time, the above code takes about **15 ms** per run.


## A More Direct Approach

Before looking up pandas solutions, I had written a more direct algorithm:

- Scan the sequence
- Detect where streaks start and end
- Compute streak lengths
- Record streaks above a minimum length


**Create a boolean series by comparing consecutive values.**

```python
def gen_bool_ser(
    original_ser: pd.Series,
    min_length: int = 10
    ) -> pd.Series[bool]:
    """
    Generate a boolean series of comparisons of consecutive values.
    """
    if int(min_length) < 2:
        raise ValueError("Streak Length must be at least 2")
    original_ser = original_ser.dropna()
    ser = orig_ser.eq(original_ser.shift(-1)).iloc[:-1]
    return ser
```

**Search fo streaks**

```python
def loop(
    ser:pd.Series[bool],
    min_length: int = 10
    ):
    """
    Loop to find streaks using the boolean comparison series.
    """
    n = len(ser)
    streaks = []
    start = 0
    stop = 0
    while start < n:
        # Search for streak start.
        # Skip the search region to after the previous streak
        g = (k for k in range(stop + 1, n) if ser.iloc[k] )
        start = next(g, None)
        if start is None: # if no streak is found, stop the loop
            break
        # Search for streak end.
        #Skip the search region to after the streak start
        h = (j for j in range(start, n) if not ser.iloc[j])
        stop = next(h, None)
         # If streak continues to the end of series, stop the loop
        if stop is None:
            if n - start + 1 >= min_length: # check length
                streaks.append([start, n, n - start + 1])
            break
        elif stop-start+1 >= min_length: # check length
            streaks.append([start, stop, stop - start + 1])
    return streaks

def streak_gen(orig_ser: pd.Series, min_length: int = 10):
    return loop(gen_bool_ser(orig_ser, min_length), min_length)
```


This version is conceptually simple and runs in linear time. But performance-wise, it takes about **200 ms** per run, on the same dataset.

That is over **10 times** slower than the pandas solution!

## Why This Was Slow

That was the primary question. At its core, streak detection is simple: find consecutive runs of identical values.

I wanted to understand what pandas could be doing better that made it so much faster. It also hurt my ego a little, so that was another motivation.

The issue, as I realised, wasn't the algorithm, but the execution method. When I looked into the *why* more closely,
I found that **pandas** and **NumPy** actually use **C-compiled** code behind the scenes. 

So that's what it meant when I kept hearing about the *vectorized* methods!

**Key problems**

- Python-level loops

- Repeated `pd.Series.iloc[...]` access

- Python generator expressions

Even though the algorithm is *O(n)*, each iteration carries significant overhead. 


That led me to **Cython** and **C-extensions**, and my current solution.


## My Current Solution

The idea is to move the loop function into a Cython module, leveraging typed memoryviews. That way you avoid the Python overhead.

### Reduce Pandas Overhead

Instead of working directly with pandas objects, I convert the series to a NumPy array of integers and return a comparison boolean int array.

```python
def gen_streaks_bool(series: ArrayLike) -> np.NDArray:
    """
    Returns the boolean integer array of consecutive value comparisons.
    """
    series = np.array(series.factorize()[0])
    bool_ser = np.array((series[1:] == series[:-1]), dtype=np.intc)
    return bool_ser
```

### Move the Loop to Cython

The main speedup came from moving the loop to Cython.

Importantly, the algorithm didn't change, but only now it runs in a C compiled code.

```python
@cython.boundscheck(False)
@cython.wraparound(False)
def streak_gen(
    streak_start: cython.bint[:], # Boolean int array.
    min_length: cython.int,
) -> cython.int[:, :]:
    n: cython.int = streak_start.shape[0]
    start: cython.int = 0
    stop: cython.int = 0
    i: cython.int = 0

    streaks: cython.int[:, :] = np.zeros((n, 3), dtype=np.intc)

    while start < n:
        g = (k for k in range(stop + 1, n) if streak_start[k])
        start = next(g, -1)
        if start == -1:
            break

        h = (j for j in range(start, n) if not streak_start[j])
        stop = next(h, -1)

        if stop == -1:
            if n - start + 1 >= min_length:
                streaks[i, 0] = start
                streaks[i, 1] = n
                streaks[i, 2] = n - start + 1
            i += 1
            break
        elif stop - start + 1 >= min_length:
            streaks[i, 0] = start
            streaks[i, 1] = stop
            streaks[i, 2] = stop - start + 1
            i += 1

    return np.asarray(streaks)[:i, :]
```

This code returns a 2D array of start, stop and streak lengths.

That brings runtime down to about **10 ms** per run!

Ever so slightly faster than the pandas solution.


## Key Insight

Performance might depend more on where your code runs than on what your algorithm is.

- Python loop → slow

- Pandas and NumPy (vectorized) → fast

- Cython (compiled loop) → as fast as the vectorized method

The algorithm stayed essentially the same throughout.


## Notes

This Cython version still uses generator expressions, so it’s not fully optimized. We could further improve performance by: 


1. Using Cython syntax instead of Pure Python syntax
2. Rewriting the loop with pure C-level iteration instead of generators

The pandas solution is still more concise and perfectly reasonable for most use cases.


## Conclusion

It started as a simple annoyance: *Why is the 'idiomatic' Pandas way faster than my direct loop?*

Conceptually, they aren't very different. But digging into the why led me down the rabbit hole of **CPython**.
By using **Cython** to run my original loop in a C-compiled environment, I was able to match the performance of **Pandas** without needing a more clever approach. 

Thus, I realised that while my algorithm was mathematically sound, it was being held back by the overhead of the Python interpreter itself!

A tiny bit of ego and curiosity might lead you to some very interesting things which you'd otherwise likely never discover.
