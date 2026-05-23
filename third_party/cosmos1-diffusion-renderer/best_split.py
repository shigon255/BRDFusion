from functools import lru_cache

def best_equal_split(total_frames, acceptable_lengths):
    acceptable_lengths = sorted(acceptable_lengths, reverse=True)

    # First try perfect uniform split
    for l in acceptable_lengths:
        if total_frames % l == 0:
            return [l] * (total_frames // l)

    # If no uniform split, find minimal-split combination
    @lru_cache(maxsize=None)
    def backtrack(remaining):
        if remaining == 0:
            return []
        best = None
        for l in acceptable_lengths:
            if l <= remaining:
                sub = backtrack(remaining - l)
                if sub is not None:
                    candidate = [l] + sub
                    # Prefer fewer splits, and more equal segments
                    if best is None or (
                        len(candidate) < len(best) or
                        (len(candidate) == len(best) and max(candidate) - min(candidate) < max(best) - min(best))
                    ):
                        best = candidate
        return best

    return backtrack(total_frames)

acceptable_lengths = [8 * n + 1 for n in range(16)] + [10, 117]
total_frames = 198

combo = best_equal_split(total_frames, tuple(acceptable_lengths))
print("Best split:", combo)
print("Total frames:", sum(combo))
print("Number of splits:", len(combo))