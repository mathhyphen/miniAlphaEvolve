"""Mutable candidate for proof search.

Replace `sort_list` with the target function for your conjecture or property.
"""


def sort_list(items):
    result = list(items)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if result[i] > result[j]:
                result[i], result[j] = result[j], result[i]
    return result
