from BacktracingAlgo import Solutions



if __name__ == "__main__":
    solutions = Solutions()
    result = solutions.findSubsequences([1,2,2])

    for index, item in enumerate(result):
        print(item)