from BacktracingAlgo import Solutions



if __name__ == "__main__":
    solutions = Solutions()
    result = solutions.combinationSum([2,3,6,7], 7)

    for index, item in enumerate(result):
        print(item)