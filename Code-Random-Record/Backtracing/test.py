from BacktracingAlgo import Solutions



if __name__ == "__main__":
    solutions = Solutions()
    result = solutions.permuteUnique([0,1])

    for index, item in enumerate(result):
        print(item)