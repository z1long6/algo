import TreeNode
from queue import PriorityQueue
import heapq
'''
    implement a heap use compelete binary tree
'''
class MyHeap:
    # init
    def __init__(self):
        self.heap = []

    # left
    def left(self, i) -> int:
        return 2*i + 1
    
    # right
    def right(self, i) -> int:
        return 2*i + 2
    
    # parent
    def parent(self, i) -> int:
        return (i - 1) // 2
    
    # peek
    def peek(self, i) -> int:
        return self.heap[0]
    
    # push
    def push(self, val):
        self.heap.append(val)
        self.sift_up(len(self.heap) - 1)
    
    # pop
    def pop(self):
        if self.heap is None:
            raise IndexError("heap is None")

        # swap root and last
        self.swap(0, len(self.heap)-1)

        val = self.heap.pop()

        # sift top-down
        self.sift_top_down(0)

        return val



    # swap
    def swap(self, p: int, q: int):
        self.heap[p], self.heap[q] = self.heap[q], self.heap[p]

    # sift up
    def sift_up(self, i):
        # sift from bottom to top
        while True:
            p = self.parent()
            if p < 0 or self.heap[i] <= self.heap[p]:
                break
            self.swap(i, p)
            i = p
    
    # sift top down
    def sift_top_down(self, i: int):
        # sift from top to bottom
        while True:
            left, right, max = self.left(i), self.right(i), i
            if left < len(self.heap) and self.heap[left] > self.heap[max]:
                max = left
            if right < len(self.heap) and self.heap[right] > self.heap[max]:
                max = right

            if max == i:
                break

            self.swap(i, max)
            i = max
    
if __name__ == "__main__":
    pass
