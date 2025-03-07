'''
    use two stacks to implement a queue
'''
class MyQueue:
    def __init__(self):
        self.list1 = []
        self.list2 = []


    def push(self, x: int) -> None:
        self.list1.append(x)

    def pop(self) -> int:
        if len(self.list2) == 0 and len(self.list1) != 0:
            # add numbers in list1 to list2
            j = len(self.list1)-1
            while j >= 0:
                self.list2.append(self.list1[j])
                j -= 1
            self.list1 = []
        return self.list2.pop()
        

    def peek(self) -> int:
        if len(self.list2) == 0 and len(self.list1) != 0:
            # add numbers in list1 to list2
            j = len(self.list1)-1
            while j >= 0:
                self.list2.append(self.list1[j])
                j -= 1
            self.list1 = []

        if len(self.list2) != 0:
            return self.list2[len(self.list2)-1]
        else:
            return None

    def empty(self) -> bool:
        if len(self.list1) == 0 and len(self.list2) == 0:
            return True
        else:
            return False
