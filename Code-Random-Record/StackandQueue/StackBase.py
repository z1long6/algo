class Solution:
    # LT.20.有效的括号
    def isValid(self, s: str) -> bool:
        list = []
        for i, item in enumerate(s):
            if item in ['(', '[', '{']:
                list.append(item)
            elif item == ')':
                if len(list) == 0 or list.pop() != '(':
                    return False   
            elif item == ']':
                if len(list) == 0 or list.pop() != '[':
                    return False  
            elif item == '}':
                if len(list) == 0 or list.pop() != '{':
                    return False
        if len(list) != 0:
            return False  
        return True
    # LT.1047.删除字符串中的所有相邻重复项
    def removeDuplicates(self, s: str) -> str:
        stack = []
        for i, item in enumerate(s):
            if len(stack) > 0:
                if stack[len(stack)-1] == item:
                    stack.pop()
                    continue

            stack.append(item)

        return ''.join(stack)