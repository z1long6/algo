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