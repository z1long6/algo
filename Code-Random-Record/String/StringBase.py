class Solution:
    # LT.344.反转字符串
    def reverseString(self, s: list[str]) -> None:
        j = len(s) - 1
        for i in range(len(s) // 2):
            s[i], s[j] = s[j], s[i]
            j -= 1
    
    # LT.541.反转字符串2
    def reverseString(self, s: list[str], k: int) -> None:
        p = 0
        while p < len(s):
            p2 = p + k
            s = s[:p] + s[p:p2][::-1] + s[p2:]
            p += 2*k
        return s
    # LT.151.翻转字符串里的单词
    def reverseWords(self, s: str) -> str:
        return ' '.join(s.split(' ')[::-1])
    # LT.459.重复的子字符串
    def repeatedSubstringPattern(self, s: str) -> bool:
        # math
        return s in (s + s)[1:-1]
    

