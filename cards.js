export const CARDS = [
    {
      id:1, lc:1, title:"Two Sum", topic:"Arrays", difficulty:"Easy",
      question:"Given an array of integers nums and an integer target, return indices of the two numbers that add up to target. Exactly one solution exists.",
      hint:"Use a hash map to store complements. O(n) time.",
      explain:"Iterate once through the array. For each number, calculate its complement (target - num) and check if it's already in a hash map. If yes, we're done. Otherwise, store the current number's index. This avoids the O(n²) brute force by trading space for time.",
      timeC:"O(n)", spaceC:"O(n)",
      code:`<span class="kw">def</span> <span class="fn">twoSum</span>(nums, target):
      seen = {}
      <span class="kw">for</span> i, n <span class="kw">in</span> <span class="fn">enumerate</span>(nums):
          diff = target - n
          <span class="kw">if</span> diff <span class="kw">in</span> seen:
              <span class="kw">return</span> [seen[diff], i]
          seen[n] = i`
    },
    {
      id:2, lc:121, title:"Best Time to Buy and Sell Stock", topic:"Arrays", difficulty:"Easy",
      question:"Given prices[i] as the stock price on day i, find the maximum profit from buying low and selling on a later day.",
      hint:"Track min price seen so far, update max profit each step.",
      explain:"Single pass greedy. Keep a running minimum price seen so far. At each step, compute profit if we sold today (price - min_price) and update the global max. We never need to look back because selling after buying is guaranteed by tracking the minimum before the current index.",
      timeC:"O(n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">maxProfit</span>(prices):
      min_p, max_p = <span class="fn">float</span>(<span class="st">'inf'</span>), <span class="nm">0</span>
      <span class="kw">for</span> p <span class="kw">in</span> prices:
          min_p = <span class="fn">min</span>(min_p, p)
          max_p = <span class="fn">max</span>(max_p, p - min_p)
      <span class="kw">return</span> max_p`
    },
    {
      id:3, lc:217, title:"Contains Duplicate", topic:"Arrays", difficulty:"Easy",
      question:"Given an integer array nums, return true if any value appears at least twice, false if all elements are distinct.",
      hint:"Compare length of set vs array.",
      explain:"Converting to a set removes duplicates. If the set is smaller than the original array, at least one duplicate existed. Alternatively, add elements to a set one by one and return True the moment an element already exists.",
      timeC:"O(n)", spaceC:"O(n)",
      code:`<span class="kw">def</span> <span class="fn">containsDuplicate</span>(nums):
      <span class="kw">return</span> <span class="fn">len</span>(nums) != <span class="fn">len</span>(<span class="fn">set</span>(nums))`
    },
    {
      id:4, lc:238, title:"Product of Array Except Self", topic:"Arrays", difficulty:"Medium",
      question:"Return answer[i] = product of all nums except nums[i]. Must run in O(n) without division.",
      hint:"Two passes: prefix products left-to-right, then suffix right-to-left.",
      explain:"Two-pass prefix/suffix trick. First pass fills result[i] with the product of everything to the LEFT. Second pass multiplies in the product of everything to the RIGHT using a running suffix variable. No division needed, and we reuse the result array so space is O(1) extra.",
      timeC:"O(n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">productExceptSelf</span>(nums):
      n = <span class="fn">len</span>(nums)
      res = [<span class="nm">1</span>] * n
      pre = <span class="nm">1</span>
      <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(n):
          res[i] = pre; pre *= nums[i]
      suf = <span class="nm">1</span>
      <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(n-<span class="nm">1</span>,-<span class="nm">1</span>,-<span class="nm">1</span>):
          res[i] *= suf; suf *= nums[i]
      <span class="kw">return</span> res`
    },
    {
      id:5, lc:53, title:"Maximum Subarray", topic:"Arrays", difficulty:"Medium",
      question:"Find the contiguous subarray with the largest sum and return its sum. (Kadane's Algorithm)",
      hint:"Kadane's: keep running sum, reset if negative.",
      explain:"Kadane's algorithm: at each position, decide whether to extend the existing subarray or start fresh. If the running sum becomes negative, it only hurts future elements, so reset to the current element. Track the global best at each step.",
      timeC:"O(n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">maxSubArray</span>(nums):
      cur = best = nums[<span class="nm">0</span>]
      <span class="kw">for</span> n <span class="kw">in</span> nums[<span class="nm">1</span>:]:
          cur = <span class="fn">max</span>(n, cur + n)
          best = <span class="fn">max</span>(best, cur)
      <span class="kw">return</span> best`
    },
    {
      id:6, lc:152, title:"Maximum Product Subarray", topic:"Arrays", difficulty:"Medium",
      question:"Find the contiguous subarray with the largest product and return the product.",
      hint:"Track both max and min (negatives can flip). Update result each step.",
      explain:"Unlike sum, a negative number can flip the max to min and vice versa. So we track both the running maximum AND minimum. At each step we compute all three candidates (n alone, max*n, min*n) because multiplying a large negative by another negative gives a large positive.",
      timeC:"O(n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">maxProduct</span>(nums):
      res = mx = mn = nums[<span class="nm">0</span>]
      <span class="kw">for</span> n <span class="kw">in</span> nums[<span class="nm">1</span>:]:
          mx, mn = (<span class="fn">max</span>(n,mx*n,mn*n), <span class="fn">min</span>(n,mx*n,mn*n))
          res = <span class="fn">max</span>(res, mx)
      <span class="kw">return</span> res`
    },
    {
      id:7, lc:153, title:"Find Minimum in Rotated Sorted Array", topic:"Arrays", difficulty:"Medium",
      question:"Find the minimum element in a rotated sorted array in O(log n) time.",
      hint:"Binary search: compare mid to right boundary.",
      explain:"The minimum is where the rotation starts. Binary search: if nums[mid] > nums[right], the min is in the right half (go right). Otherwise it's in the left half including mid (go left). This halves the search space each step.",
      timeC:"O(log n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">findMin</span>(nums):
      l, r = <span class="nm">0</span>, <span class="fn">len</span>(nums)-<span class="nm">1</span>
      <span class="kw">while</span> l < r:
          m = (l+r)//<span class="nm">2</span>
          <span class="kw">if</span> nums[m] > nums[r]: l = m+<span class="nm">1</span>
          <span class="kw">else</span>: r = m
      <span class="kw">return</span> nums[l]`
    },
    {
      id:8, lc:33, title:"Search in Rotated Sorted Array", topic:"Arrays", difficulty:"Medium",
      question:"Search target in a rotated sorted array in O(log n). Return index or -1.",
      hint:"Binary search with extra condition to determine which half is sorted.",
      explain:"At least one half of the array is always sorted. Check which half: if left half is sorted AND target is in that range, search left; otherwise search right. Repeat. This preserves O(log n) despite the rotation.",
      timeC:"O(log n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">search</span>(nums, target):
      l, r = <span class="nm">0</span>, <span class="fn">len</span>(nums)-<span class="nm">1</span>
      <span class="kw">while</span> l <= r:
          m = (l+r)//<span class="nm">2</span>
          <span class="kw">if</span> nums[m]==target: <span class="kw">return</span> m
          <span class="kw">if</span> nums[l]<=nums[m]:
              <span class="kw">if</span> nums[l]<=target<nums[m]: r=m-<span class="nm">1</span>
              <span class="kw">else</span>: l=m+<span class="nm">1</span>
          <span class="kw">else</span>:
              <span class="kw">if</span> nums[m]<target<=nums[r]: l=m+<span class="nm">1</span>
              <span class="kw">else</span>: r=m-<span class="nm">1</span>
      <span class="kw">return</span> -<span class="nm">1</span>`
    },
    {
      id:9, lc:15, title:"3Sum", topic:"Arrays", difficulty:"Medium",
      question:"Return all unique triplets [a,b,c] from nums such that a + b + c == 0. No duplicate triplets.",
      hint:"Sort, fix one element, then use two pointers for the rest.",
      explain:"Sort the array. For each index i, use two pointers (l=i+1, r=end) to find pairs summing to -nums[i]. If sum too small, move l right; too large, move r left. Skip duplicates by advancing past repeated values at each level.",
      timeC:"O(n²)", spaceC:"O(n)",
      code:`<span class="kw">def</span> <span class="fn">threeSum</span>(nums):
      nums.sort(); res = []
      <span class="kw">for</span> i, n <span class="kw">in</span> <span class="fn">enumerate</span>(nums):
          <span class="kw">if</span> i><span class="nm">0</span> <span class="kw">and</span> n==nums[i-<span class="nm">1</span>]: <span class="kw">continue</span>
          l, r = i+<span class="nm">1</span>, <span class="fn">len</span>(nums)-<span class="nm">1</span>
          <span class="kw">while</span> l<r:
              s = n+nums[l]+nums[r]
              <span class="kw">if</span> s==<span class="nm">0</span>:
                  res.append([n,nums[l],nums[r]]); l+=<span class="nm">1</span>
                  <span class="kw">while</span> l<r <span class="kw">and</span> nums[l]==nums[l-<span class="nm">1</span>]: l+=<span class="nm">1</span>
              <span class="kw">elif</span> s<<span class="nm">0</span>: l+=<span class="nm">1</span>
              <span class="kw">else</span>: r-=<span class="nm">1</span>
      <span class="kw">return</span> res`
    },
    {
      id:10, lc:11, title:"Container With Most Water", topic:"Arrays", difficulty:"Medium",
      question:"Find two lines that form a container holding the most water.",
      hint:"Two pointers from both ends. Move the shorter side inward.",
      explain:"Two pointers start at opposite ends. Area = min(height[l], height[r]) * (r - l). Moving the taller side inward can only decrease width without increasing height, so we always move the SHORTER side to have any chance of improvement.",
      timeC:"O(n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">maxArea</span>(height):
      l, r = <span class="nm">0</span>, <span class="fn">len</span>(height)-<span class="nm">1</span>
      res = <span class="nm">0</span>
      <span class="kw">while</span> l<r:
          res = <span class="fn">max</span>(res, <span class="fn">min</span>(height[l],height[r])*(r-l))
          <span class="kw">if</span> height[l]<height[r]: l+=<span class="nm">1</span>
          <span class="kw">else</span>: r-=<span class="nm">1</span>
      <span class="kw">return</span> res`
    },
    {
      id:11, lc:42, title:"Trapping Rain Water", topic:"Arrays", difficulty:"Hard",
      question:"Compute how much water can be trapped given an elevation map.",
      hint:"Two pointers. Water at i = min(maxL, maxR) - height[i]. Process from the shorter side.",
      explain:"Water trapped at position i = min(max_left, max_right) - height[i]. Two-pointer approach: process the side with the smaller max first (since it's the bottleneck). Maintain left/right max as you go, computing trapped water in O(1) per step.",
      timeC:"O(n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">trap</span>(height):
      l, r = <span class="nm">0</span>, <span class="fn">len</span>(height)-<span class="nm">1</span>
      ml, mr = height[l], height[r]
      res = <span class="nm">0</span>
      <span class="kw">while</span> l<r:
          <span class="kw">if</span> ml<=mr:
              l+=<span class="nm">1</span>; ml=<span class="fn">max</span>(ml,height[l]); res+=ml-height[l]
          <span class="kw">else</span>:
              r-=<span class="nm">1</span>; mr=<span class="fn">max</span>(mr,height[r]); res+=mr-height[r]
      <span class="kw">return</span> res`
    },
    {
      id:12, lc:3, title:"Longest Substring Without Repeating Characters", topic:"Sliding Window", difficulty:"Medium",
      question:"Find the length of the longest substring without repeating characters.",
      hint:"Sliding window with a set. Shrink from left when duplicate found.",
      explain:"Sliding window: expand right by adding s[r] to a set. If it already exists, shrink from the left (remove s[l], advance l) until the duplicate is gone. Window [l, r] always has unique characters. Track the max window size.",
      timeC:"O(n)", spaceC:"O(min(n,a))",
      code:`<span class="kw">def</span> <span class="fn">lengthOfLongestSubstring</span>(s):
      seen = <span class="fn">set</span>()
      l = res = <span class="nm">0</span>
      <span class="kw">for</span> r, c <span class="kw">in</span> <span class="fn">enumerate</span>(s):
          <span class="kw">while</span> c <span class="kw">in</span> seen:
              seen.discard(s[l]); l+=<span class="nm">1</span>
          seen.add(c)
          res = <span class="fn">max</span>(res, r-l+<span class="nm">1</span>)
      <span class="kw">return</span> res`
    },
    {
      id:13, lc:424, title:"Longest Repeating Character Replacement", topic:"Sliding Window", difficulty:"Medium",
      question:"Replace at most k characters to get the longest substring of one repeated letter.",
      hint:"Sliding window. Window valid if (window_len - max_freq) <= k.",
      explain:"Keep a window [l, r] and a frequency map. The window is valid when (window size - count of most frequent char) <= k — meaning we only need k replacements. If invalid, shrink left. We never shrink the window below the best seen, giving an efficient monotonically growing window.",
      timeC:"O(n)", spaceC:"O(26)",
      code:`<span class="kw">from</span> collections <span class="kw">import</span> defaultdict
  <span class="kw">def</span> <span class="fn">characterReplacement</span>(s, k):
      count = defaultdict(<span class="fn">int</span>)
      l = mx = res = <span class="nm">0</span>
      <span class="kw">for</span> r, c <span class="kw">in</span> <span class="fn">enumerate</span>(s):
          count[c]+=<span class="nm">1</span>; mx=<span class="fn">max</span>(mx,count[c])
          <span class="kw">while</span> (r-l+<span class="nm">1</span>)-mx>k:
              count[s[l]]-=<span class="nm">1</span>; l+=<span class="nm">1</span>
          res=<span class="fn">max</span>(res,r-l+<span class="nm">1</span>)
      <span class="kw">return</span> res`
    },
    {
      id:14, lc:76, title:"Minimum Window Substring", topic:"Sliding Window", difficulty:"Hard",
      question:"Return the minimum window substring of s that contains all characters of t.",
      hint:"Sliding window with two freq maps. Expand right, shrink left when valid.",
      explain:"Use two counters: need (freq of t) and have (freq in current window). Track how many unique chars in t are fully satisfied (formed). Expand right to satisfy all chars, then contract left to minimize window while valid. Record the smallest valid window found.",
      timeC:"O(n+m)", spaceC:"O(n+m)",
      code:`<span class="kw">from</span> collections <span class="kw">import</span> Counter
  <span class="kw">def</span> <span class="fn">minWindow</span>(s, t):
      need,have = Counter(t),{}
      formed,req = <span class="nm">0</span>,<span class="fn">len</span>(need)
      l,res,rlen = <span class="nm">0</span>,[-<span class="nm">1</span>,-<span class="nm">1</span>],<span class="fn">float</span>(<span class="st">'inf'</span>)
      <span class="kw">for</span> r,c <span class="kw">in</span> <span class="fn">enumerate</span>(s):
          have[c]=have.get(c,<span class="nm">0</span>)+<span class="nm">1</span>
          <span class="kw">if</span> c <span class="kw">in</span> need <span class="kw">and</span> have[c]==need[c]: formed+=<span class="nm">1</span>
          <span class="kw">while</span> formed==req:
              <span class="kw">if</span> r-l+<span class="nm">1</span><rlen: res,rlen=[l,r],r-l+<span class="nm">1</span>
              have[s[l]]-=<span class="nm">1</span>
              <span class="kw">if</span> s[l] <span class="kw">in</span> need <span class="kw">and</span> have[s[l]]<need[s[l]]: formed-=<span class="nm">1</span>
              l+=<span class="nm">1</span>
      <span class="kw">return</span> s[res[<span class="nm">0</span>]:res[<span class="nm">1</span>]+<span class="nm">1</span>] <span class="kw">if</span> rlen!=<span class="fn">float</span>(<span class="st">'inf'</span>) <span class="kw">else</span> <span class="st">""</span>`
    },
    {
      id:15, lc:242, title:"Valid Anagram", topic:"Arrays", difficulty:"Easy",
      question:"Return true if t is an anagram of s (same characters, same counts).",
      hint:"Compare character frequency counts.",
      explain:"Two strings are anagrams iff they have identical character frequency distributions. Python's Counter builds a frequency dict. Comparing two Counters checks every character and count — O(n) time. Could also sort both strings and compare, but that's O(n log n).",
      timeC:"O(n)", spaceC:"O(1)",
      code:`<span class="kw">from</span> collections <span class="kw">import</span> Counter
  <span class="kw">def</span> <span class="fn">isAnagram</span>(s, t):
      <span class="kw">return</span> Counter(s) == Counter(t)`
    },
    {
      id:16, lc:49, title:"Group Anagrams", topic:"Arrays", difficulty:"Medium",
      question:"Group an array of strings by anagram family.",
      hint:"Sort each word as the hash key. Group by sorted key.",
      explain:"Anagrams share the same sorted character sequence. Use a defaultdict of lists; the key is tuple(sorted(word)). This groups all anagrams together in O(n * k log k) where k is the max word length.",
      timeC:"O(nk log k)", spaceC:"O(nk)",
      code:`<span class="kw">from</span> collections <span class="kw">import</span> defaultdict
  <span class="kw">def</span> <span class="fn">groupAnagrams</span>(strs):
      d = defaultdict(<span class="fn">list</span>)
      <span class="kw">for</span> s <span class="kw">in</span> strs:
          d[<span class="fn">tuple</span>(<span class="fn">sorted</span>(s))].append(s)
      <span class="kw">return</span> <span class="fn">list</span>(d.values())`
    },
    {
      id:17, lc:20, title:"Valid Parentheses", topic:"Arrays", difficulty:"Easy",
      question:"Determine if a string of brackets is properly opened and closed in correct order.",
      hint:"Use a stack. Push open brackets, pop and match close brackets.",
      explain:"A stack models the nesting naturally. Push every opening bracket. When a closing bracket is seen, pop from the stack and verify it matches. If the stack is empty at the end and no mismatch occurred, the string is valid.",
      timeC:"O(n)", spaceC:"O(n)",
      code:`<span class="kw">def</span> <span class="fn">isValid</span>(s):
      stack = []
      match = {<span class="st">')'</span>:<span class="st">'('</span>,<span class="st">'}'</span>:<span class="st">'{'</span>,<span class="st">']'</span>:<span class="st">'['</span>}
      <span class="kw">for</span> c <span class="kw">in</span> s:
          <span class="kw">if</span> c <span class="kw">in</span> match:
              <span class="kw">if not</span> stack <span class="kw">or</span> stack[-<span class="nm">1</span>]!=match[c]: <span class="kw">return False</span>
              stack.pop()
          <span class="kw">else</span>: stack.append(c)
      <span class="kw">return not</span> stack`
    },
    {
      id:18, lc:125, title:"Valid Palindrome", topic:"Arrays", difficulty:"Easy",
      question:"Ignoring non-alphanumeric chars and case, return true if s is a palindrome.",
      hint:"Filter alphanumerics, lowercase, then compare with reverse.",
      explain:"Filter out everything that isn't a letter or digit, lowercase it, then check if the result equals its reverse. Python list slicing with [::-1] does the reversal cleanly.",
      timeC:"O(n)", spaceC:"O(n)",
      code:`<span class="kw">def</span> <span class="fn">isPalindrome</span>(s):
      s=[c.lower() <span class="kw">for</span> c <span class="kw">in</span> s <span class="kw">if</span> c.isalnum()]
      <span class="kw">return</span> s==s[::-<span class="nm">1</span>]`
    },
    {
      id:19, lc:56, title:"Merge Intervals", topic:"Arrays", difficulty:"Medium",
      question:"Given a list of intervals, merge all overlapping intervals.",
      hint:"Sort by start. Merge if current start <= last end.",
      explain:"Sort by start time. Iterate: if the current interval's start is within the last merged interval (start <= res[-1][1]), merge by extending the end. Otherwise, append as a new interval. Sorting guarantees we only need to look at the previous interval.",
      timeC:"O(n log n)", spaceC:"O(n)",
      code:`<span class="kw">def</span> <span class="fn">merge</span>(intervals):
      intervals.sort(); res=[intervals[<span class="nm">0</span>]]
      <span class="kw">for</span> s,e <span class="kw">in</span> intervals[<span class="nm">1</span>:]:
          <span class="kw">if</span> s<=res[-<span class="nm">1</span>][<span class="nm">1</span>]: res[-<span class="nm">1</span>][<span class="nm">1</span>]=<span class="fn">max</span>(res[-<span class="nm">1</span>][<span class="nm">1</span>],e)
          <span class="kw">else</span>: res.append([s,e])
      <span class="kw">return</span> res`
    },
    {
      id:20, lc:48, title:"Rotate Image", topic:"Arrays", difficulty:"Medium",
      question:"Rotate an n×n matrix 90° clockwise in-place.",
      hint:"Transpose the matrix, then reverse each row.",
      explain:"90° clockwise rotation = transpose + reverse each row. Transpose swaps matrix[i][j] with matrix[j][i] (mirror along main diagonal). Reversing each row then gives the clockwise rotation. Both operations are in-place.",
      timeC:"O(n²)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">rotate</span>(matrix):
      n=<span class="fn">len</span>(matrix)
      <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(n):
          <span class="kw">for</span> j <span class="kw">in</span> <span class="fn">range</span>(i+<span class="nm">1</span>,n):
              matrix[i][j],matrix[j][i]=matrix[j][i],matrix[i][j]
      <span class="kw">for</span> row <span class="kw">in</span> matrix: row.reverse()`
    },
    {
      id:21, lc:54, title:"Spiral Matrix", topic:"Arrays", difficulty:"Medium",
      question:"Return all elements of an m×n matrix in spiral order.",
      hint:"Peel outer layers. Shrink boundaries (top, bottom, left, right) inward.",
      explain:"Four-boundary approach: maintain top, bottom, left, right pointers. Each loop iteration peels one full layer: go right across top, down the right side, left across bottom, up the left side. Shrink the corresponding boundary after each traversal.",
      timeC:"O(m·n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">spiralOrder</span>(matrix):
      res=[]
      t,b=<span class="nm">0</span>,<span class="fn">len</span>(matrix)-<span class="nm">1</span>
      l,r=<span class="nm">0</span>,<span class="fn">len</span>(matrix[<span class="nm">0</span>])-<span class="nm">1</span>
      <span class="kw">while</span> t<=b <span class="kw">and</span> l<=r:
          <span class="kw">for</span> c <span class="kw">in</span> <span class="fn">range</span>(l,r+<span class="nm">1</span>): res.append(matrix[t][c])
          t+=<span class="nm">1</span>
          <span class="kw">for</span> row <span class="kw">in</span> <span class="fn">range</span>(t,b+<span class="nm">1</span>): res.append(matrix[row][r])
          r-=<span class="nm">1</span>
          <span class="kw">if</span> t<=b:
              <span class="kw">for</span> c <span class="kw">in</span> <span class="fn">range</span>(r,l-<span class="nm">1</span>,-<span class="nm">1</span>): res.append(matrix[b][c])
              b-=<span class="nm">1</span>
          <span class="kw">if</span> l<=r:
              <span class="kw">for</span> row <span class="kw">in</span> <span class="fn">range</span>(b,t-<span class="nm">1</span>,-<span class="nm">1</span>): res.append(matrix[row][l])
              l+=<span class="nm">1</span>
      <span class="kw">return</span> res`
    },
    {
      id:22, lc:73, title:"Set Matrix Zeroes", topic:"Arrays", difficulty:"Medium",
      question:"If a cell is 0, set its entire row and column to 0 in-place.",
      hint:"Use first row and column as markers.",
      explain:"Two-pass approach with O(1) extra space. Use the first row/column as flags. First scan all cells: if matrix[i][j]==0, set matrix[i][0]=0 and matrix[0][j]=0. Then use those flags to zero out rows/cols. Handle first row/col separately.",
      timeC:"O(m·n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">setZeroes</span>(matrix):
      rows,cols=<span class="fn">set</span>(),<span class="fn">set</span>()
      <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(matrix)):
          <span class="kw">for</span> j <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(matrix[<span class="nm">0</span>])):
              <span class="kw">if</span> matrix[i][j]==<span class="nm">0</span>: rows.add(i);cols.add(j)
      <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(matrix)):
          <span class="kw">for</span> j <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(matrix[<span class="nm">0</span>])):
              <span class="kw">if</span> i <span class="kw">in</span> rows <span class="kw">or</span> j <span class="kw">in</span> cols: matrix[i][j]=<span class="nm">0</span>`
    },
    {
      id:23, lc:79, title:"Word Search", topic:"Backtracking", difficulty:"Medium",
      question:"Return true if a word exists in a 2D character grid using adjacent (not reused) cells.",
      hint:"DFS/backtracking from each cell. Mark visited, unmark on backtrack.",
      explain:"Try starting DFS from every cell. If board[i][j] matches word[k], mark it visited (temporarily '#'), recurse in 4 directions for word[k+1], then restore the cell (backtrack). Base case: k == len(word) means we found the full word.",
      timeC:"O(m·n·4^L)", spaceC:"O(L)",
      code:`<span class="kw">def</span> <span class="fn">exist</span>(board, word):
      m,n=<span class="fn">len</span>(board),<span class="fn">len</span>(board[<span class="nm">0</span>])
      <span class="kw">def</span> <span class="fn">dfs</span>(i,j,k):
          <span class="kw">if</span> k==<span class="fn">len</span>(word): <span class="kw">return True</span>
          <span class="kw">if not</span>(<span class="nm">0</span><=i<m <span class="kw">and</span> <span class="nm">0</span><=j<n) <span class="kw">or</span> board[i][j]!=word[k]: <span class="kw">return False</span>
          tmp,board[i][j]=board[i][j],<span class="st">'#'</span>
          res=<span class="kw">any</span>(<span class="fn">dfs</span>(i+di,j+dj,k+<span class="nm">1</span>) <span class="kw">for</span> di,dj <span class="kw">in</span> [(<span class="nm">0</span>,<span class="nm">1</span>),(<span class="nm">0</span>,-<span class="nm">1</span>),(<span class="nm">1</span>,<span class="nm">0</span>),(-<span class="nm">1</span>,<span class="nm">0</span>)])
          board[i][j]=tmp; <span class="kw">return</span> res
      <span class="kw">return any</span>(<span class="fn">dfs</span>(i,j,<span class="nm">0</span>) <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(m) <span class="kw">for</span> j <span class="kw">in</span> <span class="fn">range</span>(n))`
    },
    {
      id:24, lc:704, title:"Binary Search", topic:"Binary Search", difficulty:"Easy",
      question:"Search for target in a sorted array. Return its index or -1.",
      hint:"Classic binary search. Maintain l, r, compute mid = (l+r)//2.",
      explain:"Canonical binary search. Maintain l and r as the inclusive search range. Compute mid = (l+r)//2. If nums[mid] == target, found. If less than target, discard left half (l=mid+1). If greater, discard right half (r=mid-1). Halves the search space each iteration.",
      timeC:"O(log n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">search</span>(nums, target):
      l,r=<span class="nm">0</span>,<span class="fn">len</span>(nums)-<span class="nm">1</span>
      <span class="kw">while</span> l<=r:
          m=(l+r)//<span class="nm">2</span>
          <span class="kw">if</span> nums[m]==target: <span class="kw">return</span> m
          <span class="kw">elif</span> nums[m]<target: l=m+<span class="nm">1</span>
          <span class="kw">else</span>: r=m-<span class="nm">1</span>
      <span class="kw">return</span> -<span class="nm">1</span>`
    },
    {
      id:25, lc:34, title:"Find First and Last Position in Sorted Array", topic:"Binary Search", difficulty:"Medium",
      question:"Find the starting and ending position of target in a sorted array. O(log n) required.",
      hint:"Two binary searches: one biased left, one biased right.",
      explain:"Run binary search twice with different bias. For the left boundary: when we find the target, store it and keep searching left (r=mid-1). For the right boundary: when found, keep searching right (l=mid+1). This pins down both endpoints in O(log n) each.",
      timeC:"O(log n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">searchRange</span>(nums, target):
      <span class="kw">def</span> <span class="fn">bs</span>(left):
          l,r,idx=<span class="nm">0</span>,<span class="fn">len</span>(nums)-<span class="nm">1</span>,-<span class="nm">1</span>
          <span class="kw">while</span> l<=r:
              m=(l+r)//<span class="nm">2</span>
              <span class="kw">if</span> nums[m]==target:
                  idx=m
                  <span class="kw">if</span> left: r=m-<span class="nm">1</span>
                  <span class="kw">else</span>: l=m+<span class="nm">1</span>
              <span class="kw">elif</span> nums[m]<target: l=m+<span class="nm">1</span>
              <span class="kw">else</span>: r=m-<span class="nm">1</span>
          <span class="kw">return</span> idx
      <span class="kw">return</span> [<span class="fn">bs</span>(<span class="kw">True</span>),<span class="fn">bs</span>(<span class="kw">False</span>)]`
    },
    {
      id:26, lc:74, title:"Search a 2D Matrix", topic:"Binary Search", difficulty:"Medium",
      question:"Search for target in an m×n matrix where each row is sorted and first of each row > last of previous.",
      hint:"Treat as 1D array. mid maps to matrix[mid//n][mid%n].",
      explain:"The matrix can be treated as a flattened sorted 1D array. Map index mid to matrix[mid // n][mid % n]. Run standard binary search on range [0, m*n-1]. This avoids actually flattening, giving O(log(m*n)) time.",
      timeC:"O(log(m·n))", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">searchMatrix</span>(matrix, target):
      m,n=<span class="fn">len</span>(matrix),<span class="fn">len</span>(matrix[<span class="nm">0</span>])
      l,r=<span class="nm">0</span>,m*n-<span class="nm">1</span>
      <span class="kw">while</span> l<=r:
          mid=(l+r)//<span class="nm">2</span>
          val=matrix[mid//n][mid%n]
          <span class="kw">if</span> val==target: <span class="kw">return True</span>
          <span class="kw">elif</span> val<target: l=mid+<span class="nm">1</span>
          <span class="kw">else</span>: r=mid-<span class="nm">1</span>
      <span class="kw">return False</span>`
    },
    {
      id:27, lc:875, title:"Koko Eating Bananas", topic:"Binary Search", difficulty:"Medium",
      question:"Find minimum eating speed k so Koko can eat all piles within h hours.",
      hint:"Binary search on k. For each k, calculate total hours with ceil(pile/k).",
      explain:"Binary search on the answer space [1, max(piles)]. For a given speed k, hours needed = sum(ceil(pile/k)). If hours <= h, k is feasible — try lower. Otherwise try higher. This 'search on the answer' pattern converts an optimization problem to repeated feasibility checks.",
      timeC:"O(n log m)", spaceC:"O(1)",
      code:`<span class="kw">import</span> math
  <span class="kw">def</span> <span class="fn">minEatingSpeed</span>(piles, h):
      l,r=<span class="nm">1</span>,<span class="fn">max</span>(piles)
      <span class="kw">while</span> l<r:
          m=(l+r)//<span class="nm">2</span>
          hours=<span class="fn">sum</span>(math.ceil(p/m) <span class="kw">for</span> p <span class="kw">in</span> piles)
          <span class="kw">if</span> hours<=h: r=m
          <span class="kw">else</span>: l=m+<span class="nm">1</span>
      <span class="kw">return</span> l`
    },
    {
      id:28, lc:4, title:"Median of Two Sorted Arrays", topic:"Binary Search", difficulty:"Hard",
      question:"Find the median of two sorted arrays in O(log(m+n)) time.",
      hint:"Binary search on the smaller array. Partition both so left half = right half.",
      explain:"Binary search on the shorter array A. For each partition of A at index i, compute j = (m+n+1)/2 - i for B. Check if A[i-1] <= B[j] and B[j-1] <= A[i] (valid partition). The median is the max of left elements (odd total) or avg of max-left and min-right (even total).",
      timeC:"O(log(min(m,n)))", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">findMedianSortedArrays</span>(A, B):
      <span class="kw">if</span> <span class="fn">len</span>(A)><span class="fn">len</span>(B): A,B=B,A
      m,n=<span class="fn">len</span>(A),<span class="fn">len</span>(B)
      lo,hi=<span class="nm">0</span>,m
      <span class="kw">while</span> lo<=hi:
          i=(lo+hi)//<span class="nm">2</span>; j=(m+n+<span class="nm">1</span>)//<span class="nm">2</span>-i
          al=A[i-<span class="nm">1</span>] <span class="kw">if</span> i><span class="nm">0</span> <span class="kw">else</span> -<span class="fn">float</span>(<span class="st">'inf'</span>)
          ar=A[i]   <span class="kw">if</span> i<m <span class="kw">else</span>  <span class="fn">float</span>(<span class="st">'inf'</span>)
          bl=B[j-<span class="nm">1</span>] <span class="kw">if</span> j><span class="nm">0</span> <span class="kw">else</span> -<span class="fn">float</span>(<span class="st">'inf'</span>)
          br=B[j]   <span class="kw">if</span> j<n <span class="kw">else</span>  <span class="fn">float</span>(<span class="st">'inf'</span>)
          <span class="kw">if</span> al<=br <span class="kw">and</span> bl<=ar:
              <span class="kw">if</span> (m+n)%<span class="nm">2</span>: <span class="kw">return</span> <span class="fn">max</span>(al,bl)
              <span class="kw">return</span> (<span class="fn">max</span>(al,bl)+<span class="fn">min</span>(ar,br))/<span class="nm">2</span>
          <span class="kw">elif</span> al>br: hi=i-<span class="nm">1</span>
          <span class="kw">else</span>: lo=i+<span class="nm">1</span>`
    },
    {
      id:29, lc:981, title:"Time Based Key-Value Store", topic:"Binary Search", difficulty:"Medium",
      question:"Design a key-value store supporting set(key,value,timestamp) and get(key,timestamp) returning the value with largest timestamp ≤ given.",
      hint:"Store list of (timestamp, value) per key. Binary search on get.",
      explain:"Store per-key lists of (timestamp, value) in insertion order (timestamps always increase). On get, binary search the list to find the rightmost timestamp ≤ query. Python's bisect_right finds the insertion point for (ts, chr(127)), giving us the correct predecessor.",
      timeC:"O(log n) get, O(1) set", spaceC:"O(n)",
      code:`<span class="kw">from</span> collections <span class="kw">import</span> defaultdict
  <span class="kw">import</span> bisect
  <span class="kw">class</span> <span class="fn">TimeMap</span>:
      <span class="kw">def</span> <span class="fn">__init__</span>(self): self.d=defaultdict(<span class="fn">list</span>)
      <span class="kw">def</span> <span class="fn">set</span>(self,key,val,ts): self.d[key].append((ts,val))
      <span class="kw">def</span> <span class="fn">get</span>(self,key,ts):
          pairs=self.d[key]
          i=bisect.bisect_right(pairs,(ts,<span class="fn">chr</span>(<span class="nm">127</span>)))
          <span class="kw">return</span> pairs[i-<span class="nm">1</span>][<span class="nm">1</span>] <span class="kw">if</span> i><span class="nm">0</span> <span class="kw">else</span> <span class="st">""</span>`
    },
    {
      id:30, lc:206, title:"Reverse a Linked List", topic:"Linked Lists", difficulty:"Easy",
      question:"Reverse a singly linked list and return the reversed list head.",
      hint:"Iterative: use prev=None, cur=head. Reassign next pointers in-place.",
      explain:"Iterative three-pointer technique: prev=None, cur=head. At each step, save cur.next, point cur.next to prev, then advance both prev and cur. When cur is None, prev is the new head. No extra space needed.",
      timeC:"O(n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">reverseList</span>(head):
      prev=<span class="kw">None</span>; cur=head
      <span class="kw">while</span> cur:
          nxt=cur.next; cur.next=prev
          prev=cur; cur=nxt
      <span class="kw">return</span> prev`
    },
    {
      id:31, lc:141, title:"Detect Cycle in Linked List", topic:"Linked Lists", difficulty:"Easy",
      question:"Determine if a linked list has a cycle.",
      hint:"Floyd's cycle detection: slow/fast pointers. If they meet, cycle exists.",
      explain:"Floyd's tortoise and hare: slow moves 1 step, fast moves 2. If there's a cycle, fast will lap slow and they'll meet inside the cycle. If fast reaches null, no cycle. Mathematical proof: the gap decreases by 1 each step when both are in the cycle.",
      timeC:"O(n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">hasCycle</span>(head):
      slow=fast=head
      <span class="kw">while</span> fast <span class="kw">and</span> fast.next:
          slow=slow.next; fast=fast.next.next
          <span class="kw">if</span> slow==fast: <span class="kw">return True</span>
      <span class="kw">return False</span>`
    },
    {
      id:32, lc:21, title:"Merge Two Sorted Lists", topic:"Linked Lists", difficulty:"Easy",
      question:"Merge two sorted linked lists into one sorted list.",
      hint:"Use a dummy head. Iteratively attach the smaller node.",
      explain:"Dummy head simplifies edge cases (no special handling for empty input). Maintain a cur pointer. At each step, compare l1.val and l2.val, attach the smaller node to cur, and advance. After one list is exhausted, attach the remainder of the other (already sorted).",
      timeC:"O(m+n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">mergeTwoLists</span>(l1, l2):
      dummy=cur=ListNode(<span class="nm">0</span>)
      <span class="kw">while</span> l1 <span class="kw">and</span> l2:
          <span class="kw">if</span> l1.val<=l2.val: cur.next=l1; l1=l1.next
          <span class="kw">else</span>: cur.next=l2; l2=l2.next
          cur=cur.next
      cur.next=l1 <span class="kw">or</span> l2
      <span class="kw">return</span> dummy.next`
    },
    {
      id:33, lc:23, title:"Merge K Sorted Lists", topic:"Linked Lists", difficulty:"Hard",
      question:"Merge k sorted linked lists into one sorted linked list.",
      hint:"Use a min-heap of (val, idx, node). Pop smallest, push its next.",
      explain:"Initialize a min-heap with the head of each list (tuples include list index to break value ties). Repeatedly pop the smallest node, attach it to the result, and push that node's next if it exists. Heap always contains at most k elements — O(N log k) total.",
      timeC:"O(N log k)", spaceC:"O(k)",
      code:`<span class="kw">import</span> heapq
  <span class="kw">def</span> <span class="fn">mergeKLists</span>(lists):
      heap=[]
      <span class="kw">for</span> i,node <span class="kw">in</span> <span class="fn">enumerate</span>(lists):
          <span class="kw">if</span> node: heapq.heappush(heap,(node.val,i,node))
      dummy=cur=ListNode(<span class="nm">0</span>)
      <span class="kw">while</span> heap:
          val,i,node=heapq.heappop(heap)
          cur.next=node; cur=cur.next
          <span class="kw">if</span> node.next: heapq.heappush(heap,(node.next.val,i,node.next))
      <span class="kw">return</span> dummy.next`
    },
    {
      id:34, lc:19, title:"Remove Nth Node From End of List", topic:"Linked Lists", difficulty:"Medium",
      question:"Remove the nth node from the end of a linked list in one pass.",
      hint:"Two pointers: advance fast n steps, then move both until fast.next is None.",
      explain:"Two pointers with a gap of n. Advance fast n+1 steps from a dummy node (the +1 means slow stops at the node BEFORE the one to delete). Then move both until fast is null. Now slow.next is the target — skip it. Dummy head handles edge case of removing the head.",
      timeC:"O(n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">removeNthFromEnd</span>(head, n):
      dummy=ListNode(<span class="nm">0</span>,head)
      slow=fast=dummy
      <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(n+<span class="nm">1</span>): fast=fast.next
      <span class="kw">while</span> fast: slow=slow.next; fast=fast.next
      slow.next=slow.next.next
      <span class="kw">return</span> dummy.next`
    },
    {
      id:35, lc:143, title:"Reorder List", topic:"Linked Lists", difficulty:"Medium",
      question:"Reorder L0→L1→…→Ln to L0→Ln→L1→Ln-1→L2→Ln-2→… in-place.",
      hint:"Find middle, reverse second half, then merge two halves alternating.",
      explain:"Three steps: (1) Find the midpoint using slow/fast pointers. (2) Reverse the second half of the list. (3) Interleave the two halves — take one from each alternately. Each step is O(n) and O(1) space.",
      timeC:"O(n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">reorderList</span>(head):
      slow=fast=head
      <span class="kw">while</span> fast <span class="kw">and</span> fast.next: slow=slow.next; fast=fast.next.next
      prev,cur=<span class="kw">None</span>,slow.next; slow.next=<span class="kw">None</span>
      <span class="kw">while</span> cur:
          nxt=cur.next; cur.next=prev; prev=cur; cur=nxt
      l1,l2=head,prev
      <span class="kw">while</span> l2:
          l1n,l2n=l1.next,l2.next
          l1.next=l2; l2.next=l1n; l1,l2=l1n,l2n`
    },
    {
      id:36, lc:146, title:"LRU Cache", topic:"Linked Lists", difficulty:"Medium",
      question:"Design an LRU cache with O(1) get and put.",
      hint:"Doubly linked list + hashmap. Move to front on access, evict from tail.",
      explain:"Python's OrderedDict maintains insertion order and supports move_to_end(key). On get, move the key to the end (most recently used). On put, insert/update and move to end. If over capacity, pop the item at the front (least recently used). All O(1) amortized.",
      timeC:"O(1) amortized", spaceC:"O(cap)",
      code:`<span class="kw">from</span> collections <span class="kw">import</span> OrderedDict
  <span class="kw">class</span> <span class="fn">LRUCache</span>:
      <span class="kw">def</span> <span class="fn">__init__</span>(self,cap):
          self.cap=cap; self.cache=OrderedDict()
      <span class="kw">def</span> <span class="fn">get</span>(self,key):
          <span class="kw">if</span> key <span class="kw">not in</span> self.cache: <span class="kw">return</span> -<span class="nm">1</span>
          self.cache.move_to_end(key); <span class="kw">return</span> self.cache[key]
      <span class="kw">def</span> <span class="fn">put</span>(self,key,val):
          self.cache[key]=val; self.cache.move_to_end(key)
          <span class="kw">if</span> <span class="fn">len</span>(self.cache)>self.cap: self.cache.popitem(last=<span class="kw">False</span>)`
    },
    {
      id:37, lc:138, title:"Copy List with Random Pointer", topic:"Linked Lists", difficulty:"Medium",
      question:"Deep copy a linked list where each node has a next and a random pointer.",
      hint:"Two passes: first map old→new nodes, then assign next/random pointers.",
      explain:"Two-pass solution with a hashmap: Pass 1 creates a clone of every node (without setting pointers) and maps original → clone. Pass 2 wires up next and random pointers using the map. The {None: None} sentinel handles null pointers cleanly.",
      timeC:"O(n)", spaceC:"O(n)",
      code:`<span class="kw">def</span> <span class="fn">copyRandomList</span>(head):
      old_to_new={<span class="kw">None</span>:<span class="kw">None</span>}
      cur=head
      <span class="kw">while</span> cur:
          old_to_new[cur]=Node(cur.val); cur=cur.next
      cur=head
      <span class="kw">while</span> cur:
          old_to_new[cur].next=old_to_new[cur.next]
          old_to_new[cur].random=old_to_new[cur.random]
          cur=cur.next
      <span class="kw">return</span> old_to_new[head]`
    },
    {
      id:38, lc:287, title:"Find the Duplicate Number", topic:"Linked Lists", difficulty:"Medium",
      question:"Find the duplicate in an array of n+1 integers [1..n] without modifying the array, using O(1) space.",
      hint:"Floyd's cycle detection treating array as a linked list.",
      explain:"Treat each value as a pointer to the next index (i → nums[i]). Because of the duplicate, two indices eventually point to the same node — creating a cycle. Floyd's algorithm finds where the cycle starts, which is exactly the duplicate value.",
      timeC:"O(n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">findDuplicate</span>(nums):
      slow=fast=nums[<span class="nm">0</span>]
      <span class="kw">while True</span>:
          slow=nums[slow]; fast=nums[nums[fast]]
          <span class="kw">if</span> slow==fast: <span class="kw">break</span>
      slow=nums[<span class="nm">0</span>]
      <span class="kw">while</span> slow!=fast:
          slow=nums[slow]; fast=nums[fast]
      <span class="kw">return</span> slow`
    },
    {
      id:39, lc:2, title:"Add Two Numbers", topic:"Linked Lists", difficulty:"Medium",
      question:"Add two non-negative integers represented as reversed linked lists and return the sum as a linked list.",
      hint:"Simulate addition with carry. Process both lists digit by digit.",
      explain:"Simulate grade-school addition. At each step, sum the current digits plus carry. New digit = sum % 10, new carry = sum // 10 (Python's divmod). Use a dummy head so the result builds naturally. Continue while either list has nodes OR there's a carry remaining.",
      timeC:"O(max(m,n))", spaceC:"O(max(m,n))",
      code:`<span class="kw">def</span> <span class="fn">addTwoNumbers</span>(l1, l2):
      dummy=cur=ListNode(<span class="nm">0</span>); carry=<span class="nm">0</span>
      <span class="kw">while</span> l1 <span class="kw">or</span> l2 <span class="kw">or</span> carry:
          v1=l1.val <span class="kw">if</span> l1 <span class="kw">else</span> <span class="nm">0</span>
          v2=l2.val <span class="kw">if</span> l2 <span class="kw">else</span> <span class="nm">0</span>
          carry,val=divmod(v1+v2+carry,<span class="nm">10</span>)
          cur.next=ListNode(val); cur=cur.next
          l1=l1.next <span class="kw">if</span> l1 <span class="kw">else None</span>
          l2=l2.next <span class="kw">if</span> l2 <span class="kw">else None</span>
      <span class="kw">return</span> dummy.next`
    },
    {
      id:40, lc:226, title:"Invert Binary Tree", topic:"Trees", difficulty:"Easy",
      question:"Invert a binary tree (mirror it) and return its root.",
      hint:"Recursively swap left and right children at every node.",
      explain:"Post-order DFS: recursively invert the left and right subtrees, then swap them at the current node. Base case: null node returns null. This legendary problem famously tripped up a Google engineer (Max Howell, creator of Homebrew).",
      timeC:"O(n)", spaceC:"O(h)",
      code:`<span class="kw">def</span> <span class="fn">invertTree</span>(root):
      <span class="kw">if not</span> root: <span class="kw">return None</span>
      root.left,root.right=(
          <span class="fn">invertTree</span>(root.right),<span class="fn">invertTree</span>(root.left))
      <span class="kw">return</span> root`
    },
    {
      id:41, lc:104, title:"Maximum Depth of Binary Tree", topic:"Trees", difficulty:"Easy",
      question:"Return the maximum depth of a binary tree.",
      hint:"Recursive DFS: return 1 + max(left_depth, right_depth).",
      explain:"Classic recursive DFS. At each node, the depth is 1 + the deeper of its two subtrees. Base case: null → 0. You can also do iterative BFS counting levels. The recursive form is extremely concise.",
      timeC:"O(n)", spaceC:"O(h)",
      code:`<span class="kw">def</span> <span class="fn">maxDepth</span>(root):
      <span class="kw">if not</span> root: <span class="kw">return</span> <span class="nm">0</span>
      <span class="kw">return</span> <span class="nm">1</span>+<span class="fn">max</span>(<span class="fn">maxDepth</span>(root.left),<span class="fn">maxDepth</span>(root.right))`
    },
    {
      id:42, lc:100, title:"Same Tree", topic:"Trees", difficulty:"Easy",
      question:"Check if two binary trees are structurally identical with same node values.",
      hint:"Recursively check: both None, or values equal and both subtrees same.",
      explain:"Recursive co-traversal: both trees must be null simultaneously, or both non-null with equal values AND recursively same left/right subtrees. Any other combination returns false.",
      timeC:"O(n)", spaceC:"O(h)",
      code:`<span class="kw">def</span> <span class="fn">isSameTree</span>(p, q):
      <span class="kw">if not</span> p <span class="kw">and not</span> q: <span class="kw">return True</span>
      <span class="kw">if not</span> p <span class="kw">or not</span> q: <span class="kw">return False</span>
      <span class="kw">return</span>(p.val==q.val <span class="kw">and</span>
             <span class="fn">isSameTree</span>(p.left,q.left) <span class="kw">and</span>
             <span class="fn">isSameTree</span>(p.right,q.right))`
    },
    {
      id:43, lc:572, title:"Subtree of Another Tree", topic:"Trees", difficulty:"Easy",
      question:"Return true if subRoot is a subtree of root.",
      hint:"For each node in root, check if isSameTree(node, subRoot).",
      explain:"DFS through root. At each node, call isSameTree to check if the subtrees match. If either child contains a match, return true. This is O(m*n) worst case but very clean. Optimization: serialize both trees and use string matching.",
      timeC:"O(m·n)", spaceC:"O(h)",
      code:`<span class="kw">def</span> <span class="fn">isSubtree</span>(root, sub):
      <span class="kw">if not</span> root: <span class="kw">return False</span>
      <span class="kw">if</span> <span class="fn">isSame</span>(root,sub): <span class="kw">return True</span>
      <span class="kw">return</span> <span class="fn">isSubtree</span>(root.left,sub) <span class="kw">or</span> <span class="fn">isSubtree</span>(root.right,sub)
  <span class="kw">def</span> <span class="fn">isSame</span>(p,q):
      <span class="kw">if not</span> p <span class="kw">and not</span> q: <span class="kw">return True</span>
      <span class="kw">if not</span> p <span class="kw">or not</span> q: <span class="kw">return False</span>
      <span class="kw">return</span> p.val==q.val <span class="kw">and</span> <span class="fn">isSame</span>(p.left,q.left) <span class="kw">and</span> <span class="fn">isSame</span>(p.right,q.right)`
    },
    {
      id:44, lc:235, title:"Lowest Common Ancestor of a BST", topic:"Trees", difficulty:"Medium",
      question:"Find the LCA of two nodes p and q in a BST.",
      hint:"If both p,q > root go right; if both < root go left; else root is LCA.",
      explain:"BST property tells us where the LCA must be. If both nodes are greater than root, LCA is in the right subtree. If both smaller, it's in the left. Otherwise, root is the LCA (one is in each subtree, or one equals root). Iterative solution uses O(1) space.",
      timeC:"O(h)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">lowestCommonAncestor</span>(root,p,q):
      <span class="kw">while</span> root:
          <span class="kw">if</span> p.val>root.val <span class="kw">and</span> q.val>root.val: root=root.right
          <span class="kw">elif</span> p.val<root.val <span class="kw">and</span> q.val<root.val: root=root.left
          <span class="kw">else</span>: <span class="kw">return</span> root`
    },
    {
      id:45, lc:102, title:"Binary Tree Level Order Traversal", topic:"Trees", difficulty:"Medium",
      question:"Return the level-order traversal of a binary tree's values (left to right, level by level).",
      hint:"BFS with a queue. Process all nodes at each level before moving on.",
      explain:"Standard BFS using a deque. At the start of each iteration, the queue holds exactly all nodes for the current level. Process len(q) nodes (snapshot the size), collecting their values and enqueuing their children. The snapshot prevents mixing levels.",
      timeC:"O(n)", spaceC:"O(n)",
      code:`<span class="kw">from</span> collections <span class="kw">import</span> deque
  <span class="kw">def</span> <span class="fn">levelOrder</span>(root):
      <span class="kw">if not</span> root: <span class="kw">return</span> []
      q,res=deque([root]),[]
      <span class="kw">while</span> q:
          level=[]
          <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(q)):
              node=q.popleft(); level.append(node.val)
              <span class="kw">if</span> node.left: q.append(node.left)
              <span class="kw">if</span> node.right: q.append(node.right)
          res.append(level)
      <span class="kw">return</span> res`
    },
    {
      id:46, lc:98, title:"Validate Binary Search Tree", topic:"Trees", difficulty:"Medium",
      question:"Determine if a binary tree is a valid BST.",
      hint:"Pass min/max bounds recursively.",
      explain:"A node is valid if its value is strictly between its inherited lower and upper bounds. Initially bounds are ±∞. Going left: upper bound becomes current node's value. Going right: lower bound becomes current node's value. This validates the entire subtree, not just parent-child.",
      timeC:"O(n)", spaceC:"O(h)",
      code:`<span class="kw">def</span> <span class="fn">isValidBST</span>(root):
      <span class="kw">def</span> <span class="fn">valid</span>(node,lo,hi):
          <span class="kw">if not</span> node: <span class="kw">return True</span>
          <span class="kw">if not</span>(lo<node.val<hi): <span class="kw">return False</span>
          <span class="kw">return</span>(<span class="fn">valid</span>(node.left,lo,node.val) <span class="kw">and</span>
                 <span class="fn">valid</span>(node.right,node.val,hi))
      <span class="kw">return</span> <span class="fn">valid</span>(root,-<span class="fn">float</span>(<span class="st">'inf'</span>),<span class="fn">float</span>(<span class="st">'inf'</span>))`
    },
    {
      id:47, lc:230, title:"Kth Smallest Element in a BST", topic:"Trees", difficulty:"Medium",
      question:"Return the kth smallest value in a BST.",
      hint:"In-order traversal of BST gives sorted order.",
      explain:"In-order traversal (left → root → right) of a BST visits nodes in ascending order. Use an iterative stack-based in-order traversal. Decrement a counter each time you visit a node. When counter reaches 0, that node's value is the answer.",
      timeC:"O(h+k)", spaceC:"O(h)",
      code:`<span class="kw">def</span> <span class="fn">kthSmallest</span>(root,k):
      stack,cur=[],root; n=<span class="nm">0</span>
      <span class="kw">while</span> stack <span class="kw">or</span> cur:
          <span class="kw">while</span> cur: stack.append(cur); cur=cur.left
          cur=stack.pop(); n+=<span class="nm">1</span>
          <span class="kw">if</span> n==k: <span class="kw">return</span> cur.val
          cur=cur.right`
    },
    {
      id:48, lc:105, title:"Construct Tree from Preorder and Inorder", topic:"Trees", difficulty:"Medium",
      question:"Construct a binary tree given its preorder and inorder traversal arrays.",
      hint:"preorder[0] is root. Find its index in inorder to split subtrees.",
      explain:"Preorder's first element is always the root. Find that root in inorder — everything to its left is the left subtree, everything to its right is the right subtree. Recurse with the corresponding slices of both arrays. Using a hashmap for inorder lookup speeds this to O(n).",
      timeC:"O(n)", spaceC:"O(n)",
      code:`<span class="kw">def</span> <span class="fn">buildTree</span>(preorder, inorder):
      <span class="kw">if not</span> preorder: <span class="kw">return None</span>
      root=TreeNode(preorder[<span class="nm">0</span>])
      mid=inorder.index(preorder[<span class="nm">0</span>])
      root.left=<span class="fn">buildTree</span>(preorder[<span class="nm">1</span>:mid+<span class="nm">1</span>],inorder[:mid])
      root.right=<span class="fn">buildTree</span>(preorder[mid+<span class="nm">1</span>:],inorder[mid+<span class="nm">1</span>:])
      <span class="kw">return</span> root`
    },
    {
      id:49, lc:124, title:"Binary Tree Maximum Path Sum", topic:"Trees", difficulty:"Hard",
      question:"Find the maximum sum path between any two nodes in a binary tree.",
      hint:"DFS: at each node, max gain = node.val + max(0, left_gain, right_gain). Track global max.",
      explain:"DFS returning the max one-sided gain from each node (can't split at two places). For the global answer, at each node we CAN split: answer candidate = node.val + left_gain + right_gain (both sides). Update a global max. The function returns the one-sided gain for the parent to use.",
      timeC:"O(n)", spaceC:"O(h)",
      code:`<span class="kw">def</span> <span class="fn">maxPathSum</span>(root):
      res=[root.val]
      <span class="kw">def</span> <span class="fn">dfs</span>(node):
          <span class="kw">if not</span> node: <span class="kw">return</span> <span class="nm">0</span>
          l=<span class="fn">max</span>(<span class="fn">dfs</span>(node.left),<span class="nm">0</span>)
          r=<span class="fn">max</span>(<span class="fn">dfs</span>(node.right),<span class="nm">0</span>)
          res[<span class="nm">0</span>]=<span class="fn">max</span>(res[<span class="nm">0</span>],node.val+l+r)
          <span class="kw">return</span> node.val+<span class="fn">max</span>(l,r)
      <span class="fn">dfs</span>(root); <span class="kw">return</span> res[<span class="nm">0</span>]`
    },
    {
      id:50, lc:297, title:"Serialize and Deserialize Binary Tree", topic:"Trees", difficulty:"Hard",
      question:"Design serialize/deserialize functions for a binary tree.",
      hint:"BFS/preorder with 'N' for null. Reconstruct using a queue/index.",
      explain:"Preorder DFS serialization: append node value or 'N' for null. Produces a flat list capturing full tree structure. Deserialization: use an iterator over the split list. Recursively consume values — if 'N', return null; otherwise create a node and recurse for left and right.",
      timeC:"O(n)", spaceC:"O(n)",
      code:`<span class="kw">class</span> <span class="fn">Codec</span>:
      <span class="kw">def</span> <span class="fn">serialize</span>(self,root):
          res=[]
          <span class="kw">def</span> <span class="fn">dfs</span>(n):
              <span class="kw">if not</span> n: res.append(<span class="st">"N"</span>); <span class="kw">return</span>
              res.append(<span class="fn">str</span>(n.val)); <span class="fn">dfs</span>(n.left); <span class="fn">dfs</span>(n.right)
          <span class="fn">dfs</span>(root); <span class="kw">return</span> <span class="st">","</span>.join(res)
      <span class="kw">def</span> <span class="fn">deserialize</span>(self,data):
          vals=<span class="fn">iter</span>(data.split(<span class="st">","</span>))
          <span class="kw">def</span> <span class="fn">dfs</span>():
              v=<span class="fn">next</span>(vals)
              <span class="kw">if</span> v==<span class="st">"N"</span>: <span class="kw">return None</span>
              node=TreeNode(<span class="fn">int</span>(v))
              node.left,node.right=<span class="fn">dfs</span>(),<span class="fn">dfs</span>()
              <span class="kw">return</span> node
          <span class="kw">return</span> <span class="fn">dfs</span>()`
    },
    {
      id:51, lc:543, title:"Diameter of Binary Tree", topic:"Trees", difficulty:"Easy",
      question:"Return the length of the longest path between any two nodes (not necessarily through root).",
      hint:"DFS. Diameter at each node = left_depth + right_depth. Track global max.",
      explain:"At each node, the diameter passing through it = left_depth + right_depth. DFS returns depth (1 + max(left, right)). Update a global max at each node with left + right. The answer might not pass through the root, so we must check every node.",
      timeC:"O(n)", spaceC:"O(h)",
      code:`<span class="kw">def</span> <span class="fn">diameterOfBinaryTree</span>(root):
      res=[<span class="nm">0</span>]
      <span class="kw">def</span> <span class="fn">dfs</span>(node):
          <span class="kw">if not</span> node: <span class="kw">return</span> -<span class="nm">1</span>
          l,r=<span class="fn">dfs</span>(node.left),<span class="fn">dfs</span>(node.right)
          res[<span class="nm">0</span>]=<span class="fn">max</span>(res[<span class="nm">0</span>],l+r+<span class="nm">2</span>)
          <span class="kw">return</span> <span class="nm">1</span>+<span class="fn">max</span>(l,r)
      <span class="fn">dfs</span>(root); <span class="kw">return</span> res[<span class="nm">0</span>]`
    },
    {
      id:52, lc:1448, title:"Count Good Nodes in Binary Tree", topic:"Trees", difficulty:"Medium",
      question:"A node is 'good' if no node on the path from root to it has a greater value. Count good nodes.",
      hint:"DFS, pass current max. Node is good if node.val >= max seen so far.",
      explain:"DFS while tracking the maximum value seen on the current root-to-node path. A node is 'good' if its value ≥ that maximum (it's a new maximum or equal). Update the running max as we recurse deeper.",
      timeC:"O(n)", spaceC:"O(h)",
      code:`<span class="kw">def</span> <span class="fn">goodNodes</span>(root):
      <span class="kw">def</span> <span class="fn">dfs</span>(node,mx):
          <span class="kw">if not</span> node: <span class="kw">return</span> <span class="nm">0</span>
          good=<span class="nm">1</span> <span class="kw">if</span> node.val>=mx <span class="kw">else</span> <span class="nm">0</span>
          mx=<span class="fn">max</span>(mx,node.val)
          <span class="kw">return</span> good+<span class="fn">dfs</span>(node.left,mx)+<span class="fn">dfs</span>(node.right,mx)
      <span class="kw">return</span> <span class="fn">dfs</span>(root,root.val)`
    },
    {
      id:53, lc:200, title:"Number of Islands", topic:"Graphs", difficulty:"Medium",
      question:"Count the number of islands (groups of adjacent '1's surrounded by '0's) in a 2D binary grid.",
      hint:"BFS/DFS from each unvisited '1'. Mark cells as visited by setting to '0'.",
      explain:"For each unvisited '1' cell, increment the island count and run DFS/BFS to sink the entire island (set connected '1's to '0'). This prevents double-counting. The number of DFS/BFS initiations equals the number of islands.",
      timeC:"O(m·n)", spaceC:"O(m·n)",
      code:`<span class="kw">def</span> <span class="fn">numIslands</span>(grid):
      count=<span class="nm">0</span>
      <span class="kw">def</span> <span class="fn">dfs</span>(i,j):
          <span class="kw">if</span> i<<span class="nm">0</span> <span class="kw">or</span> i>=<span class="fn">len</span>(grid) <span class="kw">or</span> j<<span class="nm">0</span> <span class="kw">or</span> j>=<span class="fn">len</span>(grid[<span class="nm">0</span>]) <span class="kw">or</span> grid[i][j]!=<span class="st">'1'</span>: <span class="kw">return</span>
          grid[i][j]=<span class="st">'0'</span>
          <span class="kw">for</span> di,dj <span class="kw">in</span> [(<span class="nm">1</span>,<span class="nm">0</span>),(-<span class="nm">1</span>,<span class="nm">0</span>),(<span class="nm">0</span>,<span class="nm">1</span>),(<span class="nm">0</span>,-<span class="nm">1</span>)]: <span class="fn">dfs</span>(i+di,j+dj)
      <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(grid)):
          <span class="kw">for</span> j <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(grid[<span class="nm">0</span>])):
              <span class="kw">if</span> grid[i][j]==<span class="st">'1'</span>: <span class="fn">dfs</span>(i,j); count+=<span class="nm">1</span>
      <span class="kw">return</span> count`
    },
    {
      id:54, lc:133, title:"Clone Graph", topic:"Graphs", difficulty:"Medium",
      question:"Return a deep copy of an undirected graph.",
      hint:"BFS/DFS with a visited hashmap {original: clone}.",
      explain:"DFS with a hashmap from original nodes to their clones. When visiting a node, check if it's already cloned (cycle/revisit detection). If not, create the clone, store it, then recursively clone all neighbors and wire up the cloned neighbors list.",
      timeC:"O(V+E)", spaceC:"O(V)",
      code:`<span class="kw">def</span> <span class="fn">cloneGraph</span>(node):
      clones={}
      <span class="kw">def</span> <span class="fn">dfs</span>(n):
          <span class="kw">if</span> n <span class="kw">in</span> clones: <span class="kw">return</span> clones[n]
          clone=Node(n.val); clones[n]=clone
          <span class="kw">for</span> nb <span class="kw">in</span> n.neighbors: clone.neighbors.append(<span class="fn">dfs</span>(nb))
          <span class="kw">return</span> clone
      <span class="kw">return</span> <span class="fn">dfs</span>(node) <span class="kw">if</span> node <span class="kw">else None</span>`
    },
    {
      id:55, lc:417, title:"Pacific Atlantic Water Flow", topic:"Graphs", difficulty:"Medium",
      question:"Return cells from which water can flow to both the Pacific and Atlantic oceans.",
      hint:"Reverse BFS from both oceans. A cell is valid if in both reachable sets.",
      explain:"Instead of simulating forward flow from every cell (expensive), reverse BFS from the borders: Pacific = top/left edges, Atlantic = bottom/right edges. In reverse, flow goes to HIGHER or equal cells. A cell reachable from both sets is an answer.",
      timeC:"O(m·n)", spaceC:"O(m·n)",
      code:`<span class="kw">from</span> collections <span class="kw">import</span> deque
  <span class="kw">def</span> <span class="fn">pacificAtlantic</span>(heights):
      m,n=<span class="fn">len</span>(heights),<span class="fn">len</span>(heights[<span class="nm">0</span>])
      <span class="kw">def</span> <span class="fn">bfs</span>(starts):
          vis=<span class="fn">set</span>(starts); q=deque(starts)
          <span class="kw">while</span> q:
              r,c=q.popleft()
              <span class="kw">for</span> dr,dc <span class="kw">in</span> [(<span class="nm">1</span>,<span class="nm">0</span>),(-<span class="nm">1</span>,<span class="nm">0</span>),(<span class="nm">0</span>,<span class="nm">1</span>),(<span class="nm">0</span>,-<span class="nm">1</span>)]:
                  nr,nc=r+dr,c+dc
                  <span class="kw">if</span> <span class="nm">0</span><=nr<m <span class="kw">and</span> <span class="nm">0</span><=nc<n <span class="kw">and</span> (nr,nc) <span class="kw">not in</span> vis <span class="kw">and</span> heights[nr][nc]>=heights[r][c]:
                      vis.add((nr,nc)); q.append((nr,nc))
          <span class="kw">return</span> vis
      pac=<span class="fn">bfs</span>([(i,<span class="nm">0</span>) <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(m)]+[(<span class="nm">0</span>,j) <span class="kw">for</span> j <span class="kw">in</span> <span class="fn">range</span>(n)])
      atl=<span class="fn">bfs</span>([(i,n-<span class="nm">1</span>) <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(m)]+[(m-<span class="nm">1</span>,j) <span class="kw">for</span> j <span class="kw">in</span> <span class="fn">range</span>(n)])
      <span class="kw">return</span> [[r,c] <span class="kw">for</span> r,c <span class="kw">in</span> pac <span class="kw">if</span> (r,c) <span class="kw">in</span> atl]`
    },
    {
      id:56, lc:207, title:"Course Schedule", topic:"Graphs", difficulty:"Medium",
      question:"Given course prerequisites, determine if you can finish all courses (i.e. no cycle exists).",
      hint:"Cycle detection in directed graph via DFS with 3 states: unvisited, visiting, visited.",
      explain:"Build a directed adjacency list. DFS with 3-state visited array: 0=unvisited, 1=currently in DFS path (cycle if we revisit), 2=fully processed (safe). If we reach state 1 again, a cycle exists. Once all neighbors are safe, mark as 2.",
      timeC:"O(V+E)", spaceC:"O(V+E)",
      code:`<span class="kw">def</span> <span class="fn">canFinish</span>(n,prereqs):
      adj=[[] <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(n)]
      <span class="kw">for</span> a,b <span class="kw">in</span> prereqs: adj[a].append(b)
      vis=[<span class="nm">0</span>]*n
      <span class="kw">def</span> <span class="fn">dfs</span>(c):
          <span class="kw">if</span> vis[c]==<span class="nm">1</span>: <span class="kw">return False</span>
          <span class="kw">if</span> vis[c]==<span class="nm">2</span>: <span class="kw">return True</span>
          vis[c]=<span class="nm">1</span>
          <span class="kw">for</span> pre <span class="kw">in</span> adj[c]:
              <span class="kw">if not</span> <span class="fn">dfs</span>(pre): <span class="kw">return False</span>
          vis[c]=<span class="nm">2</span>; <span class="kw">return True</span>
      <span class="kw">return all</span>(<span class="fn">dfs</span>(c) <span class="kw">for</span> c <span class="kw">in</span> <span class="fn">range</span>(n))`
    },
    {
      id:57, lc:210, title:"Course Schedule II", topic:"Graphs", difficulty:"Medium",
      question:"Return a valid course ordering (topological sort), or [] if a cycle exists.",
      hint:"Topological sort via DFS. Append to result after processing all neighbors (post-order).",
      explain:"Same cycle detection as Course Schedule I, but now we also build the topological order. Append each node to the order list AFTER all its dependencies are processed (post-order). Reverse not needed since we append in finishing order, which IS topological.",
      timeC:"O(V+E)", spaceC:"O(V+E)",
      code:`<span class="kw">def</span> <span class="fn">findOrder</span>(n,prereqs):
      adj=[[] <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(n)]
      <span class="kw">for</span> a,b <span class="kw">in</span> prereqs: adj[a].append(b)
      visit,order=[<span class="nm">0</span>]*n,[]
      <span class="kw">def</span> <span class="fn">dfs</span>(c):
          <span class="kw">if</span> visit[c]==<span class="nm">1</span>: <span class="kw">return False</span>
          <span class="kw">if</span> visit[c]==<span class="nm">2</span>: <span class="kw">return True</span>
          visit[c]=<span class="nm">1</span>
          <span class="kw">for</span> pre <span class="kw">in</span> adj[c]:
              <span class="kw">if not</span> <span class="fn">dfs</span>(pre): <span class="kw">return False</span>
          visit[c]=<span class="nm">2</span>; order.append(c); <span class="kw">return True</span>
      <span class="kw">for</span> c <span class="kw">in</span> <span class="fn">range</span>(n):
          <span class="kw">if not</span> <span class="fn">dfs</span>(c): <span class="kw">return</span> []
      <span class="kw">return</span> order`
    },
    {
      id:58, lc:994, title:"Rotting Oranges", topic:"Graphs", difficulty:"Medium",
      question:"Find minimum minutes until no fresh orange remains (rotten spread to adjacent fresh each minute). Return -1 if impossible.",
      hint:"Multi-source BFS from all rotten oranges simultaneously.",
      explain:"Multi-source BFS: enqueue ALL initially rotten oranges at time=0. BFS naturally processes all cells at distance d before distance d+1. Count fresh oranges initially; decrement as they rot. At the end, if any fresh remain → return -1, else return the time of the last rotting.",
      timeC:"O(m·n)", spaceC:"O(m·n)",
      code:`<span class="kw">from</span> collections <span class="kw">import</span> deque
  <span class="kw">def</span> <span class="fn">orangesRotting</span>(grid):
      m,n=<span class="fn">len</span>(grid),<span class="fn">len</span>(grid[<span class="nm">0</span>])
      q,fresh=deque(),<span class="nm">0</span>
      <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(m):
          <span class="kw">for</span> j <span class="kw">in</span> <span class="fn">range</span>(n):
              <span class="kw">if</span> grid[i][j]==<span class="nm">2</span>: q.append((i,j,<span class="nm">0</span>))
              <span class="kw">elif</span> grid[i][j]==<span class="nm">1</span>: fresh+=<span class="nm">1</span>
      mins=<span class="nm">0</span>
      <span class="kw">while</span> q:
          r,c,t=q.popleft()
          <span class="kw">for</span> dr,dc <span class="kw">in</span> [(<span class="nm">1</span>,<span class="nm">0</span>),(-<span class="nm">1</span>,<span class="nm">0</span>),(<span class="nm">0</span>,<span class="nm">1</span>),(<span class="nm">0</span>,-<span class="nm">1</span>)]:
              nr,nc=r+dr,c+dc
              <span class="kw">if</span> <span class="nm">0</span><=nr<m <span class="kw">and</span> <span class="nm">0</span><=nc<n <span class="kw">and</span> grid[nr][nc]==<span class="nm">1</span>:
                  grid[nr][nc]=<span class="nm">2</span>; fresh-=<span class="nm">1</span>; mins=t+<span class="nm">1</span>; q.append((nr,nc,t+<span class="nm">1</span>))
      <span class="kw">return</span> mins <span class="kw">if</span> fresh==<span class="nm">0</span> <span class="kw">else</span> -<span class="nm">1</span>`
    },
    {
      id:59, lc:1091, title:"Shortest Path in Binary Matrix", topic:"Graphs", difficulty:"Medium",
      question:"Find the shortest clear 8-directional path from top-left to bottom-right in a binary matrix.",
      hint:"BFS from (0,0). Level = path length. Check all 8 directions.",
      explain:"BFS guarantees the shortest path in an unweighted graph. Start from (0,0) with distance 1. At each step, explore all 8 neighbors. Mark visited immediately (set to 1) to avoid revisiting. First time we reach (n-1, n-1), return the current distance.",
      timeC:"O(n²)", spaceC:"O(n²)",
      code:`<span class="kw">from</span> collections <span class="kw">import</span> deque
  <span class="kw">def</span> <span class="fn">shortestPathBinaryMatrix</span>(grid):
      n=<span class="fn">len</span>(grid)
      <span class="kw">if</span> grid[<span class="nm">0</span>][<span class="nm">0</span>] <span class="kw">or</span> grid[n-<span class="nm">1</span>][n-<span class="nm">1</span>]: <span class="kw">return</span> -<span class="nm">1</span>
      q=deque([(<span class="nm">0</span>,<span class="nm">0</span>,<span class="nm">1</span>)]); grid[<span class="nm">0</span>][<span class="nm">0</span>]=<span class="nm">1</span>
      dirs=[(-<span class="nm">1</span>,-<span class="nm">1</span>),(-<span class="nm">1</span>,<span class="nm">0</span>),(-<span class="nm">1</span>,<span class="nm">1</span>),(<span class="nm">0</span>,-<span class="nm">1</span>),(<span class="nm">0</span>,<span class="nm">1</span>),(<span class="nm">1</span>,-<span class="nm">1</span>),(<span class="nm">1</span>,<span class="nm">0</span>),(<span class="nm">1</span>,<span class="nm">1</span>)]
      <span class="kw">while</span> q:
          r,c,d=q.popleft()
          <span class="kw">if</span> r==n-<span class="nm">1</span> <span class="kw">and</span> c==n-<span class="nm">1</span>: <span class="kw">return</span> d
          <span class="kw">for</span> dr,dc <span class="kw">in</span> dirs:
              nr,nc=r+dr,c+dc
              <span class="kw">if</span> <span class="nm">0</span><=nr<n <span class="kw">and</span> <span class="nm">0</span><=nc<n <span class="kw">and not</span> grid[nr][nc]:
                  grid[nr][nc]=<span class="nm">1</span>; q.append((nr,nc,d+<span class="nm">1</span>))
      <span class="kw">return</span> -<span class="nm">1</span>`
    },
    {
      id:60, lc:127, title:"Word Ladder", topic:"Graphs", difficulty:"Hard",
      question:"Return the length of the shortest transformation sequence from beginWord to endWord, changing one letter at a time (each intermediate word must be in wordList).",
      hint:"BFS. Try all 26 letter swaps at each position. Use a set for O(1) lookup.",
      explain:"Model as graph BFS: each word is a node, edges connect words that differ by one letter. BFS finds the shortest path. At each word, generate all possible one-letter mutations and check if they're in the word set. Remove visited words from the set to prevent cycles.",
      timeC:"O(N·L·26)", spaceC:"O(N·L)",
      code:`<span class="kw">from</span> collections <span class="kw">import</span> deque
  <span class="kw">def</span> <span class="fn">ladderLength</span>(begin,end,wordList):
      ws=<span class="fn">set</span>(wordList)
      <span class="kw">if</span> end <span class="kw">not in</span> ws: <span class="kw">return</span> <span class="nm">0</span>
      q=deque([(begin,<span class="nm">1</span>)])
      <span class="kw">while</span> q:
          word,steps=q.popleft()
          <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(word)):
              <span class="kw">for</span> c <span class="kw">in</span> <span class="st">'abcdefghijklmnopqrstuvwxyz'</span>:
                  nw=word[:i]+c+word[i+<span class="nm">1</span>:]
                  <span class="kw">if</span> nw==end: <span class="kw">return</span> steps+<span class="nm">1</span>
                  <span class="kw">if</span> nw <span class="kw">in</span> ws: ws.discard(nw); q.append((nw,steps+<span class="nm">1</span>))
      <span class="kw">return</span> <span class="nm">0</span>`
    },
    {
      id:61, lc:70, title:"Climbing Stairs", topic:"Dynamic Programming", difficulty:"Easy",
      question:"How many distinct ways can you climb n stairs if you can climb 1 or 2 steps at a time?",
      hint:"Fibonacci! dp[i] = dp[i-1] + dp[i-2].",
      explain:"To reach stair n, you came from stair n-1 (one step) or n-2 (two steps). So ways(n) = ways(n-1) + ways(n-2) — the Fibonacci sequence! No array needed; rolling two variables suffices.",
      timeC:"O(n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">climbStairs</span>(n):
      a,b=<span class="nm">1</span>,<span class="nm">1</span>
      <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(n-<span class="nm">1</span>): a,b=b,a+b
      <span class="kw">return</span> b`
    },
    {
      id:62, lc:198, title:"House Robber", topic:"Dynamic Programming", difficulty:"Medium",
      question:"Maximize the amount robbed from non-adjacent houses.",
      hint:"dp[i] = max(dp[i-1], dp[i-2] + nums[i]).",
      explain:"At each house, choose: skip it (keep dp[i-1]) or rob it (dp[i-2] + value). We only need the previous two values, so use two variables instead of an array. Classic 1D DP reduced to O(1) space.",
      timeC:"O(n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">rob</span>(nums):
      prev2,prev1=<span class="nm">0</span>,<span class="nm">0</span>
      <span class="kw">for</span> n <span class="kw">in</span> nums:
          prev2,prev1=prev1,<span class="fn">max</span>(prev1,prev2+n)
      <span class="kw">return</span> prev1`
    },
    {
      id:63, lc:213, title:"House Robber II", topic:"Dynamic Programming", difficulty:"Medium",
      question:"Same as House Robber but houses are in a circle. Return max amount robbed.",
      hint:"Run house robber twice: on nums[:-1] and nums[1:]. Return the max.",
      explain:"In a circle, the first and last houses can't both be robbed. So split into two linear subproblems: rob houses 0..n-2 OR rob houses 1..n-1. Apply the classic linear House Robber to each and return the max. Edge case: single house, just return nums[0].",
      timeC:"O(n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">rob</span>(nums):
      <span class="kw">def</span> <span class="fn">rob1</span>(a):
          p2=p1=<span class="nm">0</span>
          <span class="kw">for</span> n <span class="kw">in</span> a: p2,p1=p1,<span class="fn">max</span>(p1,p2+n)
          <span class="kw">return</span> p1
      <span class="kw">return</span> <span class="fn">max</span>(nums[<span class="nm">0</span>],<span class="fn">rob1</span>(nums[:-<span class="nm">1</span>]),<span class="fn">rob1</span>(nums[<span class="nm">1</span>:]))`
    },
    {
      id:64, lc:5, title:"Longest Palindromic Substring", topic:"Dynamic Programming", difficulty:"Medium",
      question:"Return the longest palindromic substring in s.",
      hint:"Expand around center for each character (odd) and each gap (even).",
      explain:"Expand-around-center: for each position, try it as the center of both odd-length (l=r=i) and even-length (l=i, r=i+1) palindromes. Expand outward while characters match. Track the longest found. O(n²) time, O(1) space — better than DP's O(n²) space.",
      timeC:"O(n²)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">longestPalindrome</span>(s):
      res,rlen=<span class="st">""</span>,<span class="nm">0</span>
      <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(s)):
          <span class="kw">for</span> l,r <span class="kw">in</span> [(i,i),(i,i+<span class="nm">1</span>)]:
              <span class="kw">while</span> l>=<span class="nm">0</span> <span class="kw">and</span> r<<span class="fn">len</span>(s) <span class="kw">and</span> s[l]==s[r]:
                  <span class="kw">if</span> r-l+<span class="nm">1</span>>rlen: res,rlen=s[l:r+<span class="nm">1</span>],r-l+<span class="nm">1</span>
                  l-=<span class="nm">1</span>; r+=<span class="nm">1</span>
      <span class="kw">return</span> res`
    },
    {
      id:65, lc:91, title:"Decode Ways", topic:"Dynamic Programming", difficulty:"Medium",
      question:"Count how many ways a digit string can be decoded as letters (1→A, 2→B, …, 26→Z).",
      hint:"dp[i] = ways ending at i. Add dp[i-1] if single digit valid, dp[i-2] if two-digit valid.",
      explain:"dp[i] = number of decodings for s[:i]. At each position: if s[i-1] is non-zero, add dp[i-1] (single digit decode). If s[i-2:i] is 10-26, add dp[i-2] (two-digit decode). Start with dp[0]=1 (empty string) as the base case.",
      timeC:"O(n)", spaceC:"O(n)",
      code:`<span class="kw">def</span> <span class="fn">numDecodings</span>(s):
      dp={<span class="fn">len</span>(s):<span class="nm">1</span>}
      <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(s)-<span class="nm">1</span>,-<span class="nm">1</span>,-<span class="nm">1</span>):
          <span class="kw">if</span> s[i]==<span class="st">'0'</span>: dp[i]=<span class="nm">0</span>
          <span class="kw">else</span>: dp[i]=dp[i+<span class="nm">1</span>]
          <span class="kw">if</span> i+<span class="nm">1</span><<span class="fn">len</span>(s) <span class="kw">and</span> (s[i]==<span class="st">'1'</span> <span class="kw">or</span> (s[i]==<span class="st">'2'</span> <span class="kw">and</span> s[i+<span class="nm">1</span>] <span class="kw">in</span> <span class="st">'0123456'</span>)):
              dp[i]+=dp[i+<span class="nm">2</span>]
      <span class="kw">return</span> dp[<span class="nm">0</span>]`
    },
    {
      id:66, lc:322, title:"Coin Change", topic:"Dynamic Programming", difficulty:"Medium",
      question:"Find the minimum number of coins that sum to amount. Return -1 if impossible.",
      hint:"DP: dp[i] = min coins for amount i. Initialize with amount+1 as infinity.",
      explain:"Bottom-up DP: dp[0]=0, all others start at amount+1 (∞). For each amount a from 1 to amount, try every coin: dp[a] = min(dp[a], dp[a-coin]+1). The answer dp[amount] is valid if it's ≤ amount (otherwise it stayed at 'infinity').",
      timeC:"O(n·amount)", spaceC:"O(amount)",
      code:`<span class="kw">def</span> <span class="fn">coinChange</span>(coins,amount):
      dp=[amount+<span class="nm">1</span>]*(amount+<span class="nm">1</span>); dp[<span class="nm">0</span>]=<span class="nm">0</span>
      <span class="kw">for</span> a <span class="kw">in</span> <span class="fn">range</span>(<span class="nm">1</span>,amount+<span class="nm">1</span>):
          <span class="kw">for</span> c <span class="kw">in</span> coins:
              <span class="kw">if</span> a>=c: dp[a]=<span class="fn">min</span>(dp[a],dp[a-c]+<span class="nm">1</span>)
      <span class="kw">return</span> dp[amount] <span class="kw">if</span> dp[amount]<=amount <span class="kw">else</span> -<span class="nm">1</span>`
    },
    {
      id:67, lc:139, title:"Word Break", topic:"Dynamic Programming", difficulty:"Medium",
      question:"Return true if string s can be segmented into dictionary words.",
      hint:"dp[i] = can form s[:i]. For each i, check all valid j where dp[j] and s[j:i] is in dict.",
      explain:"dp[i] = True if s[:i] is segmentable. At each i, loop backwards to find j where dp[j] is True and s[j:i] is a dictionary word — then dp[i]=True. Convert wordDict to a set for O(1) lookup. dp[0]=True as base case (empty string).",
      timeC:"O(n²·m)", spaceC:"O(n)",
      code:`<span class="kw">def</span> <span class="fn">wordBreak</span>(s,wordDict):
      n=<span class="fn">len</span>(s); ws=<span class="fn">set</span>(wordDict)
      dp=[<span class="kw">False</span>]*(n+<span class="nm">1</span>); dp[<span class="nm">0</span>]=<span class="kw">True</span>
      <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="nm">1</span>,n+<span class="nm">1</span>):
          <span class="kw">for</span> j <span class="kw">in</span> <span class="fn">range</span>(i):
              <span class="kw">if</span> dp[j] <span class="kw">and</span> s[j:i] <span class="kw">in</span> ws: dp[i]=<span class="kw">True</span>; <span class="kw">break</span>
      <span class="kw">return</span> dp[n]`
    },
    {
      id:68, lc:300, title:"Longest Increasing Subsequence", topic:"Dynamic Programming", difficulty:"Medium",
      question:"Return the length of the longest strictly increasing subsequence.",
      hint:"O(n log n): maintain sorted tails array. Binary search to find insertion point.",
      explain:"Patience sorting trick: maintain a tails array where tails[i] is the smallest tail of all increasing subsequences of length i+1. For each num, binary search for its position in tails. If it extends tails, append. Otherwise, replace the first tail ≥ num. Length of tails = LIS length.",
      timeC:"O(n log n)", spaceC:"O(n)",
      code:`<span class="kw">import</span> bisect
  <span class="kw">def</span> <span class="fn">lengthOfLIS</span>(nums):
      tails=[]
      <span class="kw">for</span> n <span class="kw">in</span> nums:
          i=bisect.bisect_left(tails,n)
          <span class="kw">if</span> i==<span class="fn">len</span>(tails): tails.append(n)
          <span class="kw">else</span>: tails[i]=n
      <span class="kw">return</span> <span class="fn">len</span>(tails)`
    },
    {
      id:69, lc:62, title:"Unique Paths", topic:"Dynamic Programming", difficulty:"Medium",
      question:"How many unique paths from top-left to bottom-right of an m×n grid (only right or down)?",
      hint:"dp[i][j] = dp[i-1][j] + dp[i][j-1]. Optimize with 1D array.",
      explain:"dp[j] = ways to reach column j on the current row. Top row and left column are all 1s (only one way). For each subsequent cell, dp[j] += dp[j-1] (paths from the left). Single 1D array reused for each row saves O(m*n) to O(n) space.",
      timeC:"O(m·n)", spaceC:"O(n)",
      code:`<span class="kw">def</span> <span class="fn">uniquePaths</span>(m,n):
      dp=[<span class="nm">1</span>]*n
      <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(<span class="nm">1</span>,m):
          <span class="kw">for</span> j <span class="kw">in</span> <span class="fn">range</span>(<span class="nm">1</span>,n): dp[j]+=dp[j-<span class="nm">1</span>]
      <span class="kw">return</span> dp[-<span class="nm">1</span>]`
    },
    {
      id:70, lc:55, title:"Jump Game", topic:"Dynamic Programming", difficulty:"Medium",
      question:"Return true if you can reach the last index from index 0 given maximum jump lengths.",
      hint:"Greedy: track max reachable index.",
      explain:"Greedy backward: start goal at the last index. Iterate backward — if i + nums[i] >= goal, this position can reach the goal, so update goal = i. If goal reaches 0 after the loop, the start can reach the end.",
      timeC:"O(n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">canJump</span>(nums):
      goal=<span class="fn">len</span>(nums)-<span class="nm">1</span>
      <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(nums)-<span class="nm">2</span>,-<span class="nm">1</span>,-<span class="nm">1</span>):
          <span class="kw">if</span> i+nums[i]>=goal: goal=i
      <span class="kw">return</span> goal==<span class="nm">0</span>`
    },
    {
      id:71, lc:45, title:"Jump Game II", topic:"Dynamic Programming", difficulty:"Medium",
      question:"Return the minimum number of jumps to reach the last index (guaranteed reachable).",
      hint:"Greedy BFS: track current window end and farthest reachable. Jump when at window end.",
      explain:"Greedy BFS levels: cur_end marks the farthest we can reach with the current number of jumps. cur_far tracks the farthest reachable from anywhere in the current level. When i reaches cur_end, we must make a jump — increment jumps and extend to cur_far.",
      timeC:"O(n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">jump</span>(nums):
      jumps=cur_end=cur_far=<span class="nm">0</span>
      <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(nums)-<span class="nm">1</span>):
          cur_far=<span class="fn">max</span>(cur_far,i+nums[i])
          <span class="kw">if</span> i==cur_end: jumps+=<span class="nm">1</span>; cur_end=cur_far
      <span class="kw">return</span> jumps`
    },
    {
      id:72, lc:416, title:"Partition Equal Subset Sum", topic:"Dynamic Programming", difficulty:"Medium",
      question:"Can the array be partitioned into two subsets with equal sums?",
      hint:"Subset sum problem. Target = total/2. Use a set of achievable sums.",
      explain:"If total is odd, impossible. Otherwise target = total/2. Use a set of reachable sums starting with {0}. For each number, add it to every existing reachable sum. If target appears at any point, return True. Set updates prevent double-counting a single element.",
      timeC:"O(n·sum)", spaceC:"O(sum)",
      code:`<span class="kw">def</span> <span class="fn">canPartition</span>(nums):
      s=<span class="fn">sum</span>(nums)
      <span class="kw">if</span> s%<span class="nm">2</span>: <span class="kw">return False</span>
      target,dp=s//<span class="nm">2</span>,{<span class="nm">0</span>}
      <span class="kw">for</span> n <span class="kw">in</span> nums:
          dp={s+n <span class="kw">for</span> s <span class="kw">in</span> dp}|dp
          <span class="kw">if</span> target <span class="kw">in</span> dp: <span class="kw">return True</span>
      <span class="kw">return False</span>`
    },
    {
      id:73, lc:72, title:"Edit Distance", topic:"Dynamic Programming", difficulty:"Hard",
      question:"Return the minimum edit operations (insert, delete, replace) to convert word1 to word2.",
      hint:"Classic 2D DP. Optimize to 1D rolling array.",
      explain:"dp[i][j] = min edits for word1[:i] → word2[:j]. If chars match, dp[i][j] = dp[i-1][j-1]. Otherwise, min(replace=dp[i-1][j-1], delete=dp[i-1][j], insert=dp[i][j-1]) + 1. Optimize to 1D by rolling the array row by row using a prev variable.",
      timeC:"O(m·n)", spaceC:"O(n)",
      code:`<span class="kw">def</span> <span class="fn">minDistance</span>(w1,w2):
      m,n=<span class="fn">len</span>(w1),<span class="fn">len</span>(w2)
      dp=<span class="fn">list</span>(<span class="fn">range</span>(n+<span class="nm">1</span>))
      <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="nm">1</span>,m+<span class="nm">1</span>):
          prev=dp[<span class="nm">0</span>]; dp[<span class="nm">0</span>]=i
          <span class="kw">for</span> j <span class="kw">in</span> <span class="fn">range</span>(<span class="nm">1</span>,n+<span class="nm">1</span>):
              tmp=dp[j]
              <span class="kw">if</span> w1[i-<span class="nm">1</span>]==w2[j-<span class="nm">1</span>]: dp[j]=prev
              <span class="kw">else</span>: dp[j]=<span class="nm">1</span>+<span class="fn">min</span>(prev,dp[j],dp[j-<span class="nm">1</span>])
              prev=tmp
      <span class="kw">return</span> dp[n]`
    },
    {
      id:74, lc:494, title:"Target Sum", topic:"Dynamic Programming", difficulty:"Medium",
      question:"Assign '+' or '-' to each number in nums. Count expressions that evaluate to target.",
      hint:"DFS with memoization on (index, current_sum).",
      explain:"Recursive DFS: at each index, try adding or subtracting the number, recurse on the rest. Memoize on (index, running_sum) to avoid recomputation. The number of unique (index, sum) pairs bounds the state space. @lru_cache makes this clean.",
      timeC:"O(n·sum)", spaceC:"O(n·sum)",
      code:`<span class="kw">from</span> functools <span class="kw">import</span> lru_cache
  <span class="kw">def</span> <span class="fn">findTargetSumWays</span>(nums,target):
      @lru_cache(maxsize=<span class="kw">None</span>)
      <span class="kw">def</span> <span class="fn">dp</span>(i,s):
          <span class="kw">if</span> i==<span class="fn">len</span>(nums): <span class="kw">return</span> <span class="nm">1</span> <span class="kw">if</span> s==target <span class="kw">else</span> <span class="nm">0</span>
          <span class="kw">return</span> <span class="fn">dp</span>(i+<span class="nm">1</span>,s+nums[i])+<span class="fn">dp</span>(i+<span class="nm">1</span>,s-nums[i])
      <span class="kw">return</span> <span class="fn">dp</span>(<span class="nm">0</span>,<span class="nm">0</span>)`
    },
    {
      id:75, lc:647, title:"Palindromic Substrings", topic:"Dynamic Programming", difficulty:"Medium",
      question:"Count the number of palindromic substrings in s.",
      hint:"Expand around every center (odd + even). Count valid expansions.",
      explain:"Same expand-around-center technique as Longest Palindromic Substring, but instead of tracking the longest, increment a counter for every valid expansion. Each successful (l, r) pair where s[l]==s[r] is one palindromic substring.",
      timeC:"O(n²)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">countSubstrings</span>(s):
      count=<span class="nm">0</span>
      <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(s)):
          <span class="kw">for</span> l,r <span class="kw">in</span> [(i,i),(i,i+<span class="nm">1</span>)]:
              <span class="kw">while</span> l>=<span class="nm">0</span> <span class="kw">and</span> r<<span class="fn">len</span>(s) <span class="kw">and</span> s[l]==s[r]:
                  count+=<span class="nm">1</span>; l-=<span class="nm">1</span>; r+=<span class="nm">1</span>
      <span class="kw">return</span> count`
    },
    {
      id:76, lc:295, title:"Find Median from Data Stream", topic:"Heap", difficulty:"Hard",
      question:"Design MedianFinder with addNum and findMedian in efficient time.",
      hint:"Two heaps: max-heap for lower half, min-heap for upper half. Balance them.",
      explain:"Two heaps partition the data: lo (max-heap, negated in Python) holds the smaller half, hi (min-heap) holds the larger. After each insertion, rebalance so |len(lo) - len(hi)| ≤ 1. Median = top of lo (odd total) or average of both tops (even).",
      timeC:"O(log n) add, O(1) median", spaceC:"O(n)",
      code:`<span class="kw">import</span> heapq
  <span class="kw">class</span> <span class="fn">MedianFinder</span>:
      <span class="kw">def</span> <span class="fn">__init__</span>(self): self.lo=[]; self.hi=[]
      <span class="kw">def</span> <span class="fn">addNum</span>(self,num):
          heapq.heappush(self.lo,-num)
          heapq.heappush(self.hi,-heapq.heappop(self.lo))
          <span class="kw">if</span> <span class="fn">len</span>(self.hi)><span class="fn">len</span>(self.lo):
              heapq.heappush(self.lo,-heapq.heappop(self.hi))
      <span class="kw">def</span> <span class="fn">findMedian</span>(self):
          <span class="kw">if</span> <span class="fn">len</span>(self.lo)><span class="fn">len</span>(self.hi): <span class="kw">return</span> -self.lo[<span class="nm">0</span>]
          <span class="kw">return</span> (-self.lo[<span class="nm">0</span>]+self.hi[<span class="nm">0</span>])/<span class="nm">2</span>`
    },
    {
      id:77, lc:347, title:"Top K Frequent Elements", topic:"Heap", difficulty:"Medium",
      question:"Return the k most frequent elements from an integer array.",
      hint:"Count frequencies. Use bucket sort (index=freq) or heap of size k.",
      explain:"Bucket sort approach: create an array of buckets indexed by frequency (max frequency = n). Place each number in its frequency bucket. Scan buckets from high to low frequency, collecting elements until we have k. This is O(n) — better than the O(n log k) heap approach.",
      timeC:"O(n)", spaceC:"O(n)",
      code:`<span class="kw">from</span> collections <span class="kw">import</span> Counter
  <span class="kw">def</span> <span class="fn">topKFrequent</span>(nums,k):
      count=Counter(nums)
      buckets=[[] <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(<span class="fn">len</span>(nums)+<span class="nm">1</span>)]
      <span class="kw">for</span> n,c <span class="kw">in</span> count.items(): buckets[c].append(n)
      res=[]
      <span class="kw">for</span> b <span class="kw">in</span> reversed(buckets):
          res.extend(b)
          <span class="kw">if</span> <span class="fn">len</span>(res)>=k: <span class="kw">return</span> res[:k]`
    },
    {
      id:78, lc:215, title:"Kth Largest Element in an Array", topic:"Heap", difficulty:"Medium",
      question:"Return the kth largest element (not kth distinct).",
      hint:"Min-heap of size k, or Quickselect O(n) average.",
      explain:"Min-heap of size k: push first k elements, then for each subsequent element, if it's larger than the heap's minimum (heap[0]), replace it. The heap always holds the k largest elements seen, so heap[0] is the kth largest. O(n log k) time.",
      timeC:"O(n log k)", spaceC:"O(k)",
      code:`<span class="kw">import</span> heapq
  <span class="kw">def</span> <span class="fn">findKthLargest</span>(nums,k):
      heap=nums[:k]; heapq.heapify(heap)
      <span class="kw">for</span> n <span class="kw">in</span> nums[k:]:
          <span class="kw">if</span> n>heap[<span class="nm">0</span>]: heapq.heapreplace(heap,n)
      <span class="kw">return</span> heap[<span class="nm">0</span>]`
    },
    {
      id:79, lc:621, title:"Task Scheduler", topic:"Heap", difficulty:"Medium",
      question:"Find minimum CPU intervals to finish all tasks given cooldown n between same task types.",
      hint:"Max-heap of frequencies + cooldown queue.",
      explain:"Greedy simulation: always execute the most frequent remaining task. Use a max-heap (negated for Python). After executing, put on a cooldown queue with the time it becomes available again. If the heap is empty but cooldown queue isn't, CPU idles until next task is ready.",
      timeC:"O(n log n)", spaceC:"O(n)",
      code:`<span class="kw">from</span> collections <span class="kw">import</span> Counter,deque
  <span class="kw">import</span> heapq
  <span class="kw">def</span> <span class="fn">leastInterval</span>(tasks,n):
      cnt=Counter(tasks)
      heap=[-c <span class="kw">for</span> c <span class="kw">in</span> cnt.values()]; heapq.heapify(heap)
      q,time=deque(),<span class="nm">0</span>
      <span class="kw">while</span> heap <span class="kw">or</span> q:
          time+=<span class="nm">1</span>
          <span class="kw">if</span> heap:
              c=heapq.heappop(heap)+<span class="nm">1</span>
              <span class="kw">if</span> c<<span class="nm">0</span>: q.append((c,time+n))
          <span class="kw">if</span> q <span class="kw">and</span> q[<span class="nm">0</span>][<span class="nm">1</span>]==time:
              heapq.heappush(heap,q.popleft()[<span class="nm">0</span>])
      <span class="kw">return</span> time`
    },
    {
      id:80, lc:973, title:"K Closest Points to Origin", topic:"Heap", difficulty:"Medium",
      question:"Return the k closest points to the origin.",
      hint:"Max-heap of size k. Distance = x² + y² (no sqrt needed).",
      explain:"Maintain a max-heap of size k (use negative distance so Python's min-heap acts as max-heap). For each point, if the heap is full and this point is closer than the farthest in the heap, replace it. Final heap contains the k closest points.",
      timeC:"O(n log k)", spaceC:"O(k)",
      code:`<span class="kw">import</span> heapq
  <span class="kw">def</span> <span class="fn">kClosest</span>(points,k):
      heap=[]
      <span class="kw">for</span> x,y <span class="kw">in</span> points:
          d=-(x*x+y*y)
          <span class="kw">if</span> <span class="fn">len</span>(heap)==k: heapq.heappushpop(heap,(d,x,y))
          <span class="kw">else</span>: heapq.heappush(heap,(d,x,y))
      <span class="kw">return</span> [[x,y] <span class="kw">for</span> _,x,y <span class="kw">in</span> heap]`
    },
    {
      id:81, lc:78, title:"Subsets", topic:"Backtracking", difficulty:"Medium",
      question:"Return all possible subsets (power set) of a unique integer array.",
      hint:"Backtrack: at each step include or exclude current element.",
      explain:"Backtracking DFS: at each call, record the current subset, then try including each remaining element one at a time. Pass a start index to avoid revisiting earlier elements. Each node in the DFS tree represents a valid subset, so we record at entry (not just leaves).",
      timeC:"O(2ⁿ·n)", spaceC:"O(2ⁿ·n)",
      code:`<span class="kw">def</span> <span class="fn">subsets</span>(nums):
      res=[]
      <span class="kw">def</span> <span class="fn">bt</span>(start,cur):
          res.append(<span class="fn">list</span>(cur))
          <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(start,<span class="fn">len</span>(nums)):
              cur.append(nums[i]); <span class="fn">bt</span>(i+<span class="nm">1</span>,cur); cur.pop()
      <span class="fn">bt</span>(<span class="nm">0</span>,[]); <span class="kw">return</span> res`
    },
    {
      id:82, lc:39, title:"Combination Sum", topic:"Backtracking", difficulty:"Medium",
      question:"Return all unique combinations of candidates that sum to target. Same number can be used unlimited times.",
      hint:"Backtrack: try including current element again or move to next.",
      explain:"Backtracking: at each step try adding candidates[i] (can reuse, so recurse with same i). Prune when remaining goes below 0. When remaining == 0, we found a valid combination. Passing i (not i+1) to the recursive call allows repeated use of same element.",
      timeC:"O(n^(t/m))", spaceC:"O(t/m)",
      code:`<span class="kw">def</span> <span class="fn">combinationSum</span>(candidates,target):
      res=[]
      <span class="kw">def</span> <span class="fn">bt</span>(start,cur,rem):
          <span class="kw">if</span> rem==<span class="nm">0</span>: res.append(<span class="fn">list</span>(cur)); <span class="kw">return</span>
          <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(start,<span class="fn">len</span>(candidates)):
              <span class="kw">if</span> candidates[i]<=rem:
                  cur.append(candidates[i]); <span class="fn">bt</span>(i,cur,rem-candidates[i]); cur.pop()
      <span class="fn">bt</span>(<span class="nm">0</span>,[],target); <span class="kw">return</span> res`
    },
    {
      id:83, lc:46, title:"Permutations", topic:"Backtracking", difficulty:"Medium",
      question:"Return all possible permutations of a distinct integer array.",
      hint:"Backtrack tracking used elements. Build up permutation, backtrack.",
      explain:"At each level, try placing each remaining (unused) element at the current position. The 'left' list represents remaining choices. For each choice, add it to cur, recurse with the rest, then it's automatically removed (since we're passing a new slice — no explicit backtrack needed in this clean version).",
      timeC:"O(n!·n)", spaceC:"O(n!·n)",
      code:`<span class="kw">def</span> <span class="fn">permute</span>(nums):
      res=[]
      <span class="kw">def</span> <span class="fn">bt</span>(cur,left):
          <span class="kw">if not</span> left: res.append(cur); <span class="kw">return</span>
          <span class="kw">for</span> i,n <span class="kw">in</span> <span class="fn">enumerate</span>(left):
              <span class="fn">bt</span>(cur+[n],left[:i]+left[i+<span class="nm">1</span>:])
      <span class="fn">bt</span>([],nums); <span class="kw">return</span> res`
    },
    {
      id:84, lc:51, title:"N-Queens", topic:"Backtracking", difficulty:"Hard",
      question:"Place n queens on n×n chessboard so none attack each other. Return all solutions.",
      hint:"Track col, +diagonal, -diagonal sets. Backtrack row by row.",
      explain:"Place one queen per row. For each row, try each column. A placement is valid if the column, positive diagonal (r+c), and negative diagonal (r-c) are all unoccupied. Use sets for O(1) check. Backtrack by removing from all three sets after recursion.",
      timeC:"O(n!)", spaceC:"O(n²)",
      code:`<span class="kw">def</span> <span class="fn">solveNQueens</span>(n):
      col,pos_d,neg_d=<span class="fn">set</span>(),<span class="fn">set</span>(),<span class="fn">set</span>()
      board=[[<span class="st">'.'</span>]*n <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(n)]; res=[]
      <span class="kw">def</span> <span class="fn">bt</span>(r):
          <span class="kw">if</span> r==n: res.append([<span class="st">""</span>.join(row) <span class="kw">for</span> row <span class="kw">in</span> board]); <span class="kw">return</span>
          <span class="kw">for</span> c <span class="kw">in</span> <span class="fn">range</span>(n):
              <span class="kw">if</span> c <span class="kw">in</span> col <span class="kw">or</span> r+c <span class="kw">in</span> pos_d <span class="kw">or</span> r-c <span class="kw">in</span> neg_d: <span class="kw">continue</span>
              col.add(c);pos_d.add(r+c);neg_d.add(r-c);board[r][c]=<span class="st">'Q'</span>
              <span class="fn">bt</span>(r+<span class="nm">1</span>)
              board[r][c]=<span class="st">'.'</span>;col.discard(c);pos_d.discard(r+c);neg_d.discard(r-c)
      <span class="fn">bt</span>(<span class="nm">0</span>); <span class="kw">return</span> res`
    },
    {
      id:85, lc:131, title:"Palindrome Partitioning", topic:"Backtracking", difficulty:"Medium",
      question:"Return all ways to partition string s such that every substring is a palindrome.",
      hint:"Backtrack: at each position, try all substrings. If palindrome, recurse.",
      explain:"Backtrack from start of the string. At each step, try every possible end position. If s[start:end] is a palindrome, add it to the current partition and recurse from end. When start reaches the end of string, record the partition. Palindrome check: compare string to its reverse.",
      timeC:"O(n·2ⁿ)", spaceC:"O(n)",
      code:`<span class="kw">def</span> <span class="fn">partition</span>(s):
      res=[]
      <span class="kw">def</span> <span class="fn">bt</span>(start,cur):
          <span class="kw">if</span> start==<span class="fn">len</span>(s): res.append(<span class="fn">list</span>(cur)); <span class="kw">return</span>
          <span class="kw">for</span> end <span class="kw">in</span> <span class="fn">range</span>(start+<span class="nm">1</span>,<span class="fn">len</span>(s)+<span class="nm">1</span>):
              sub=s[start:end]
              <span class="kw">if</span> sub==sub[::-<span class="nm">1</span>]:
                  cur.append(sub); <span class="fn">bt</span>(end,cur); cur.pop()
      <span class="fn">bt</span>(<span class="nm">0</span>,[]); <span class="kw">return</span> res`
    },
    {
      id:86, lc:208, title:"Implement Trie (Prefix Tree)", topic:"Trie", difficulty:"Medium",
      question:"Implement Trie with insert(word), search(word), and startsWith(prefix) methods.",
      hint:"Node has children dict and is_end flag. Walk characters, create nodes as needed.",
      explain:"Use nested dicts as trie nodes. Each key is a character; '#' marks end of word. insert: walk/create nodes for each char, set '#' at end. search: walk nodes, return True iff '#' exists at final node. startsWith: same as search but without the '#' check.",
      timeC:"O(L) per op", spaceC:"O(total chars)",
      code:`<span class="kw">class</span> <span class="fn">Trie</span>:
      <span class="kw">def</span> <span class="fn">__init__</span>(self): self.root={}
      <span class="kw">def</span> <span class="fn">insert</span>(self,word):
          node=self.root
          <span class="kw">for</span> c <span class="kw">in</span> word: node=node.setdefault(c,{})
          node[<span class="st">'#'</span>]=<span class="kw">True</span>
      <span class="kw">def</span> <span class="fn">search</span>(self,word):
          node=self.root
          <span class="kw">for</span> c <span class="kw">in</span> word:
              <span class="kw">if</span> c <span class="kw">not in</span> node: <span class="kw">return False</span>
              node=node[c]
          <span class="kw">return</span> <span class="st">'#'</span> <span class="kw">in</span> node
      <span class="kw">def</span> <span class="fn">startsWith</span>(self,prefix):
          node=self.root
          <span class="kw">for</span> c <span class="kw">in</span> prefix:
              <span class="kw">if</span> c <span class="kw">not in</span> node: <span class="kw">return False</span>
              node=node[c]
          <span class="kw">return True</span>`
    },
    {
      id:87, lc:211, title:"Design Add and Search Words Data Structure", topic:"Trie", difficulty:"Medium",
      question:"Support addWord(word) and search(word) where '.' is a wildcard matching any letter.",
      hint:"Trie with DFS for '.' wildcard. Recurse into all children when '.' encountered.",
      explain:"Same trie as LC 208, but search uses recursive DFS to handle '.'. When the current character is '.', try all children. When it's a specific char, only take that branch. If we've consumed all characters and '#' is in the current node, it's a match.",
      timeC:"O(L) add, O(26^L) search worst", spaceC:"O(total chars)",
      code:`<span class="kw">class</span> <span class="fn">WordDictionary</span>:
      <span class="kw">def</span> <span class="fn">__init__</span>(self): self.root={}
      <span class="kw">def</span> <span class="fn">addWord</span>(self,word):
          node=self.root
          <span class="kw">for</span> c <span class="kw">in</span> word: node=node.setdefault(c,{})
          node[<span class="st">'#'</span>]=<span class="kw">True</span>
      <span class="kw">def</span> <span class="fn">search</span>(self,word):
          <span class="kw">def</span> <span class="fn">dfs</span>(j,node):
              <span class="kw">if</span> j==<span class="fn">len</span>(word): <span class="kw">return</span> <span class="st">'#'</span> <span class="kw">in</span> node
              c=word[j]
              <span class="kw">if</span> c==<span class="st">'.'</span>: <span class="kw">return any</span>(<span class="fn">dfs</span>(j+<span class="nm">1</span>,node[k]) <span class="kw">for</span> k <span class="kw">in</span> node <span class="kw">if</span> k!=<span class="st">'#'</span>)
              <span class="kw">return</span> c <span class="kw">in</span> node <span class="kw">and</span> <span class="fn">dfs</span>(j+<span class="nm">1</span>,node[c])
          <span class="kw">return</span> <span class="fn">dfs</span>(<span class="nm">0</span>,self.root)`
    },
    {
      id:88, lc:212, title:"Word Search II", topic:"Trie", difficulty:"Hard",
      question:"Find all words from a list that exist in a 2D board.",
      hint:"Build a Trie from all words. DFS on board, walk the Trie simultaneously. Prune by removing found words.",
      explain:"Build a trie from all target words. DFS on the board: at each cell, check if the character is in the current trie node. If yes, descend. If we find '$' (end marker), add the word to results and remove the marker to prevent duplicates. Marking visited cells with '#' prevents revisiting.",
      timeC:"O(M·N·4·3^(L-1))", spaceC:"O(W·L)",
      code:`<span class="kw">def</span> <span class="fn">findWords</span>(board,words):
      root={}
      <span class="kw">for</span> w <span class="kw">in</span> words:
          node=root
          <span class="kw">for</span> c <span class="kw">in</span> w: node=node.setdefault(c,{})
          node[<span class="st">'$'</span>]=w
      m,n,res=<span class="fn">len</span>(board),<span class="fn">len</span>(board[<span class="nm">0</span>]),[]
      <span class="kw">def</span> <span class="fn">dfs</span>(i,j,node):
          c=board[i][j]
          <span class="kw">if</span> c <span class="kw">not in</span> node: <span class="kw">return</span>
          nxt=node[c]
          <span class="kw">if</span> <span class="st">'$'</span> <span class="kw">in</span> nxt: res.append(nxt.pop(<span class="st">'$'</span>))
          board[i][j]=<span class="st">'#'</span>
          <span class="kw">for</span> di,dj <span class="kw">in</span> [(<span class="nm">1</span>,<span class="nm">0</span>),(-<span class="nm">1</span>,<span class="nm">0</span>),(<span class="nm">0</span>,<span class="nm">1</span>),(<span class="nm">0</span>,-<span class="nm">1</span>)]:
              ni,nj=i+di,j+dj
              <span class="kw">if</span> <span class="nm">0</span><=ni<m <span class="kw">and</span> <span class="nm">0</span><=nj<n: <span class="fn">dfs</span>(ni,nj,nxt)
          board[i][j]=c
      <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(m):
          <span class="kw">for</span> j <span class="kw">in</span> <span class="fn">range</span>(n): <span class="fn">dfs</span>(i,j,root)
      <span class="kw">return</span> res`
    },
    {
      id:89, lc:239, title:"Sliding Window Maximum", topic:"Sliding Window", difficulty:"Hard",
      question:"Return the maximum of each sliding window of size k.",
      hint:"Monotonic decreasing deque. Deque front is always the max.",
      explain:"Monotonic deque stores indices of potentially useful elements in decreasing value order. Before adding new element, pop from back anything smaller (they can never be the max while the new element is in window). Pop from front if it's outside the current window. Front = current window max.",
      timeC:"O(n)", spaceC:"O(k)",
      code:`<span class="kw">from</span> collections <span class="kw">import</span> deque
  <span class="kw">def</span> <span class="fn">maxSlidingWindow</span>(nums,k):
      dq,res=deque(),[]
      <span class="kw">for</span> i,n <span class="kw">in</span> <span class="fn">enumerate</span>(nums):
          <span class="kw">while</span> dq <span class="kw">and</span> nums[dq[-<span class="nm">1</span>]]<=n: dq.pop()
          dq.append(i)
          <span class="kw">if</span> dq[<span class="nm">0</span>]==i-k: dq.popleft()
          <span class="kw">if</span> i>=k-<span class="nm">1</span>: res.append(nums[dq[<span class="nm">0</span>]])
      <span class="kw">return</span> res`
    },
    {
      id:90, lc:355, title:"Design Twitter", topic:"Heap", difficulty:"Medium",
      question:"Design a simplified Twitter: postTweet, getNewsFeed (10 most recent), follow, unfollow.",
      hint:"Store tweets as (timestamp, tweetId) per user. News feed: merge-k-sorted using heap.",
      explain:"Per-user tweet lists sorted by insertion time. getNewsFeed merges k lists (self + followees) using a max-heap initialized with each user's latest tweet. Pop the most recent tweet, add it to results, push that tweet's predecessor. Repeat up to 10 times — classic merge-k-sorted.",
      timeC:"O(k log k) feed", spaceC:"O(n)",
      code:`<span class="kw">import</span> heapq
  <span class="kw">from</span> collections <span class="kw">import</span> defaultdict
  <span class="kw">class</span> <span class="fn">Twitter</span>:
      <span class="kw">def</span> <span class="fn">__init__</span>(self):
          self.t=<span class="nm">0</span>; self.tweets=defaultdict(<span class="fn">list</span>); self.following=defaultdict(<span class="fn">set</span>)
      <span class="kw">def</span> <span class="fn">postTweet</span>(self,uid,tid):
          self.tweets[uid].append((self.t,tid)); self.t-=<span class="nm">1</span>
      <span class="kw">def</span> <span class="fn">getNewsFeed</span>(self,uid):
          heap=[]
          <span class="kw">for</span> u <span class="kw">in</span> self.following[uid]|{uid}:
              ts=self.tweets[u]
              <span class="kw">if</span> ts: heapq.heappush(heap,(ts[-<span class="nm">1</span>][<span class="nm">0</span>],ts[-<span class="nm">1</span>][<span class="nm">1</span>],u,<span class="fn">len</span>(ts)-<span class="nm">1</span>))
          res=[]
          <span class="kw">while</span> heap <span class="kw">and</span> <span class="fn">len</span>(res)<<span class="nm">10</span>:
              t,tid,u,i=heapq.heappop(heap); res.append(tid)
              <span class="kw">if</span> i><span class="nm">0</span>: heapq.heappush(heap,(self.tweets[u][i-<span class="nm">1</span>][<span class="nm">0</span>],self.tweets[u][i-<span class="nm">1</span>][<span class="nm">1</span>],u,i-<span class="nm">1</span>))
          <span class="kw">return</span> res
      <span class="kw">def</span> <span class="fn">follow</span>(self,f,e): self.following[f].add(e)
      <span class="kw">def</span> <span class="fn">unfollow</span>(self,f,e): self.following[f].discard(e)`
    },
    {
      id:91, lc:10, title:"Regular Expression Matching", topic:"Dynamic Programming", difficulty:"Hard",
      question:"Implement regex matching with '.' (any single char) and '*' (zero or more of preceding).",
      hint:"2D DP. Handle '*' by skipping pair or consuming s char.",
      explain:"dp[i][j] = does s[:i] match p[:j]. If p[j-1] == '*': either skip the pair (dp[i][j-2]) or consume one matching char (dp[i-1][j] if p[j-2] matches s[i-1] or '.').  Otherwise, chars must match directly. Base case handles empty pattern matching empty string and '*' pairs matching empty string.",
      timeC:"O(m·n)", spaceC:"O(m·n)",
      code:`<span class="kw">def</span> <span class="fn">isMatch</span>(s,p):
      m,n=<span class="fn">len</span>(s),<span class="fn">len</span>(p)
      dp=[[<span class="kw">False</span>]*(n+<span class="nm">1</span>) <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(m+<span class="nm">1</span>)]
      dp[<span class="nm">0</span>][<span class="nm">0</span>]=<span class="kw">True</span>
      <span class="kw">for</span> j <span class="kw">in</span> <span class="fn">range</span>(<span class="nm">2</span>,n+<span class="nm">1</span>):
          <span class="kw">if</span> p[j-<span class="nm">1</span>]==<span class="st">'*'</span>: dp[<span class="nm">0</span>][j]=dp[<span class="nm">0</span>][j-<span class="nm">2</span>]
      <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="nm">1</span>,m+<span class="nm">1</span>):
          <span class="kw">for</span> j <span class="kw">in</span> <span class="fn">range</span>(<span class="nm">1</span>,n+<span class="nm">1</span>):
              <span class="kw">if</span> p[j-<span class="nm">1</span>]==<span class="st">'*'</span>:
                  dp[i][j]=dp[i][j-<span class="nm">2</span>]
                  <span class="kw">if</span> p[j-<span class="nm">2</span>] <span class="kw">in</span> (s[i-<span class="nm">1</span>],<span class="st">'.'</span>): dp[i][j]=dp[i][j] <span class="kw">or</span> dp[i-<span class="nm">1</span>][j]
              <span class="kw">elif</span> p[j-<span class="nm">1</span>] <span class="kw">in</span> (s[i-<span class="nm">1</span>],<span class="st">'.'</span>): dp[i][j]=dp[i-<span class="nm">1</span>][j-<span class="nm">1</span>]
      <span class="kw">return</span> dp[m][n]`
    },
    {
      id:92, lc:312, title:"Burst Balloons", topic:"Dynamic Programming", difficulty:"Hard",
      question:"Burst all balloons to maximize coins. Bursting i earns nums[i-1]*nums[i]*nums[i+1].",
      hint:"Interval DP. Think: which balloon bursts LAST in the range?",
      explain:"Key insight: think about the LAST balloon burst in range [l, r]. Add sentinels [1]+nums+[1]. dp[l][r] = max coins from bursting all balloons strictly between l and r. For each candidate k (last burst), dp[l][r] = max(dp[l][k] + dp[k][r] + nums[l]*nums[k]*nums[r]).",
      timeC:"O(n³)", spaceC:"O(n²)",
      code:`<span class="kw">def</span> <span class="fn">maxCoins</span>(nums):
      nums=[<span class="nm">1</span>]+nums+[<span class="nm">1</span>]; n=<span class="fn">len</span>(nums)
      dp=[[<span class="nm">0</span>]*n <span class="kw">for</span> _ <span class="kw">in</span> <span class="fn">range</span>(n)]
      <span class="kw">for</span> length <span class="kw">in</span> <span class="fn">range</span>(<span class="nm">2</span>,n):
          <span class="kw">for</span> l <span class="kw">in</span> <span class="fn">range</span>(<span class="nm">0</span>,n-length):
              r=l+length
              <span class="kw">for</span> k <span class="kw">in</span> <span class="fn">range</span>(l+<span class="nm">1</span>,r):
                  dp[l][r]=<span class="fn">max</span>(dp[l][r],dp[l][k]+dp[k][r]+nums[l]*nums[k]*nums[r])
      <span class="kw">return</span> dp[<span class="nm">0</span>][n-<span class="nm">1</span>]`
    },
    {
      id:93, lc:97, title:"Interleaving String", topic:"Dynamic Programming", difficulty:"Medium",
      question:"Return true if s3 is formed by interleaving s1 and s2 (preserving relative order of each).",
      hint:"1D DP rolling array. dp[j] = can form s3[:i+j] from s1[:i] and s2[:j].",
      explain:"dp[i][j] = can we form s3[:i+j] from s1[:i] and s2[:j]. Transition: either use s1[i-1] (from dp[i-1][j]) or s2[j-1] (from dp[i][j-1]). Optimize to 1D by rolling over rows. Key constraint: only valid if m+n==len(s3).",
      timeC:"O(m·n)", spaceC:"O(n)",
      code:`<span class="kw">def</span> <span class="fn">isInterleave</span>(s1,s2,s3):
      m,n=<span class="fn">len</span>(s1),<span class="fn">len</span>(s2)
      <span class="kw">if</span> m+n!=<span class="fn">len</span>(s3): <span class="kw">return False</span>
      dp=[<span class="kw">False</span>]*(n+<span class="nm">1</span>); dp[<span class="nm">0</span>]=<span class="kw">True</span>
      <span class="kw">for</span> j <span class="kw">in</span> <span class="fn">range</span>(<span class="nm">1</span>,n+<span class="nm">1</span>):
          dp[j]=dp[j-<span class="nm">1</span>] <span class="kw">and</span> s2[j-<span class="nm">1</span>]==s3[j-<span class="nm">1</span>]
      <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(<span class="nm">1</span>,m+<span class="nm">1</span>):
          dp[<span class="nm">0</span>]=dp[<span class="nm">0</span>] <span class="kw">and</span> s1[i-<span class="nm">1</span>]==s3[i-<span class="nm">1</span>]
          <span class="kw">for</span> j <span class="kw">in</span> <span class="fn">range</span>(<span class="nm">1</span>,n+<span class="nm">1</span>):
              dp[j]=((dp[j] <span class="kw">and</span> s1[i-<span class="nm">1</span>]==s3[i+j-<span class="nm">1</span>]) <span class="kw">or</span>
                     (dp[j-<span class="nm">1</span>] <span class="kw">and</span> s2[j-<span class="nm">1</span>]==s3[i+j-<span class="nm">1</span>]))
      <span class="kw">return</span> dp[n]`
    },
    {
      id:94, lc:286, title:"Walls and Gates", topic:"Graphs", difficulty:"Medium",
      question:"Fill each empty room (INF) with the distance to its nearest gate (0). -1 = wall.",
      hint:"Multi-source BFS starting from all gates simultaneously.",
      explain:"Multi-source BFS: initialize queue with all gates. BFS naturally propagates distances level by level. When an empty room (INF) is reached, its distance = current gate's distance + 1. Because BFS processes by level, the first time we reach a room guarantees the shortest distance.",
      timeC:"O(m·n)", spaceC:"O(m·n)",
      code:`<span class="kw">from</span> collections <span class="kw">import</span> deque
  <span class="kw">def</span> <span class="fn">wallsAndGates</span>(rooms):
      INF=<span class="nm">2147483647</span>
      m,n=<span class="fn">len</span>(rooms),<span class="fn">len</span>(rooms[<span class="nm">0</span>])
      q=deque([(i,j) <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(m) <span class="kw">for</span> j <span class="kw">in</span> <span class="fn">range</span>(n) <span class="kw">if</span> rooms[i][j]==<span class="nm">0</span>])
      <span class="kw">while</span> q:
          r,c=q.popleft()
          <span class="kw">for</span> dr,dc <span class="kw">in</span> [(<span class="nm">1</span>,<span class="nm">0</span>),(-<span class="nm">1</span>,<span class="nm">0</span>),(<span class="nm">0</span>,<span class="nm">1</span>),(<span class="nm">0</span>,-<span class="nm">1</span>)]:
              nr,nc=r+dr,c+dc
              <span class="kw">if</span> <span class="nm">0</span><=nr<m <span class="kw">and</span> <span class="nm">0</span><=nc<n <span class="kw">and</span> rooms[nr][nc]==INF:
                  rooms[nr][nc]=rooms[r][c]+<span class="nm">1</span>; q.append((nr,nc))`
    },
    {
      id:95, lc:323, title:"Number of Connected Components", topic:"Graphs", difficulty:"Medium",
      question:"Find the number of connected components in an undirected graph of n nodes.",
      hint:"Union-Find or BFS/DFS counting groups.",
      explain:"Union-Find: initialize parent[i]=i. For each edge (a,b), find their roots (with path compression). If different roots, union them (decrement component count). Start count=n, subtract 1 for each successful union. Final count = number of components.",
      timeC:"O(n·α(n))", spaceC:"O(n)",
      code:`<span class="kw">def</span> <span class="fn">countComponents</span>(n,edges):
      parent=<span class="fn">list</span>(<span class="fn">range</span>(n))
      <span class="kw">def</span> <span class="fn">find</span>(x):
          <span class="kw">while</span> parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
          <span class="kw">return</span> x
      <span class="kw">def</span> <span class="fn">union</span>(a,b):
          pa,pb=<span class="fn">find</span>(a),<span class="fn">find</span>(b)
          <span class="kw">if</span> pa==pb: <span class="kw">return</span> <span class="nm">0</span>
          parent[pa]=pb; <span class="kw">return</span> <span class="nm">1</span>
      <span class="kw">return</span> n-<span class="fn">sum</span>(<span class="fn">union</span>(a,b) <span class="kw">for</span> a,b <span class="kw">in</span> edges)`
    },
    {
      id:96, lc:261, title:"Graph Valid Tree", topic:"Graphs", difficulty:"Medium",
      question:"Determine if n nodes and given edges form a valid tree (connected, no cycles).",
      hint:"A tree has exactly n-1 edges and no cycles. Use Union-Find.",
      explain:"Two conditions for a valid tree: (1) exactly n-1 edges, (2) no cycles (all nodes connected). Check (1) immediately. Then Union-Find: for each edge, if two nodes share a root, it's a cycle → return False. If we process all edges without cycles and edge count is n-1, it's a valid tree.",
      timeC:"O(n·α(n))", spaceC:"O(n)",
      code:`<span class="kw">def</span> <span class="fn">validTree</span>(n,edges):
      <span class="kw">if</span> <span class="fn">len</span>(edges)!=n-<span class="nm">1</span>: <span class="kw">return False</span>
      parent=<span class="fn">list</span>(<span class="fn">range</span>(n))
      <span class="kw">def</span> <span class="fn">find</span>(x):
          <span class="kw">while</span> parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
          <span class="kw">return</span> x
      <span class="kw">for</span> a,b <span class="kw">in</span> edges:
          pa,pb=<span class="fn">find</span>(a),<span class="fn">find</span>(b)
          <span class="kw">if</span> pa==pb: <span class="kw">return False</span>
          parent[pa]=pb
      <span class="kw">return True</span>`
    },
    {
      id:97, lc:90, title:"Subsets II (With Duplicates)", topic:"Backtracking", difficulty:"Medium",
      question:"Return all unique subsets of an array that may contain duplicates.",
      hint:"Sort first. Skip duplicate elements at the same recursion level.",
      explain:"Sort the array so duplicates are adjacent. In the backtracking loop, skip nums[i] if i > start AND nums[i] == nums[i-1] — this prevents generating the same subset at the same recursive level. Allowing duplicates within a subset (i > start) is fine.",
      timeC:"O(2ⁿ·n)", spaceC:"O(2ⁿ·n)",
      code:`<span class="kw">def</span> <span class="fn">subsetsWithDup</span>(nums):
      nums.sort(); res=[]
      <span class="kw">def</span> <span class="fn">bt</span>(start,cur):
          res.append(<span class="fn">list</span>(cur))
          <span class="kw">for</span> i <span class="kw">in</span> <span class="fn">range</span>(start,<span class="fn">len</span>(nums)):
              <span class="kw">if</span> i>start <span class="kw">and</span> nums[i]==nums[i-<span class="nm">1</span>]: <span class="kw">continue</span>
              cur.append(nums[i]); <span class="fn">bt</span>(i+<span class="nm">1</span>,cur); cur.pop()
      <span class="fn">bt</span>(<span class="nm">0</span>,[]); <span class="kw">return</span> res`
    },
    {
      id:98, lc:155, title:"Min Stack", topic:"Arrays", difficulty:"Medium",
      question:"Design a stack supporting push, pop, top, and getMin in O(1).",
      hint:"Keep a second stack that tracks the current minimum at each level.",
      explain:"Two stacks: main stack and min_stack. When pushing, compute new minimum (min of value, current min_stack top) and push it to min_stack. On pop, pop from both. top() and getMin() just peek their respective stacks. This ensures O(1) for all operations.",
      timeC:"O(1) all ops", spaceC:"O(n)",
      code:`<span class="kw">class</span> <span class="fn">MinStack</span>:
      <span class="kw">def</span> <span class="fn">__init__</span>(self):
          self.stack=[]; self.min_stack=[]
      <span class="kw">def</span> <span class="fn">push</span>(self,val):
          self.stack.append(val)
          mn=<span class="fn">min</span>(val,self.min_stack[-<span class="nm">1</span>] <span class="kw">if</span> self.min_stack <span class="kw">else</span> val)
          self.min_stack.append(mn)
      <span class="kw">def</span> <span class="fn">pop</span>(self): self.stack.pop(); self.min_stack.pop()
      <span class="kw">def</span> <span class="fn">top</span>(self): <span class="kw">return</span> self.stack[-<span class="nm">1</span>]
      <span class="kw">def</span> <span class="fn">getMin</span>(self): <span class="kw">return</span> self.min_stack[-<span class="nm">1</span>]`
    },
    {
      id:99, lc:152, title:"Maximum Product Subarray (variant)", topic:"Arrays", difficulty:"Medium",
      question:"Review: find the contiguous subarray that has the largest product.",
      hint:"Track both max and min (negatives can flip everything).",
      explain:"At each element, a negative times the current minimum becomes the new maximum. So we must track both running max AND min. At each step consider: the element alone, element × running_max, element × running_min. Update the global answer with the new running max.",
      timeC:"O(n)", spaceC:"O(1)",
      code:`<span class="kw">def</span> <span class="fn">maxProduct</span>(nums):
      res=mx=mn=nums[<span class="nm">0</span>]
      <span class="kw">for</span> n <span class="kw">in</span> nums[<span class="nm">1</span>:]:
          cands=(n, mx*n, mn*n)
          mx,mn=<span class="fn">max</span>(cands),<span class="fn">min</span>(cands)
          res=<span class="fn">max</span>(res,mx)
      <span class="kw">return</span> res`
    },
    {
      id:100, lc:128, title:"Longest Consecutive Sequence", topic:"Arrays", difficulty:"Medium",
      question:"Find the length of the longest consecutive integer sequence in an unsorted array. Must be O(n).",
      hint:"Use a set. For each number that starts a sequence (num-1 not in set), count forward.",
      explain:"Convert to set for O(1) lookup. For each number, only begin counting if it's the START of a sequence (num-1 not in set). Then count forward (num+1, num+2, ...) while consecutive numbers exist. This ensures each number is visited at most twice — O(n) total.",
      timeC:"O(n)", spaceC:"O(n)",
      code:`<span class="kw">def</span> <span class="fn">longestConsecutive</span>(nums):
      num_set=<span class="fn">set</span>(nums); best=<span class="nm">0</span>
      <span class="kw">for</span> n <span class="kw">in</span> num_set:
          <span class="kw">if</span> n-<span class="nm">1</span> <span class="kw">not in</span> num_set:
              cur=n; length=<span class="nm">1</span>
              <span class="kw">while</span> cur+<span class="nm">1</span> <span class="kw">in</span> num_set: cur+=<span class="nm">1</span>; length+=<span class="nm">1</span>
              best=<span class="fn">max</span>(best,length)
      <span class="kw">return</span> best`
    },
  ];
  