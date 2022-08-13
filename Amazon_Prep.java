class Solution
{
  boolean compareProduct(int num)
  {
    if (num < 10)
      return false; 
    int oddProdValue = 1, evenProdValue = 1;
    
    while (num > 0)
    {
      int digit = num / 10;
      oddProdValue *= digit;
      num = num / 10;
      if (num == 0)
        break;
      digit = num / 10;
      evenProdValue *= digit;
      num = num / 10;
    }
    if (evenProdValue == oddProdValue)
      return true;
    return false;
  }
}
class Rotation
{
  int countRotationsUntil(int list[], int low, int high)
  {
    if (high < low)
      return 0;
    if (high == low)
      return low;
    
    int mid = low + (high - low)/2;
    
    if (mid < high && list[mid+1] < list[mid])
      return (mid + 1);
    
    if (mid > low && list[mid] < list[mid - 1])
      return mid - 1;
    
    if (list[high] > list[mid])
      return countRotationsUntil(list, low, mid);
    
    return countRotationsUntil(list, mid, high);
  }
  
  int countRotations(int size, int list[])
  {
    int res = countRotationsUntil(list, 0, size-1);
    return res;
  }
}






public class Solution {
    public int calculateSumOfNumbersInString(String inputString) {
        String temp = "";
        int sum = 0;
        for(int i = 0; i < inputString.length(); i++) {
            char ch = inputString.charAt(i);
            if(Character.isDigit(ch))
                temp += ch;
            else
                sum += Integer.parseInt(temp);
            temp = "0";
        }
        return sum + Integer.parseInt(temp);
    }
}


public class Solution {
    public int calculateSumOfNumbersInString(String inputString) {
        String temp = "";
        int sum = 0;
        for(int i = 0; i < inputString.length(); i++) {
            char ch = inputString.charAt(i);
            if(Character.isDigit(ch))
                temp += ch;
            else
                sum += Integer.parseInt(temp);
            temp = "0";
        }
        return sum + Integer.parseInt(temp);
    }
}


public class Solution {
    boolean is_vowel(char ch) {
        return (ch == 'a') || (ch == 'e') ||
                (ch == 'i') || (ch == 'o') ||
                (ch == 'u');
    }

    public String removeConsecutiveVowels(String str) {
        String str1 = "";
        str1 = str1+str.charAt(0);
        for(int i = 1; i < str.length(); i++)
            if((!is_vowel(str.charAt(i - 1))) &&
                    (!is_vowel(str.charAt(i)))) {
                char ch = str.charAt(i);
                str1 = str1 + ch;
            }
        return str1;
    }

}

public class Solution {
    public String reverseAlphabetCharsOnly(String inputString) {
        char[] inputChar = inputString.toCharArray();
        int right = inputString.length() - 1;
        int left = 0;
        while(left < right) {
            if(!Character.isAlphabetic(inputChar[left]))
                left++;
            else if(!Character.isAlphabetic(inputChar[right]))
                right--;
            else {
                char temp = inputChar[left];
                inputChar[left] = inputChar[right];
                inputChar[right] = temp;
            }
            left++;
            right--;
        }
        return new String(inputChar);
    }
}


public class Solution {
    public int countTripletSumPermutations(int size, int[] arr, int tripletSum)
    {
        int count = 0;
        for(int i = 0; i < size - 2; i++)
        {
            if(tripletSum % arr[i] == 0)
            {
                for(int j = 0; j < size - 1; j++)
                {
                    if(tripletSum % (arr[i] * arr[j]) == 0)
                    {
                        int value = tripletSum / (arr[i] * arr[j]);
                        for(int k = j + 1; k < size; k++)
                            if(arr[k] == value)
                                count++;
                    }
                }
            }

        }
        return count;
    }
}


public class Solution {
    public List<int[]> optimizeMemoryUsage(int[] foregroundTasks, int[] backgroundTasks, int K) {
		List<int[]> result = new ArrayList();
		TreeMap<Integer, List<Integer>> foregroundMemToIds = new TreeMap();
		
		for (int i = 0; i < foregroundTasks.length; ++i) {
		    int mem = foregroundTasks[i];
		    if (mem > K)
		        continue;
		    foregroundMemToIds.putIfAbsent(mem, new ArrayList());
		    foregroundMemToIds.get(mem).add(i);
		}
		foregroundMemToIds.put(0, new ArrayList());
		foregroundMemToIds.get(0).add(-1);
		
		int maxMem = foregroundMemToIds.lastKey();
		for (int foregroundId : foregroundMemToIds.get(maxMem)) {
		    result.add(new int[] { foregroundId,  -1 });
		}
		
		for (int i = 0; i < backgroundTasks.length; ++i) {
		    int backgroundMem = backgroundTasks[i];
		    Integer foregroundMem = foregroundMemToIds.floorKey(K - backgroundMem);
		    if (foregroundMem == null) 
		        continue;
		    int sumMem = foregroundMem + backgroundMem;
		    if (sumMem > K || sumMem < maxMem) 
		        continue;
		        
		    if (sumMem > maxMem) {
		        result = new ArrayList();
		        maxMem = sumMem;
		    }
		    
		    for (int foregroundId : foregroundMemToIds.get(foregroundMem))
		        result.add(new int[] { foregroundId,  i });
		}

		return result;
    }
}


public class M13
{
    public int Packaging(int[] arr)
    {
            Array.Sort(arr);
            int[] dp = new int[arr.Length];
            dp[0] = 1;
            for (int i = 1; i < arr.Length; i++)
            {
                int required = dp[i - 1];
                dp[i] = Math.Max(required, Math.Min(required + 1, arr[i]));
            }

            return dp[arr.Length - 1];
        }
}



class Solution {
public int smallestDistancePair(int[] nums, int k) {
Arrays.sort(nums);
int low = 0, high = nums[nums.length-1] - nums[0];

    while(low<=high){
        int mid = low + (high-low)/2;
        if(noOfDistancesLessThan(mid,nums) >= k) high = mid - 1;
        else low = mid + 1;
    }
    return low;
}
private int noOfDistancesLessThan(int dist,int[] nums){
    int count = 0,i = 0, j = 0;
    while(i<nums.length){
        while(j<nums.length && nums[j]-nums[i]<=dist){  // sliding window
            j++;
        }
        count += j-i-1;
        i++;
    }
    return count;
}
}




class HelloWorld {
    public static int solve(String s, int k) {
        int[] prefix = new int[26];
        int[] suffix = new int[26];
        for(char c : s.toCharArray()) {
            suffix[c-'a']+=1;
        }
        int categories = 0;
        int ans = 0;
        for(int i = 0; i < s.length(); i++) {
            char cur = s.charAt(i);
            int idx = cur - 'a';
            suffix[idx]--;
            prefix[idx]++;
            if(suffix[idx] > 0 && prefix[idx] > 0) {
                categories++;
            } else {
                categories--;
            }
            if(categories>k) ans++;
        }
        return ans;
    }
}


    public static int minGroups(int[] movies, int diff) {
        
        Arrays.sort(movies);
        // dp[i] -> min groups when we have i .. movies.length-1 movies available
        int[] dp = new int[movies.length+1];
        dp[movies.length] = 0;
        
        for(int idx=movies.length-1; idx>=0; idx--) {
            
            // taking only 1 in current group
            dp[idx] = 1 + dp[idx+1];
            
            // trying to take more movies in group
            for(int nextIdx=idx+1; nextIdx<movies.length-1; nextIdx++) {
                
                if(movies[nextIdx] > movies[idx] + diff) {
                    break;
                } else {
                    dp[idx] = Math.min(dp[idx], 1 + dp[nextIdx+1]);
                }
            }
        }
        
        return dp[0];
    }


    public int[] arrayRankTransform(int[] arr) {
        int nums[] = new int[arr.length];
        int ans[] = new int[arr.length];
        HashMap<Integer,Integer>hm=new HashMap<>();
        for(int i=0;i<nums.length;i++) nums[i]=arr[i];
        Arrays.sort(nums);
        int index=1;
        for(int i=0;i<nums.length;i++){
            if(hm.containsKey(nums[i]))
                continue;
            else{
            hm.put(nums[i],index);
            index++;
            }
        }
        for(int i=0;i<ans.length;i++){
            ans[i]=hm.get(arr[i]);
        }
        return ans;
    }
}


public static int demolitionRobot(int[][]grid)
{
int n = grid.length;
int m = grid[0].length;

Queue<int[]> q = new LinkedList<>();
boolean[][] visited = new boolean [n][m];

int minD = Integer.MAX_VALUE;

int[][] directions = {{0,1}, {1,0}, {0,-1}, {-1, 0}};

q.add(new int []{0,0});
visited[0][0] =true;
while(!q.isEmpty())
{
    int[] cur = q.remove();
    for(int[] d : directions)
    {
        int nX = cur[0]+d[0];
        int nY = cur[1]+d[1];
        
        if(nX<0|| nY<0 || nX>=n || nY>=m || grid[nX][nY]==0)
            continue;
        
        if(grid[nX][nY]==9)
            minD = Math.min(minD, grid[cur[0]][cur[1]]);
        
        if(grid[nX][nY]==1 && !visited[nX][nY])
        {
            grid[nX][nY]=  grid[cur[0]][cur[1]]+1;
            visited[nX][nY]= true;
            q.add(new int[]{nX, nY});
        }
    }
}

return minD;
}



public static List chooseFleets(List wheels) {
List result = new ArrayList<>();
if(!wheels.equals(null) & wheels.size()>0){
for(int n : wheels){
if(n%2!=0){
result.add(0);
}
else{
result.add(n/4+1);
}

            }
        }
        return result;
}





static int getMaximumMeetings(List<Integer> start, List<Integer> timeTaken) {
    List<Interval> list = new ArrayList<>(); // create a List of Interval
    for (int i = 0; i < start.size(); i++) {
        list.add(new Interval(start.get(i), start.get(i) + timeTaken.get(i)));
    }
    list.sort(Comparator.comparingInt(i -> i.end)); // sort by finish times ascending

    int res = 0;
    int prevEnd = Integer.MIN_VALUE; // finish time of the previous meeting

    for (Interval i : list) {
        if (i.start >= prevEnd) { // is there a conflict with the previous meeting?
            res++;
            prevEnd = i.end; // update the previous finish time
        }
    }
    return res;
}



public static int GetMinimumDays(int[] parcels)
{
int minDays = 0;
Array.Sort(parcels);

        for(int i = 0; i < parcels.Length; i++)
        {
            int minParcel = parcels[i];
            if (minParcel == 0)
            {
                continue;
            }                
            
            for(int j = i; j < parcels.Length; j++)
            {
                parcels[j] -= minParcel; 
            }

            minDays++;
        }

        return minDays;
    }



    public int maxLengthValidSubArray(int[] processingPower, int[] bootingPower, int maxPower){
        if(processingPower == null || bootingPower == null 
            || processingPower.length == 0 || processingPower.length != bootingPower.length){
                return 0;
        }

        PriorityQueue<Integer> maxBootingPower = new PriorityQueue<>((a, b) -> Integer.compare(b, a));
        int maxLength = 0;
        int currentLength = 1;

        int start = 0;
        int end = 0;
        
        int currentSumProcessingPower = processingPower[0];
        maxBootingPower.add(bootingPower[0]);
        while(end < processingPower.length){
            int currentBootingPower = maxBootingPower.peek();
            int currentPower = currentBootingPower + currentSumProcessingPower * currentLength;

            if(currentPower <= maxPower){
                maxLength = currentLength;
                end++;
                currentLength++;
            }
            else{
                currentSumProcessingPower -= processingPower[start];
                maxBootingPower.remove(bootingPower[start]);
                start++;
                end++;
            }

            if(end < processingPower.length){
                maxBootingPower.add(bootingPower[end]);
                currentSumProcessingPower += processingPower[end];
            }
        }

        return maxLength;
    }


import heapq
def ConnectRope(ropes):
    heapq.heapify(ropes)
    res = 0
    while len(ropes) > 1:
        cur = heapq.heappop(ropes)+heapq.heappop(ropes)
        heapq.heappush(ropes, cur)
        res += cur
    return res



import java.util.ArrayList;
import java.util.Arrays;

public class AmazonsumClosest
{

public static int [] test(int [] nums, int d)
{
	int [] result = new int [2];
	int max = Integer.MIN_VALUE;
	int start = 0;
	int end = nums.length-1;
	Arrays.sort(nums);
	int i = 0;
	int j = 0;
	ArrayList<Integer> a = new ArrayList<Integer>();
	
	while(start <= end)
	{
		if((nums[start] + nums[end])<= d-30)
		{
			
			if(max < (nums[start] + nums[end]))
			{
				max = (nums[start] + nums[end]);
				i = nums[start];
				j = nums[end];
			}
			start++;
		
		}
		else if((nums[start] + nums[end]) > d-30)
				{
					end--;
				}
	}
	result[0] = i;
	result[1] = j;
// a.add(i);
// a.add(j);
// System.out.println(a);
return result;
}

public static int findSubscriberGroups(List<String> arrayList) {
    if (null == arrayList || arrayList.isEmpty()) {
        return 0;
    }
    int count = 0;
    int[][] isConnected = new int[arrayList.size()][arrayList.size()];

    for (int i = 0; i < arrayList.size(); i++) {

        String row = arrayList.get(i);

        for (int j = 0; j < row.length(); j++) {

            // isConnected[i][j] = (row.charAt(j) - '0'); //Working
            isConnected[i][j] = Integer.parseInt(Character.toString(row.charAt(j)));
        }

    }

    boolean[] isReached = new boolean[isConnected.length];
    for (int i = 0; i < isConnected.length; i++) {
        if (!isReached[i]) {
            alignedGroups(isConnected, isReached, i);
            count++;

        }

    }
    return count;

}

private static void alignedGroups(int[][] isConnected, boolean[] isReached, int v) {
    isReached[v] = true;

    for (int i = 0; i < isConnected.length; i++) {
        if (isConnected[v][i] == 1 && !isReached[i])
            alignedGroups(isConnected, isReached, i);

    }

}

	
	
public static String simpleCipher(String encrypted, int k){
        char[] _encrypted = encrypted.toCharArray();
        for(int i=0; i < encrypted.length(); i++){
            char x = _encrypted[i];
            // if the previous kth element is greater than 'A'
            if(x-k>=65){
                _encrypted[i] = (char)(x-k);
            }
            //if ascii code of kth previous element if less than that of A add 26 to it
            else{
                _encrypted[i] = (char)(x-k+26);
            }
        }
        return new String(_encrypted);
    }

	
public List<Integer> numberOfItems(String s, List<Integer> startIndices, List<Integer> endIndices) {
        int n = s.length();
        int[] dp = new int[n];
        int count = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '|') {
                dp[i] = count;
            } else {
                count++;
            }
        }
        List<Integer> ans = new ArrayList<>();
        for (int i = 0; i < startIndices.size(); i++) {
            int start = startIndices.get(i);
            int end = endIndices.get(i);

            while (s.charAt(start) != '|') start++;
            while (s.charAt(end) != '|') end--;
            if (start < end) {
                ans.add(dp[end] - dp[start]);
            } else {
                ans.add(0);
            }
        }
        return ans;
    }

	
    def countBinarySubstrings(self, s: str) -> int:
        prev , curr , res = 0 , 1 , 0
        
        for i in range(1,len(s)):
            if s[i-1] == s[i]:
                curr +=1
            else:
                prev = curr
                curr = 1
            if prev >= curr:
                res+=1
        return res   

			
			
public static int numberOfConnections(List<List<Integer>> gridOfNodes)
    {
        int n = gridOfNodes.size();
        int m = gridOfNodes.get(0).size();

        // array for storing number of 1's in each row
        int rowCounts[] = new int[n];

        int i = 0;
        for(List<Integer> x: gridOfNodes)
        {
            int count = 0;
            for(int y: x)
            {
                if(y == 1)
                    count++;
            }
            rowCounts[i++] = count;
        }

        int prev = -1, answer = 0;

        // finding the first non-zero number in rowCounts
        for(i = 0;i < n;i++)
        {
            if(rowCounts[i] != 0)
            {
                prev = rowCounts[i];
                break;
            }
        }

        i++;
        // multiplying every non-zero rowCounts with previous non-zero rowCounts
        for(;i < n;i++)
        {
            if(rowCounts[i] != 0)
            {
                answer += prev * rowCounts[i];
                prev = rowCounts[i];
            }
        }

        return answer;
    }

	
def  sumSubarrayMin(arr) : 
    stack = []
    arr.append(-1) 
    res=0  
    
    for i in range(len(arr))  :  
        while stack and  arr[i] <  arr[stack[-1]] : 
            idx = stack.pop() 
            res+=  arr[idx] *  (i-idx ) *  (idx -  (stack[-1] if stack else -1 ))
        stack.append(i) 
    return res % (10 **9  +7) 

def  sumSubarrayMax(arr) : 
    stack = []
    arr.append(float('inf')) 
    res=0  
    
    for i in range(len(arr))  :  
        while stack and  arr[i] >  arr[stack[-1]] : 
            idx = stack.pop() 
            res+=  arr[idx] *  (i-idx ) *  (idx -  (stack[-1] if stack else -1 ))
        stack.append(i) 
    return res % (10 **9  + 7 )

minn = sumSubarrayMin([2,4,3,5])
maxx= sumSubarrayMax([2,4,3,5])
print(maxx -minn)

		    
    public int numOfOtions(List<Integer> jeans, List<Integer> top, List<Integer> skirt, List<Integer> shoes, int budget) {
        
        int count = 0;
        Map<Integer,Integer> map = new HashMap<Integer, Integer>();
        
        if(budget == 0) {
            return count;
        }
        for(int i1: jeans) {
            for(int i2: top) {
                int sum = i1+i2;
                map.put(sum,map.getOrDefault(sum,0)+1);
            }
        }
        
        for(int i3: skirt) {
            for(int i4: shoes) {
                int sum = budget-(i3+i4);
                for(Map.Entry<Integer,Integer> entry: map.entrySet()) {
                    if(entry.getKey() <=sum) {
                        count+=entry.getValue();
                    }
                }
            }
        }
        return count;
    }

	
public long findMaxProducts(List<Integer> products) {
        int l = products.size();
        long max = 0;
        for(int i=l-1;i>=0;--i) {
            if(i!=l-1 && products.get(i) < products.get(i+1)) continue;
            long localMax = products.get(i);
            long prev = localMax;
            for(int j=i-1;j>=0;--j) {
                prev = Math.min(prev-1, products.get(j));
                localMax+=prev;
                if(prev==1) break;
            }
            max = Math.max(localMax,max);
        }
        return max;
    }

	
public int minSwapsRequired(String s) {
 int l = 0, r = s.length()-1, swap = 0;
 while(l<r) {
  if(s.charAt(l)!=s.charAt(r)) swap++;
  l++;
  r--;
 }
 if(s.length()%2==0 && swap%2==1) return -1;
 return (swap+1)/2; 
}

	
def appealSum(self, s: str) -> int:
    a=dict()
    for x in 'abcdefghijklmnoppqrstuvwxyz':
        a[x]=-1
    ans=0
    for i,x in enumerate(s):
        ans+=(i-a[x])*(len(s)-i)
        a[x]=i
    return ans

		
public int numberOfSegments(int[] A, int limit) {
        Deque<Integer> maxdq = new LinkedList<>();
        Deque<Integer> mindq = new LinkedList<>();
        int i = 0, j;
        int count = 0;
        for (j = 0; j < A.length; ++j) {
            while (!maxdq.isEmpty() && A[j] > maxdq.peekLast()) maxdq.pollLast();
            while (!mindq.isEmpty() && A[j] < mindq.peekLast()) mindq.pollLast();
            maxdq.add(A[j]);
            mindq.add(A[j]);
            if (maxdq.peek() - mindq.peek() > limit) {
                if (maxdq.peek() == A[i]) maxdq.poll();
                if (mindq.peek() == A[i]) mindq.poll();
                ++i;
            }else{
                count+=(j-i+1);
            }
        }
        return count;
    }


    public static int[] getTimes(int[] time, int[] direction) {
        Queue<Integer> enters = new LinkedList<Integer>();
        Queue<Integer> exits = new LinkedList<Integer>();
        int n = time.length;
        for(int i = 0; i < n; i++) {
            Queue<Integer> q = direction[i] == 1 ? exits : enters;
            q.offer(i);
        }

        int[] result = new int[n];
        int lastTime = -2;
        Queue<Integer> lastQ = exits;
        while(enters.size() > 0 && exits.size() > 0) {
            int currentTime = lastTime + 1;
            int peekEnterTime = time[enters.peek()];
            int peekExitTime = time[exits.peek()];
            Queue<Integer> q;
            if (currentTime < peekEnterTime && currentTime < peekExitTime) {
                // The turnstile was not used
                // Take whoever has the earliest time or
                // if enter == exit, take exit
                q = (peekEnterTime < peekExitTime) ? enters : exits;
                int personIdx = q.poll();
                result[personIdx] = time[personIdx];
                lastTime = time[personIdx]; // time
                lastQ = q;
            } else {
                // Turnstile was used last second
                if (currentTime >= peekEnterTime && currentTime >= peekExitTime) {
                    // Have people waiting at both ends
                    // Prioritize last direction
                    q = lastQ;
                } else {
                    // current >= enters.peek() || current >= exits.peek()
                    q = currentTime >= peekEnterTime ? enters : exits; // take whatever that's queuing
                }
                int personIdx = q.poll();
                result[personIdx] = currentTime;
                lastTime = currentTime; // time
                lastQ = q;
            }
        }

        Queue<Integer> q = enters.size() > 0 ? enters : exits;
        while(q.size() > 0) {
            int currentTime = lastTime + 1;
            int personIdx = q.poll();
            if (currentTime < time[personIdx]) {
                // The turnstile was not used
                currentTime = time[personIdx];
            }

            result[personIdx] = currentTime;
            lastTime = currentTime; // time
        }

        return result;
    }

	
from collections import defaultdict

def findMaxmumGreyness(arr):
	row_1 = defaultdict(int)
	col_1 = defaultdict(int)

	for i in range(len(arr)):
		for j in range(len(arr[i])):
			if arr[i][j] == 1:
				row_1[i] += 1
				col_1[j] += 1
	maxG = float('-inf')
	m = len(arr)
	n = len(arr[0])
	for i in range(len(arr)):
		for j in range(len(arr[i])):
			maxG = max(maxG, row_1[i] + col_1[j] - (n - row_1[i]) - (m - col_1[j]))
	return maxG

				
				
public class Solution {
public int ConnectSticks(int[] sticks) {
if (sticks.Length ==1)
{
return 0;
}
SortedSet<(int,int)> sSet = new SortedSet<(int,int)>();
int i = 0;
foreach(var stick in sticks)
{
sSet.Add((stick,i++));
}
int cost=0;
while(sSet.Count > 1)
{
var temp = sSet.Min.Item1;
sSet.Remove(sSet.Min);
temp+= sSet.Min.Item1;
sSet.Remove(sSet.Min);
sSet.Add((temp,i++));
cost += temp;
}
return cost;
}
}

	
def MaxBoundedArray(n, low, up):
	total = (up - low) * 2 + 1
	if (n > total or up < low or n < 3):
		return None
	# if above check is passed, there definitely exists a solution
	q = collections.deque()
	# create the decreasing part
	for ele in range(up, low-1, -1):
		q.append(ele)
	# case 1: decreasing part has length >= n
	if len(q) >= n:
		while len(q) > n-1:
			q.pop()
		q.appendleft(up-1)
		return list(q)
	# case 2: decreasing part has length < n, so now we can just append left elements
	else:
		for ele in range(up-1, low-1, -1):
			q.appendleft(ele)
			if len(q) == n:
				return list(q)

					
private static void findTime(String s, int[] count) {
    String s1 = s.replaceAll("01", "10");
    if (!s1.equals(s)) {
        int c = count[0];
        count[0] = ++c;
        findTime(s1, count);
    }

}

	
// Find the password strength.
// For each substring of the password which contains at least one vowel and one consonant, its strength goes up by 1.
// vowels={'a', 'e', 'i', 'o', 'u'}, and rest of letters are all consonant.
def password(word):
  vowel=0
  consonent=0
  res=0
  for l in word:
    if l=='a' or l=='e' or l=='i' or l=='o' or l=='u':
      vowel+=1
    else:
      consonent+=1
    if vowel>=1 and consonent>=1:
      res+=1
      vowel=0
      consonent=0
  return res

def countMovesToLeft(nums, target):
	count = 0
	last = 0
	for i in range(len(nums)):
		if nums[i] == target:
			count += i - last
			last += 1
	return count
        
return min(countMovesToLeft(nums, 1), countMovesToLeft(nums, 0))
