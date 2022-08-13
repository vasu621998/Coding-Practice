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
