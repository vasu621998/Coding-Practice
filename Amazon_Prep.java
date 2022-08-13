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
