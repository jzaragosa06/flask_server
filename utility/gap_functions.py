import pandas as pd
import numpy as np


def compute_count_before(df, freq, interval_length_before_gap):
    count_before = 0
    temp = interval_length_before_gap
    i = len(df) - 1

    while i > 0 and temp > 0:
        diff = df.index[i] - df.index[i - 1]

        if freq == 'D' and diff.days != 1:
              count_before += 1
              break
        elif freq == 'W' and diff.days != 7:
              count_before += 1
              break
        elif freq == 'M' and (df.index[i].month == df.index[i - 1].month or diff.days > 31):
              count_before += 1
              break
        elif freq == 'Q' and (df.index[i].quarter == df.index[i - 1].quarter or diff.days > 93):
              count_before += 1
              break
        elif freq == 'Y' and df.index[i].year == df.index[i - 1].year:
              count_before += 1
              break
        count_before += 1
        temp -= 1
        i -= 1

    return count_before


def identify_gap(df, freq):
    if freq == "D":
        # Ensure the dataframe is set to the desired frequency
        #this part causes the undesireable behaviour. because it fills the gap in the index. 
        # df = df.asfreq("D")

        gap_length = 0
        interval_length = 0

        start_interval = None
        end_interval = None

        counter = 0

        for i in range(1, len(df)):
            diff = df.index[i] - df.index[i-1]
            # print(f"Difference in days between {df.index[i]} and {df.index[i-1]}: {diff.days}")
            
            if diff.days != 1:
                counter += 1
                if counter == 1:
                    gap_length = diff.days - 1  # Adjust for the missed days
                    start_interval = df.index[i]  # Start of the gap
                elif counter == 2:
                  start_interval = df.index[i-1]
                elif counter == 3:
                    end_interval = df.index[i]  # End of the gap
                    break

        # Calculate the interval length
        if start_interval and end_interval:
            interval_length = (end_interval - start_interval).days

        return gap_length, interval_length

    elif freq == "W":
      gap_length = 0
      interval_length = 0

      start_interval = None
      end_interval = None

      counter = 0

      for i in range(1, len(df)):
        diff = df.index[i] - df.index[i-1]
        if diff.days != 7:
          counter += 1
          if counter == 1:
            gap_length = diff.days
          if counter == 2:
            start_interval = df.index[i]
            
          if counter  == 3:
            end_interval = df.index[i]
            break

      #then subtract the end with the start
      interval_length = (end_interval - start_interval).days / 7

      return gap_length, interval_length

def checkGap(df, freq):
    #let assume that there is no gap
    hasGap = False
    
    if freq == "D":
        for i in range(1, len(df)):
            diff = df.index[i] - df.index[i-1]
            
            if diff.days != 1:
                #negate the assumption
                hasGap = True
                break
        
    elif freq == "W":
        for i in range(1, len(df)):
            diff = df.index[i] - df.index[i-1]
            
            if diff.days != 7:
                #negate the assumption
                hasGap = True
                break
    elif freq == "M":
        ...
    
    #return
    return hasGap
    
    