import pandas as pd

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
