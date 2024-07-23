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