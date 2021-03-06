def counting_sort(array, maxval):
    NofLabel = maxval+1

    d = [0]*NofLabel
    K = len(array)
    dist = array
    for j in range(K):
        d[dist[j]] += 1
    result = max(d.iteritems(), key=lambda x: x[1])
    if result != testLabel[i]:
        fail += 1

    count = [0] * m               # init with zeros
    for a in array:
        count[a] += 1             # count occurences
    i = 0
    for a in range(m):            # emit
        for c in range(count[a]): # - emit 'count[a]' copies of 'a'
            array[i] = a
            i += 1
    return array

print(counting_sort( [1, 4, 7, 2, 1, 3, 2, 1, 4, 2, 3, 2, 1], 7 ))
