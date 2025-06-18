# open the text file
with open('/hdd/Data_object_det/res_net/plot/car_detection.txt') as file:
    # initialize variables for the sums of each column
    sum_col2 = 0
    sum_col3 = 0
    sum_col4 = 0

    # iterate over each line in the file
    for line in file:
        # split the line into a list of values
        values = line.split()

        # add the value in each column to its respective sum variable
        sum_col2 += float(values[1])
        sum_col3 += float(values[2])
        sum_col4 += float(values[3])

    # print the sums of each column
    print("Sum of Column 2: ", sum_col2/41)
    print("Sum of Column 3: ", sum_col3/41)
    print("Sum of Column 4: ", sum_col4/41)
