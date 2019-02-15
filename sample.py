def readfile (employees):
    with open(employees, 'r') as employees:
        lines = employees.readlines()
    points = []
    for line in lines:
        columns = line.split()
        if len(columns) == 2:
            entry = float(columns[0])
            exit = float (columns[1])
            newpoint = (entry, exit)
            # print(newpoint)
            points.append(newpoint)
        else:
            entry = float(columns[0])
            exit = float (2018)
            newpoint = (entry,exit)
            # print(newpoint)
            points.append(newpoint)
    return points

def calculateaverage (points):
    numemp = float( len(points) )
    # print (numemp, "Employees")
    sum = 0.0
    for point in points:
        entry = point[0]
        exit = point[1]
        diff = exit - entry
        # print (diff)
        sum = sum + diff
    average = sum / numemp
    return average

from os import path

filepath = "./employees.txt"
print(filepath)

# process input file
years = readfile(filepath)


# calculate average
average = calculateaverage(years)

print("Average turnover: ", average)


