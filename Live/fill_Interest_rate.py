
# adding interest rate to data frame
df.to_csv('file1.csv')
interest = []
with open('ir_rbi.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if not row[1]:
            continue;
        else:
            for i in range(21):
                interest.append(row[1])
df['interest'] = interest
# ----------------------------------