from locfdr import locfdr

f = open('sampledata.tsv')
zz = []

for line in f:
    zz.append(float(line.strip()))

results = locfdr(zz)
print 'Executed simulation described in Section 4 of the locfdr() R vignette at http://cran.r-project.org/web/packages/locfdr/vignettes/locfdr-example.pdf . See results in variable \'results\'.'
