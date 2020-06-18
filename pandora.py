from pandas import*
from datetime import*
import matplotlib.pyplot as plt

#This function takes in the inputs:
#1) filename
#2) number of rows to skip
#and returns
#pandas dataframe

def skiprows(filename, rows):
    df=read_csv(filename,skiprows=range(rows),header=None)
    return df

#This function takes in the inputs:
#1) Dataframe with full columnns from csv
#2) List of columns to keep
#and returns
#pandas dataframe

def selectcolumns(df, clist):
    df2=df[clist] #Select just clist from df
    return df2

#This function takes in the inputs:
#1) string (s) in utc
#and returns
#string in akdt
def akdt(s):
    dt=datetime.strptime(s,'%Y%m%dT%H%M%SZ')
    dt2=dt-timedelta(hours=8)
    return dt2.strftime('%Y%m%dT%H%M%SZ')

#This function takes in the inputs:
#1) string (s) in utc
#and returns
#string in akdt
def akdt2(s):
    dt=datetime.strptime(s,'%d-%m-%Y %H:%M')
    dt2=dt-timedelta(hours=8)
    return dt2.strftime('%Y%m%dT%H%M%SZ')


#This function takes in the inputs:
#1) string in YYYYMMDDZHHMMSS
#and returns
#string in DOY.SSSSSS
def doy(s):
    currenttime=datetime.strptime(s, '%Y%m%dT%H%M%SZ')
    newyeartime=datetime.strptime(s[0:4]+'0101T000000Z', '%Y%m%dT%H%M%SZ')
    timedelta=currenttime-newyeartime
    days=timedelta.days
    seconds=timedelta.seconds
    #There are 86400 seconds in a day.
    return days+seconds/86400.0

#This function compresses the dataframe
#by taking hourly averages
#Requires: dataframe with doy column
def hourlyAverage(df,slices=24):
    days=df['doy'].apply(int).unique()
    data=[]
    for day in days:
        for i in range(slices):
            selectrows=df[(df['doy']>day+float(i)/slices)&(df['doy']<day+float(i+1)/slices)]
            row=[]
            try:
                for col in df.columns:
                    if pandas.api.types.is_numeric_dtype(df[col]):
                        row.append(selectrows[col].mean())
                    else:
                        row.append(selectrows[col].iloc[0])
                data.append(row)
            except:
                #print('{} {}'.format(day,len(selectrows)))
                continue
    df2=DataFrame(data,columns=df.columns)
    return df2

def dailyAverage(df):
    days=df['datetime_AKDT'].apply(lambda s:s[0:8]).unique()
    data=[]
    for day in days:
        selectrows=df[df['datetime_AKDT'].apply(lambda s:s[0:8])==day]
        row=[]
        try:
            for col in df.columns:
                if pandas.api.types.is_numeric_dtype(df[col]):
                    row.append(selectrows[col].mean())
                else:
                    row.append(selectrows[col].iloc[0])
            data.append(row)
        except:
            continue
    df2=DataFrame(data,columns=df.columns)
    return df2


def getMonthHelper(akdt):
    return akdt[4:6]

def getMonthSlice(df,monthstr):
    return df[df['datetime_AKDT'].apply(getMonthHelper)==monthstr]

flist= ['20_March_2018.csv', '11_May_2018.csv', '19_June_2018.csv']
masterdf=DataFrame()

for filename in flist:
    r=55
    df=skiprows(filename, r)
    #col3 = Column 4 = SZA
    clist=[0, 3, 7, 8, 9, 20]
    df2=selectcolumns(df, clist)
    masterdf=masterdf.append(df2)
masterdf.index=range(len(masterdf)) #Reindex because we merged dataframes
masterdf['datetime_AKDT']=masterdf[0].apply(akdt)
masterdf['doy']=masterdf['datetime_AKDT'].apply(doy)
masterdf=masterdf[masterdf['datetime_AKDT'].apply(lambda s:int(s[9:11]))>11]
masterdf=masterdf[masterdf['datetime_AKDT'].apply(lambda s:int(s[9:11]))<17]
masterdf.index=range(len(masterdf)) #Reindex because we merged dataframes
#VCDAbs
#masterdf['VCDabs']=masterdf[7]+0.5/masterdf[20]

#masterdf=masterdf[masterdf[3]<85] #Filter SZA to keep SZA < 85
avgmasterdf=hourlyAverage(masterdf)

simpsondf=read_csv('simpson.csv')
#We will have to add a column called AMF in that and assign it a value 2.
simpsondf['AMF']=2.0
simpsondf['VCD']=(simpsondf['dSCD_HCHO_20']/simpsondf['AMF'])
#1 DU=2.69x10^16 mol/cm^-2
simpsondf['VCD']/=(2.69*10**16)

#We will have to add a column called AMF in that and assign it a value 2.
simpsondf['AMF10']=3.4
simpsondf['VCD10']=(simpsondf['dSCD_HCHO_10']/simpsondf['AMF10'])
#1 DU=2.69x10^16 mol/cm^-2
simpsondf['VCD10']/=(2.69*10**16)


simpsondf['datetime_AKDT']=simpsondf['datetime_UTC'].apply(akdt2)
simpsondf['doy']=simpsondf['datetime_AKDT'].apply(doy)
#avgsimpson=hourlyAverage(simpsondf)
avgsimpson=simpsondf.copy() #data already hourly.

#simpsondf['flux360_8_20'].describe()
#Out[693]:
#count     4747.000000
#mean     40290.541843
#std      21885.780395
#min       3748.584229
#25%      19815.667580
#50%      39591.280470
#75%      61265.829295
#max      93610.052080
#Name: flux360_8_20, dtype: float64

forthnights=['20180401','20180415','20180501','20180515','20180601','20180615']
for i in range(1,len(forthnights)):
    slicedf=simpsondf[['datetime_AKDT','flux360_8_20']][(simpsondf['datetime_AKDT'].apply(lambda s:s[:8])>forthnights[i-1])&(simpsondf['datetime_AKDT'].apply(lambda s:s[:8])<forthnights[i])]
    print('From {} to {}'.format(forthnights[i-1],forthnights[i]))
    print(slicedf['flux360_8_20'].describe())

simpsondaily=dailyAverage(simpsondf)
plt.plot(simpsondaily['doy'],simpsondaily['flux360_8_20'])
plt.show()

#Moving average from -7 to +7
simpsondaily['flux360_8_20_MA']=simpsondaily['flux360_8_20'].rolling(15,center=True).mean()
simpsoncloudydays=simpsondaily['datetime_AKDT'][simpsondaily['flux360_8_20']<0.9*simpsondaily['flux360_8_20_MA']].apply(lambda s:s[:8])

#read conditions
conditions=read_csv('conditions.csv').fillna(0)
conditions['doy']=conditions['YYYYMMDDT000000Z'].apply(doy).apply(int)
conditions['Weather']=conditions['Clear']+conditions['Overcast']+conditions['Rainy']
conditions.index=conditions['doy']
conditions['SimpWeather']=0
for day in simpsoncloudydays:
    conditions['SimpWeather'][conditions['YYYYMMDDT000000Z'].apply(lambda s:s[0:8])==day]=1

concordant=0
realsunnydates=[]
for day in conditions.index:
    if (conditions['SimpWeather'][day]==0) and (conditions['Weather'][day]==0):
        concordant+=1
        realsunnydates.append(conditions['YYYYMMDDT000000Z'][day][:8])
    if (conditions['SimpWeather'][day]>0) and (conditions['Weather'][day]>0):
        concordant+=1
print('concordant={} total={} accuracy={:.2f}%'.format(concordant,len(conditions),100.0*concordant/len(conditions)))


masterdf['Weather']=0
for i in range(len(masterdf)):
    thisdoy=int(masterdf['doy'][i])
    try:
        weather=conditions['Weather'][thisdoy]
        masterdf.loc[i,'Weather']=weather
    except:
        masterdf.loc[i,'Weather']=3 #No data

gooddf=masterdf[masterdf[20]==0]
gooddf2=avgmasterdf[avgmasterdf[20]==0]

plt.scatter(gooddf['doy'][gooddf['Weather']==0],gooddf[7][gooddf['Weather']==0],color='r')
plt.scatter(gooddf['doy'][gooddf['Weather']==1],gooddf[7][gooddf['Weather']==1],color='g')
plt.scatter(gooddf['doy'][gooddf['Weather']==2],gooddf[7][gooddf['Weather']==2],color='b')
plt.scatter(gooddf['doy'][gooddf['Weather']==3],gooddf[7][gooddf['Weather']==3],color='y')
plt.show()

simpsondf['year']=simpsondf['datetime_AKDT'].apply(lambda x:x[0:4])
gooddf['year']=gooddf['datetime_AKDT'].apply(lambda x:x[0:4])

for year in ['2017','2018']:
    MAXDOASUAF=plt.scatter(simpsondf['doy'][(simpsondf['year']==year)],simpsondf['VCD'][(simpsondf['year']==year)],color='b')
    if year=='2017':
        plt.legend((MAXDOASUAF,),('MAXDOASUAF',),scatterpoints=1,loc='upper right')
        plt.title('VCD HCHO vs DOY ({})'.format(year))
        plt.ylabel('Dobson Unit (DU)')
        plt.xlabel('Day of Year (DOY)')
        plt.show()
    else:
        PandoraUAF=plt.scatter(gooddf['doy'][(gooddf['year']==year)],gooddf[7][(gooddf['year']==year)],color='r')
        plt.legend((MAXDOASUAF,PandoraUAF),('MAXDOASUAF','PandoraUAF'),scatterpoints=1,loc='upper right')
        plt.title('VCD HCHO vs DOY ({})'.format(year))
        plt.ylabel('Dobson Unit (DU)')
        plt.xlabel('Day of Year (DOY)')
        plt.show()


if True:
        PandoraUAF=plt.scatter(gooddf['doy'],gooddf[7],color='r')
        PandoraUAFAvg=plt.scatter(gooddf2['doy'],gooddf2[7],color='b')
        plt.legend((PandoraUAF,PandoraUAFAvg),('PandoraUAF','UAF1HrAvg'),scatterpoints=1,loc='upper right')
        plt.title('VCD HCHO vs DOY ({})'.format(year))
        plt.ylabel('Dobson Unit (DU)')
        plt.xlabel('Day of Year (DOY)')
        plt.show()

hist,bins=np.histogram(masterdf[7],bins=np.arange(-10.0,10.0,0.1))
center=(bins[:-1]+bins[1:])/2
plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
plt.title('Hist of VCD (master)')
plt.show()

hist,bins=np.histogram(simpsondf['VCD'].dropna(),bins=np.arange(-10.0,10.0,0.1))
center=(bins[:-1]+bins[1:])/2
plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
plt.title('Hist of VCD (simpsondf)')
plt.show()

hist,bins=np.histogram(masterdf[7],bins=np.arange(-2.0,2.0,0.1))
center=(bins[:-1]+bins[1:])/2
plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
plt.title('Hist of VCD (master)')
plt.show()

hist,bins=np.histogram(simpsondf['VCD'].dropna(),bins=np.arange(-10.0,10.0,0.1))
center=(bins[:-1]+bins[1:])/2
plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
plt.title('Hist of VCD (simpsondf)')
plt.show()

for monthstr in ['03', '04', '05', '06', '07']:
    sliceddf=getMonthSlice(masterdf,monthstr)
    hist,bins=np.histogram(sliceddf[7],bins=np.arange(-2.0,2.0,0.1))
    center=(bins[:-1]+bins[1:])/2
    plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
    plt.title('Hist of VCD (master) for month {}'.format(monthstr))
    plt.show()

    sliceddf_simp=getMonthSlice(simpsondf.loc[(simpsondf['year']=='2018')],monthstr)
    hist,bins=np.histogram(sliceddf_simp['VCD'].dropna(),bins=np.arange(-10.0,10.0,0.1))
    center=(bins[:-1]+bins[1:])/2
    plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
    plt.title('Hist of VCD (simpson) for month {}'.format(monthstr))
    plt.show()

    sliceddf_simp=getMonthSlice(simpsondf.loc[(simpsondf['year']=='2018')],monthstr)
    hist,bins=np.histogram(sliceddf_simp['VCD'].dropna(),bins=np.arange(-2.0,2.0,0.1))
    center=(bins[:-1]+bins[1:])/2
    plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
    plt.title('Hist of VCD (simpson) for month {}'.format(monthstr))
    plt.show()

for monthstr in ['03', '04', '05', '06', '07']:
    sliceddf=hourlyAverage(getMonthSlice(masterdf,monthstr))
    hist,bins=np.histogram(sliceddf[7],bins=np.arange(-2.0,2.0,0.1))
    center=(bins[:-1]+bins[1:])/2
    plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
    plt.title('Hist of VCD (master) for month {}'.format(monthstr))
    plt.show()

    sliceddf_simp=hourlyAverage(getMonthSlice(simpsondf.loc[(simpsondf['year']=='2018')],monthstr))
    hist,bins=np.histogram(sliceddf_simp['VCD'].dropna(),bins=np.arange(-10.0,10.0,0.1))
    center=(bins[:-1]+bins[1:])/2
    plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
    plt.title('Hist of VCD (simpson) for month {}'.format(monthstr))
    plt.show()

    sliceddf_simp=hourlyAverage(getMonthSlice(simpsondf.loc[(simpsondf['year']=='2018')],monthstr))
    hist,bins=np.histogram(sliceddf_simp['VCD'].dropna(),bins=np.arange(-2.0,2.0,0.1))
    center=(bins[:-1]+bins[1:])/2
    plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
    plt.title('Hist of VCD (simpson) for month {}'.format(monthstr))
    plt.show()

print(masterdf.describe())
print(simpsondf.describe())
print('VCD on real sunny dates')
masterdf['date']=masterdf['datetime_AKDT'].apply(lambda s:s[:8])
masterdf.index=masterdf['date']
sunnydf=masterdf.loc[realsunnydates]
sunnydf.index=range(len(sunnydf.index))
masterdf.index=range(len(masterdf.index))
sunnydf=sunnydf.dropna()
sunnydf=sunnydf[sunnydf['datetime_AKDT'].apply(lambda s:int(s[9:11]))>13]
sunnydf=sunnydf[sunnydf['datetime_AKDT'].apply(lambda s:int(s[9:11]))<15]


simpsondf['date']=simpsondf['datetime_AKDT'].apply(lambda s:s[:8])
simpsondf.index=simpsondf['date']
sunnysimpsondf=simpsondf.loc[realsunnydates]
sunnysimpsondf.index=range(len(sunnysimpsondf.index))
simpsondf.index=range(len(simpsondf.index))


PandoraUAF=plt.scatter(sunnydf['doy'],sunnydf[7],color='r')
z = np.polyfit(sunnydf['doy'],sunnydf[7], 1)
p = np.poly1d(z)
x = sunnysimpsondf['doy'][(sunnysimpsondf['year']=='2018')]
plt.plot(x,p(x),"r--")
MAXDOASUAF=plt.scatter(sunnysimpsondf['doy'][(sunnysimpsondf['year']=='2018')],sunnysimpsondf['VCD'][(sunnysimpsondf['year']=='2018')],color='b')
z = np.polyfit(sunnysimpsondf.dropna()['doy'][(sunnysimpsondf['year']=='2018')],sunnysimpsondf.dropna()['VCD'][(sunnysimpsondf['year']=='2018')], 1)
p = np.poly1d(z)
plt.plot(x,p(x),"b--")

plt.legend((MAXDOASUAF,PandoraUAF),('MAXDOASUAF','PandoraUAF'),scatterpoints=1,loc='best')
plt.title('VCD HCHO (Sunny only) vs DOY ({})'.format(year))
plt.ylabel('Dobson Unit (DU)')
plt.xlabel('Day of Year (DOY)')
plt.show()

mao=skiprows('mao_hcho.csv', 57)
#High quality data - I want to plot the VCDs of those rows only.
mao=mao[mao[12]==0]
mao['datetime_AKDT']=mao[0].apply(akdt)
mao['doy']=mao['datetime_AKDT'].apply(doy)
maohr=hourlyAverage(mao)
maohr['date']=maohr['datetime_AKDT'].apply(lambda s:s[:8])
avgsimpson['date']=avgsimpson['datetime_AKDT'].apply(lambda s:s[:8])


maohr.index=maohr['date']
sunnymao=maohr.loc[realsunnydates]
sunnymao=sunnymao[notnull(sunnymao['datetime_AKDT'])]
sunnymao.index=range(len(sunnymao.index))
maohr.index=range(len(maohr.index))

avgsimpson.index=avgsimpson['date']
sunnysimpson=avgsimpson.loc[realsunnydates]
sunnysimpson=sunnysimpson[notnull(sunnysimpson['datetime_AKDT'])]
sunnysimpson.index=range(len(sunnysimpson.index))
avgsimpson.index=range(len(avgsimpson.index))

avgs=[]
times=[]
avgs2=[]
times2=[]
for i in range(24):
    sunnymao2=sunnymao[sunnymao['datetime_AKDT'].apply(lambda s:int(s[9:11]))==i]
    if len(sunnymao2):
        avgs.append(sunnymao2[7].mean())
        times.append(i)
    sunnysimpson2=sunnysimpson[sunnysimpson['datetime_AKDT'].apply(lambda s:int(s[9:11]))==i]
    if len(sunnysimpson2):
        avgs2.append(sunnysimpson2['VCD'].mean())
        times2.append(i)
sp1=plt.scatter(times,avgs,color='b')
sp2=plt.scatter(times2,avgs2,color='r')
plt.legend((sp1,sp2),('Mao','Simpson'),scatterpoints=1,loc='best')
plt.title('VCD HCHO (Sunny only) vs Time of day')
plt.ylabel('VCD')
plt.xlabel('Time of day')
plt.show()


for yyyymm in ['201805','201806','201807']:
    realsunnydates2=[d for d in realsunnydates if d[0:6]==yyyymm]
    if (len(realsunnydates2)>8):
        todrop=(len(realsunnydates2)-7)//2
        realsunnydates2=realsunnydates2[todrop:-todrop]
    maohr.index=maohr['date']
    sunnymao=maohr.loc[realsunnydates2]
    sunnymao=sunnymao[notnull(sunnymao['datetime_AKDT'])]
    sunnymao.index=range(len(sunnymao.index))
    maohr.index=range(len(maohr.index))

    avgsimpson.index=avgsimpson['date']
    sunnysimpson=avgsimpson.loc[realsunnydates2]
    sunnysimpson=sunnysimpson[notnull(sunnysimpson['datetime_AKDT'])]
    sunnysimpson.index=range(len(sunnysimpson.index))
    avgsimpson.index=range(len(avgsimpson.index))


    avgs=[]
    times=[]
    avgs2=[]
    times2=[]
    for i in range(24):
        sunnymao2=sunnymao[sunnymao['datetime_AKDT'].apply(lambda s:int(s[9:11]))==i]
        if len(sunnymao2):
            avgs.append(sunnymao2[7].mean())
            times.append(i)
        sunnysimpson2=sunnysimpson[sunnysimpson['datetime_AKDT'].apply(lambda s:int(s[9:11]))==i]
        if len(sunnysimpson2):
            avgs2.append(sunnysimpson2['VCD'].mean())
            times2.append(i)
    sp1=plt.scatter(times,avgs,color='b')
    sp2=plt.scatter(times2,avgs2,color='r')
    plt.legend((sp1,sp2),('Mao','Simpson'),scatterpoints=1,loc='best')
    plt.title('VCD HCHO (Sunny only for {}) vs Time of day'.format(yyyymm))
    plt.ylabel('VCD')
    plt.xlabel('Time of day')
    plt.show()from pandas import*
from datetime import*
import matplotlib.pyplot as plt

#This function takes in the inputs:
#1) filename
#2) number of rows to skip
#and returns
#pandas dataframe

def skiprows(filename, rows):
    df=read_csv(filename,skiprows=range(rows),header=None)
    return df

#This function takes in the inputs:
#1) Dataframe with full columnns from csv
#2) List of columns to keep
#and returns
#pandas dataframe

def selectcolumns(df, clist):
    df2=df[clist] #Select just clist from df
    return df2

#This function takes in the inputs:
#1) string (s) in utc
#and returns
#string in akdt
def akdt(s):
    dt=datetime.strptime(s,'%Y%m%dT%H%M%SZ')
    dt2=dt-timedelta(hours=8)
    return dt2.strftime('%Y%m%dT%H%M%SZ')

#This function takes in the inputs:
#1) string (s) in utc
#and returns
#string in akdt
def akdt2(s):
    dt=datetime.strptime(s,'%d-%m-%Y %H:%M')
    dt2=dt-timedelta(hours=8)
    return dt2.strftime('%Y%m%dT%H%M%SZ')


#This function takes in the inputs:
#1) string in YYYYMMDDZHHMMSS
#and returns
#string in DOY.SSSSSS
def doy(s):
    currenttime=datetime.strptime(s, '%Y%m%dT%H%M%SZ')
    newyeartime=datetime.strptime(s[0:4]+'0101T000000Z', '%Y%m%dT%H%M%SZ')
    timedelta=currenttime-newyeartime
    days=timedelta.days
    seconds=timedelta.seconds
    #There are 86400 seconds in a day.
    return days+seconds/86400.0

#This function compresses the dataframe
#by taking hourly averages
#Requires: dataframe with doy column
def hourlyAverage(df,slices=24):
    days=df['doy'].apply(int).unique()
    data=[]
    for day in days:
        for i in range(slices):
            selectrows=df[(df['doy']>day+float(i)/slices)&(df['doy']<day+float(i+1)/slices)]
            row=[]
            try:
                for col in df.columns:
                    if pandas.api.types.is_numeric_dtype(df[col]):
                        row.append(selectrows[col].mean())
                    else:
                        row.append(selectrows[col].iloc[0])
                data.append(row)
            except:
                #print('{} {}'.format(day,len(selectrows)))
                continue
    df2=DataFrame(data,columns=df.columns)
    return df2

def dailyAverage(df):
    days=df['datetime_AKDT'].apply(lambda s:s[0:8]).unique()
    data=[]
    for day in days:
        selectrows=df[df['datetime_AKDT'].apply(lambda s:s[0:8])==day]
        row=[]
        try:
            for col in df.columns:
                if pandas.api.types.is_numeric_dtype(df[col]):
                    row.append(selectrows[col].mean())
                else:
                    row.append(selectrows[col].iloc[0])
            data.append(row)
        except:
            continue
    df2=DataFrame(data,columns=df.columns)
    return df2


def getMonthHelper(akdt):
    return akdt[4:6]

def getMonthSlice(df,monthstr):
    return df[df['datetime_AKDT'].apply(getMonthHelper)==monthstr]

flist= ['20_March_2018.csv', '11_May_2018.csv', '19_June_2018.csv']
masterdf=DataFrame()

for filename in flist:
    r=55
    df=skiprows(filename, r)
    #col3 = Column 4 = SZA
    clist=[0, 3, 7, 8, 9, 20]
    df2=selectcolumns(df, clist)
    masterdf=masterdf.append(df2)
masterdf.index=range(len(masterdf)) #Reindex because we merged dataframes
masterdf['datetime_AKDT']=masterdf[0].apply(akdt)
masterdf['doy']=masterdf['datetime_AKDT'].apply(doy)
masterdf=masterdf[masterdf['datetime_AKDT'].apply(lambda s:int(s[9:11]))>11]
masterdf=masterdf[masterdf['datetime_AKDT'].apply(lambda s:int(s[9:11]))<17]
masterdf.index=range(len(masterdf)) #Reindex because we merged dataframes
#VCDAbs
#masterdf['VCDabs']=masterdf[7]+0.5/masterdf[20]

#masterdf=masterdf[masterdf[3]<85] #Filter SZA to keep SZA < 85
avgmasterdf=hourlyAverage(masterdf)

simpsondf=read_csv('simpson.csv')
#We will have to add a column called AMF in that and assign it a value 2.
simpsondf['AMF']=2.0
simpsondf['VCD']=(simpsondf['dSCD_HCHO_20']/simpsondf['AMF'])
#1 DU=2.69x10^16 mol/cm^-2
simpsondf['VCD']/=(2.69*10**16)

#We will have to add a column called AMF in that and assign it a value 2.
simpsondf['AMF10']=3.4
simpsondf['VCD10']=(simpsondf['dSCD_HCHO_10']/simpsondf['AMF10'])
#1 DU=2.69x10^16 mol/cm^-2
simpsondf['VCD10']/=(2.69*10**16)


simpsondf['datetime_AKDT']=simpsondf['datetime_UTC'].apply(akdt2)
simpsondf['doy']=simpsondf['datetime_AKDT'].apply(doy)
#avgsimpson=hourlyAverage(simpsondf)
avgsimpson=simpsondf.copy() #data already hourly.

#simpsondf['flux360_8_20'].describe()
#Out[693]:
#count     4747.000000
#mean     40290.541843
#std      21885.780395
#min       3748.584229
#25%      19815.667580
#50%      39591.280470
#75%      61265.829295
#max      93610.052080
#Name: flux360_8_20, dtype: float64

forthnights=['20180401','20180415','20180501','20180515','20180601','20180615']
for i in range(1,len(forthnights)):
    slicedf=simpsondf[['datetime_AKDT','flux360_8_20']][(simpsondf['datetime_AKDT'].apply(lambda s:s[:8])>forthnights[i-1])&(simpsondf['datetime_AKDT'].apply(lambda s:s[:8])<forthnights[i])]
    print('From {} to {}'.format(forthnights[i-1],forthnights[i]))
    print(slicedf['flux360_8_20'].describe())

simpsondaily=dailyAverage(simpsondf)
plt.plot(simpsondaily['doy'],simpsondaily['flux360_8_20'])
plt.show()

#Moving average from -7 to +7
simpsondaily['flux360_8_20_MA']=simpsondaily['flux360_8_20'].rolling(15,center=True).mean()
simpsoncloudydays=simpsondaily['datetime_AKDT'][simpsondaily['flux360_8_20']<0.9*simpsondaily['flux360_8_20_MA']].apply(lambda s:s[:8])

#read conditions
conditions=read_csv('conditions.csv').fillna(0)
conditions['doy']=conditions['YYYYMMDDT000000Z'].apply(doy).apply(int)
conditions['Weather']=conditions['Clear']+conditions['Overcast']+conditions['Rainy']
conditions.index=conditions['doy']
conditions['SimpWeather']=0
for day in simpsoncloudydays:
    conditions['SimpWeather'][conditions['YYYYMMDDT000000Z'].apply(lambda s:s[0:8])==day]=1

concordant=0
realsunnydates=[]
for day in conditions.index:
    if (conditions['SimpWeather'][day]==0) and (conditions['Weather'][day]==0):
        concordant+=1
        realsunnydates.append(conditions['YYYYMMDDT000000Z'][day][:8])
    if (conditions['SimpWeather'][day]>0) and (conditions['Weather'][day]>0):
        concordant+=1
print('concordant={} total={} accuracy={:.2f}%'.format(concordant,len(conditions),100.0*concordant/len(conditions)))


masterdf['Weather']=0
for i in range(len(masterdf)):
    thisdoy=int(masterdf['doy'][i])
    try:
        weather=conditions['Weather'][thisdoy]
        masterdf.loc[i,'Weather']=weather
    except:
        masterdf.loc[i,'Weather']=3 #No data

gooddf=masterdf[masterdf[20]==0]
gooddf2=avgmasterdf[avgmasterdf[20]==0]

plt.scatter(gooddf['doy'][gooddf['Weather']==0],gooddf[7][gooddf['Weather']==0],color='r')
plt.scatter(gooddf['doy'][gooddf['Weather']==1],gooddf[7][gooddf['Weather']==1],color='g')
plt.scatter(gooddf['doy'][gooddf['Weather']==2],gooddf[7][gooddf['Weather']==2],color='b')
plt.scatter(gooddf['doy'][gooddf['Weather']==3],gooddf[7][gooddf['Weather']==3],color='y')
plt.show()

simpsondf['year']=simpsondf['datetime_AKDT'].apply(lambda x:x[0:4])
gooddf['year']=gooddf['datetime_AKDT'].apply(lambda x:x[0:4])

for year in ['2017','2018']:
    MAXDOASUAF=plt.scatter(simpsondf['doy'][(simpsondf['year']==year)],simpsondf['VCD'][(simpsondf['year']==year)],color='b')
    if year=='2017':
        plt.legend((MAXDOASUAF,),('MAXDOASUAF',),scatterpoints=1,loc='upper right')
        plt.title('VCD HCHO vs DOY ({})'.format(year))
        plt.ylabel('Dobson Unit (DU)')
        plt.xlabel('Day of Year (DOY)')
        plt.show()
    else:
        PandoraUAF=plt.scatter(gooddf['doy'][(gooddf['year']==year)],gooddf[7][(gooddf['year']==year)],color='r')
        plt.legend((MAXDOASUAF,PandoraUAF),('MAXDOASUAF','PandoraUAF'),scatterpoints=1,loc='upper right')
        plt.title('VCD HCHO vs DOY ({})'.format(year))
        plt.ylabel('Dobson Unit (DU)')
        plt.xlabel('Day of Year (DOY)')
        plt.show()


if True:
        PandoraUAF=plt.scatter(gooddf['doy'],gooddf[7],color='r')
        PandoraUAFAvg=plt.scatter(gooddf2['doy'],gooddf2[7],color='b')
        plt.legend((PandoraUAF,PandoraUAFAvg),('PandoraUAF','UAF1HrAvg'),scatterpoints=1,loc='upper right')
        plt.title('VCD HCHO vs DOY ({})'.format(year))
        plt.ylabel('Dobson Unit (DU)')
        plt.xlabel('Day of Year (DOY)')
        plt.show()

hist,bins=np.histogram(masterdf[7],bins=np.arange(-10.0,10.0,0.1))
center=(bins[:-1]+bins[1:])/2
plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
plt.title('Hist of VCD (master)')
plt.show()

hist,bins=np.histogram(simpsondf['VCD'].dropna(),bins=np.arange(-10.0,10.0,0.1))
center=(bins[:-1]+bins[1:])/2
plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
plt.title('Hist of VCD (simpsondf)')
plt.show()

hist,bins=np.histogram(masterdf[7],bins=np.arange(-2.0,2.0,0.1))
center=(bins[:-1]+bins[1:])/2
plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
plt.title('Hist of VCD (master)')
plt.show()

hist,bins=np.histogram(simpsondf['VCD'].dropna(),bins=np.arange(-10.0,10.0,0.1))
center=(bins[:-1]+bins[1:])/2
plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
plt.title('Hist of VCD (simpsondf)')
plt.show()

for monthstr in ['03', '04', '05', '06', '07']:
    sliceddf=getMonthSlice(masterdf,monthstr)
    hist,bins=np.histogram(sliceddf[7],bins=np.arange(-2.0,2.0,0.1))
    center=(bins[:-1]+bins[1:])/2
    plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
    plt.title('Hist of VCD (master) for month {}'.format(monthstr))
    plt.show()

    sliceddf_simp=getMonthSlice(simpsondf.loc[(simpsondf['year']=='2018')],monthstr)
    hist,bins=np.histogram(sliceddf_simp['VCD'].dropna(),bins=np.arange(-10.0,10.0,0.1))
    center=(bins[:-1]+bins[1:])/2
    plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
    plt.title('Hist of VCD (simpson) for month {}'.format(monthstr))
    plt.show()

    sliceddf_simp=getMonthSlice(simpsondf.loc[(simpsondf['year']=='2018')],monthstr)
    hist,bins=np.histogram(sliceddf_simp['VCD'].dropna(),bins=np.arange(-2.0,2.0,0.1))
    center=(bins[:-1]+bins[1:])/2
    plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
    plt.title('Hist of VCD (simpson) for month {}'.format(monthstr))
    plt.show()

for monthstr in ['03', '04', '05', '06', '07']:
    sliceddf=hourlyAverage(getMonthSlice(masterdf,monthstr))
    hist,bins=np.histogram(sliceddf[7],bins=np.arange(-2.0,2.0,0.1))
    center=(bins[:-1]+bins[1:])/2
    plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
    plt.title('Hist of VCD (master) for month {}'.format(monthstr))
    plt.show()

    sliceddf_simp=hourlyAverage(getMonthSlice(simpsondf.loc[(simpsondf['year']=='2018')],monthstr))
    hist,bins=np.histogram(sliceddf_simp['VCD'].dropna(),bins=np.arange(-10.0,10.0,0.1))
    center=(bins[:-1]+bins[1:])/2
    plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
    plt.title('Hist of VCD (simpson) for month {}'.format(monthstr))
    plt.show()

    sliceddf_simp=hourlyAverage(getMonthSlice(simpsondf.loc[(simpsondf['year']=='2018')],monthstr))
    hist,bins=np.histogram(sliceddf_simp['VCD'].dropna(),bins=np.arange(-2.0,2.0,0.1))
    center=(bins[:-1]+bins[1:])/2
    plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
    plt.title('Hist of VCD (simpson) for month {}'.format(monthstr))
    plt.show()

print(masterdf.describe())
print(simpsondf.describe())
print('VCD on real sunny dates')
masterdf['date']=masterdf['datetime_AKDT'].apply(lambda s:s[:8])
masterdf.index=masterdf['date']
sunnydf=masterdf.loc[realsunnydates]
sunnydf.index=range(len(sunnydf.index))
masterdf.index=range(len(masterdf.index))
sunnydf=sunnydf.dropna()
sunnydf=sunnydf[sunnydf['datetime_AKDT'].apply(lambda s:int(s[9:11]))>13]
sunnydf=sunnydf[sunnydf['datetime_AKDT'].apply(lambda s:int(s[9:11]))<15]


simpsondf['date']=simpsondf['datetime_AKDT'].apply(lambda s:s[:8])
simpsondf.index=simpsondf['date']
sunnysimpsondf=simpsondf.loc[realsunnydates]
sunnysimpsondf.index=range(len(sunnysimpsondf.index))
simpsondf.index=range(len(simpsondf.index))


PandoraUAF=plt.scatter(sunnydf['doy'],sunnydf[7],color='r')
z = np.polyfit(sunnydf['doy'],sunnydf[7], 1)
p = np.poly1d(z)
x = sunnysimpsondf['doy'][(sunnysimpsondf['year']=='2018')]
plt.plot(x,p(x),"r--")
MAXDOASUAF=plt.scatter(sunnysimpsondf['doy'][(sunnysimpsondf['year']=='2018')],sunnysimpsondf['VCD'][(sunnysimpsondf['year']=='2018')],color='b')
z = np.polyfit(sunnysimpsondf.dropna()['doy'][(sunnysimpsondf['year']=='2018')],sunnysimpsondf.dropna()['VCD'][(sunnysimpsondf['year']=='2018')], 1)
p = np.poly1d(z)
plt.plot(x,p(x),"b--")

plt.legend((MAXDOASUAF,PandoraUAF),('MAXDOASUAF','PandoraUAF'),scatterpoints=1,loc='best')
plt.title('VCD HCHO (Sunny only) vs DOY ({})'.format(year))
plt.ylabel('Dobson Unit (DU)')
plt.xlabel('Day of Year (DOY)')
plt.show()

mao=skiprows('mao_hcho.csv', 57)
#High quality data - I want to plot the VCDs of those rows only.
mao=mao[mao[12]==0]
mao['datetime_AKDT']=mao[0].apply(akdt)
mao['doy']=mao['datetime_AKDT'].apply(doy)
maohr=hourlyAverage(mao)
maohr['date']=maohr['datetime_AKDT'].apply(lambda s:s[:8])
avgsimpson['date']=avgsimpson['datetime_AKDT'].apply(lambda s:s[:8])


maohr.index=maohr['date']
sunnymao=maohr.loc[realsunnydates]
sunnymao=sunnymao[notnull(sunnymao['datetime_AKDT'])]
sunnymao.index=range(len(sunnymao.index))
maohr.index=range(len(maohr.index))

avgsimpson.index=avgsimpson['date']
sunnysimpson=avgsimpson.loc[realsunnydates]
sunnysimpson=sunnysimpson[notnull(sunnysimpson['datetime_AKDT'])]
sunnysimpson.index=range(len(sunnysimpson.index))
avgsimpson.index=range(len(avgsimpson.index))

avgs=[]
times=[]
avgs2=[]
times2=[]
for i in range(24):
    sunnymao2=sunnymao[sunnymao['datetime_AKDT'].apply(lambda s:int(s[9:11]))==i]
    if len(sunnymao2):
        avgs.append(sunnymao2[7].mean())
        times.append(i)
    sunnysimpson2=sunnysimpson[sunnysimpson['datetime_AKDT'].apply(lambda s:int(s[9:11]))==i]
    if len(sunnysimpson2):
        avgs2.append(sunnysimpson2['VCD'].mean())
        times2.append(i)
sp1=plt.scatter(times,avgs,color='b')
sp2=plt.scatter(times2,avgs2,color='r')
plt.legend((sp1,sp2),('Mao','Simpson'),scatterpoints=1,loc='best')
plt.title('VCD HCHO (Sunny only) vs Time of day')
plt.ylabel('VCD')
plt.xlabel('Time of day')
plt.show()


for yyyymm in ['201805','201806','201807']:
    realsunnydates2=[d for d in realsunnydates if d[0:6]==yyyymm]
    if (len(realsunnydates2)>8):
        todrop=(len(realsunnydates2)-7)//2
        realsunnydates2=realsunnydates2[todrop:-todrop]
    maohr.index=maohr['date']
    sunnymao=maohr.loc[realsunnydates2]
    sunnymao=sunnymao[notnull(sunnymao['datetime_AKDT'])]
    sunnymao.index=range(len(sunnymao.index))
    maohr.index=range(len(maohr.index))

    avgsimpson.index=avgsimpson['date']
    sunnysimpson=avgsimpson.loc[realsunnydates2]
    sunnysimpson=sunnysimpson[notnull(sunnysimpson['datetime_AKDT'])]
    sunnysimpson.index=range(len(sunnysimpson.index))
    avgsimpson.index=range(len(avgsimpson.index))


    avgs=[]
    times=[]
    avgs2=[]
    times2=[]
    for i in range(24):
        sunnymao2=sunnymao[sunnymao['datetime_AKDT'].apply(lambda s:int(s[9:11]))==i]
        if len(sunnymao2):
            avgs.append(sunnymao2[7].mean())
            times.append(i)
        sunnysimpson2=sunnysimpson[sunnysimpson['datetime_AKDT'].apply(lambda s:int(s[9:11]))==i]
        if len(sunnysimpson2):
            avgs2.append(sunnysimpson2['VCD'].mean())
            times2.append(i)
    sp1=plt.scatter(times,avgs,color='b')
    sp2=plt.scatter(times2,avgs2,color='r')
    plt.legend((sp1,sp2),('Mao','Simpson'),scatterpoints=1,loc='best')
    plt.title('VCD HCHO (Sunny only for {}) vs Time of day'.format(yyyymm))
    plt.ylabel('VCD')
    plt.xlabel('Time of day')
    plt.show()from pandas import*
from datetime import*
import matplotlib.pyplot as plt

#This function takes in the inputs:
#1) filename
#2) number of rows to skip
#and returns
#pandas dataframe

def skiprows(filename, rows):
    df=read_csv(filename,skiprows=range(rows),header=None)
    return df

#This function takes in the inputs:
#1) Dataframe with full columnns from csv
#2) List of columns to keep
#and returns
#pandas dataframe

def selectcolumns(df, clist):
    df2=df[clist] #Select just clist from df
    return df2

#This function takes in the inputs:
#1) string (s) in utc
#and returns
#string in akdt
def akdt(s):
    dt=datetime.strptime(s,'%Y%m%dT%H%M%SZ')
    dt2=dt-timedelta(hours=8)
    return dt2.strftime('%Y%m%dT%H%M%SZ')

#This function takes in the inputs:
#1) string (s) in utc
#and returns
#string in akdt
def akdt2(s):
    dt=datetime.strptime(s,'%d-%m-%Y %H:%M')
    dt2=dt-timedelta(hours=8)
    return dt2.strftime('%Y%m%dT%H%M%SZ')


#This function takes in the inputs:
#1) string in YYYYMMDDZHHMMSS
#and returns
#string in DOY.SSSSSS
def doy(s):
    currenttime=datetime.strptime(s, '%Y%m%dT%H%M%SZ')
    newyeartime=datetime.strptime(s[0:4]+'0101T000000Z', '%Y%m%dT%H%M%SZ')
    timedelta=currenttime-newyeartime
    days=timedelta.days
    seconds=timedelta.seconds
    #There are 86400 seconds in a day.
    return days+seconds/86400.0

#This function compresses the dataframe
#by taking hourly averages
#Requires: dataframe with doy column
def hourlyAverage(df,slices=24):
    days=df['doy'].apply(int).unique()
    data=[]
    for day in days:
        for i in range(slices):
            selectrows=df[(df['doy']>day+float(i)/slices)&(df['doy']<day+float(i+1)/slices)]
            row=[]
            try:
                for col in df.columns:
                    if pandas.api.types.is_numeric_dtype(df[col]):
                        row.append(selectrows[col].mean())
                    else:
                        row.append(selectrows[col].iloc[0])
                data.append(row)
            except:
                #print('{} {}'.format(day,len(selectrows)))
                continue
    df2=DataFrame(data,columns=df.columns)
    return df2

def dailyAverage(df):
    days=df['datetime_AKDT'].apply(lambda s:s[0:8]).unique()
    data=[]
    for day in days:
        selectrows=df[df['datetime_AKDT'].apply(lambda s:s[0:8])==day]
        row=[]
        try:
            for col in df.columns:
                if pandas.api.types.is_numeric_dtype(df[col]):
                    row.append(selectrows[col].mean())
                else:
                    row.append(selectrows[col].iloc[0])
            data.append(row)
        except:
            continue
    df2=DataFrame(data,columns=df.columns)
    return df2


def getMonthHelper(akdt):
    return akdt[4:6]

def getMonthSlice(df,monthstr):
    return df[df['datetime_AKDT'].apply(getMonthHelper)==monthstr]

flist= ['20_March_2018.csv', '11_May_2018.csv', '19_June_2018.csv']
masterdf=DataFrame()

for filename in flist:
    r=55
    df=skiprows(filename, r)
    #col3 = Column 4 = SZA
    clist=[0, 3, 7, 8, 9, 20]
    df2=selectcolumns(df, clist)
    masterdf=masterdf.append(df2)
masterdf.index=range(len(masterdf)) #Reindex because we merged dataframes
masterdf['datetime_AKDT']=masterdf[0].apply(akdt)
masterdf['doy']=masterdf['datetime_AKDT'].apply(doy)
masterdf=masterdf[masterdf['datetime_AKDT'].apply(lambda s:int(s[9:11]))>11]
masterdf=masterdf[masterdf['datetime_AKDT'].apply(lambda s:int(s[9:11]))<17]
masterdf.index=range(len(masterdf)) #Reindex because we merged dataframes
#VCDAbs
#masterdf['VCDabs']=masterdf[7]+0.5/masterdf[20]

#masterdf=masterdf[masterdf[3]<85] #Filter SZA to keep SZA < 85
avgmasterdf=hourlyAverage(masterdf)

simpsondf=read_csv('simpson.csv')
#We will have to add a column called AMF in that and assign it a value 2.
simpsondf['AMF']=2.0
simpsondf['VCD']=(simpsondf['dSCD_HCHO_20']/simpsondf['AMF'])
#1 DU=2.69x10^16 mol/cm^-2
simpsondf['VCD']/=(2.69*10**16)

#We will have to add a column called AMF in that and assign it a value 2.
simpsondf['AMF10']=3.4
simpsondf['VCD10']=(simpsondf['dSCD_HCHO_10']/simpsondf['AMF10'])
#1 DU=2.69x10^16 mol/cm^-2
simpsondf['VCD10']/=(2.69*10**16)


simpsondf['datetime_AKDT']=simpsondf['datetime_UTC'].apply(akdt2)
simpsondf['doy']=simpsondf['datetime_AKDT'].apply(doy)
#avgsimpson=hourlyAverage(simpsondf)
avgsimpson=simpsondf.copy() #data already hourly.

#simpsondf['flux360_8_20'].describe()
#Out[693]:
#count     4747.000000
#mean     40290.541843
#std      21885.780395
#min       3748.584229
#25%      19815.667580
#50%      39591.280470
#75%      61265.829295
#max      93610.052080
#Name: flux360_8_20, dtype: float64

forthnights=['20180401','20180415','20180501','20180515','20180601','20180615']
for i in range(1,len(forthnights)):
    slicedf=simpsondf[['datetime_AKDT','flux360_8_20']][(simpsondf['datetime_AKDT'].apply(lambda s:s[:8])>forthnights[i-1])&(simpsondf['datetime_AKDT'].apply(lambda s:s[:8])<forthnights[i])]
    print('From {} to {}'.format(forthnights[i-1],forthnights[i]))
    print(slicedf['flux360_8_20'].describe())

simpsondaily=dailyAverage(simpsondf)
plt.plot(simpsondaily['doy'],simpsondaily['flux360_8_20'])
plt.show()

#Moving average from -7 to +7
simpsondaily['flux360_8_20_MA']=simpsondaily['flux360_8_20'].rolling(15,center=True).mean()
simpsoncloudydays=simpsondaily['datetime_AKDT'][simpsondaily['flux360_8_20']<0.9*simpsondaily['flux360_8_20_MA']].apply(lambda s:s[:8])

#read conditions
conditions=read_csv('conditions.csv').fillna(0)
conditions['doy']=conditions['YYYYMMDDT000000Z'].apply(doy).apply(int)
conditions['Weather']=conditions['Clear']+conditions['Overcast']+conditions['Rainy']
conditions.index=conditions['doy']
conditions['SimpWeather']=0
for day in simpsoncloudydays:
    conditions['SimpWeather'][conditions['YYYYMMDDT000000Z'].apply(lambda s:s[0:8])==day]=1

concordant=0
realsunnydates=[]
for day in conditions.index:
    if (conditions['SimpWeather'][day]==0) and (conditions['Weather'][day]==0):
        concordant+=1
        realsunnydates.append(conditions['YYYYMMDDT000000Z'][day][:8])
    if (conditions['SimpWeather'][day]>0) and (conditions['Weather'][day]>0):
        concordant+=1
print('concordant={} total={} accuracy={:.2f}%'.format(concordant,len(conditions),100.0*concordant/len(conditions)))


masterdf['Weather']=0
for i in range(len(masterdf)):
    thisdoy=int(masterdf['doy'][i])
    try:
        weather=conditions['Weather'][thisdoy]
        masterdf.loc[i,'Weather']=weather
    except:
        masterdf.loc[i,'Weather']=3 #No data

gooddf=masterdf[masterdf[20]==0]
gooddf2=avgmasterdf[avgmasterdf[20]==0]

plt.scatter(gooddf['doy'][gooddf['Weather']==0],gooddf[7][gooddf['Weather']==0],color='r')
plt.scatter(gooddf['doy'][gooddf['Weather']==1],gooddf[7][gooddf['Weather']==1],color='g')
plt.scatter(gooddf['doy'][gooddf['Weather']==2],gooddf[7][gooddf['Weather']==2],color='b')
plt.scatter(gooddf['doy'][gooddf['Weather']==3],gooddf[7][gooddf['Weather']==3],color='y')
plt.show()

simpsondf['year']=simpsondf['datetime_AKDT'].apply(lambda x:x[0:4])
gooddf['year']=gooddf['datetime_AKDT'].apply(lambda x:x[0:4])

for year in ['2017','2018']:
    MAXDOASUAF=plt.scatter(simpsondf['doy'][(simpsondf['year']==year)],simpsondf['VCD'][(simpsondf['year']==year)],color='b')
    if year=='2017':
        plt.legend((MAXDOASUAF,),('MAXDOASUAF',),scatterpoints=1,loc='upper right')
        plt.title('VCD HCHO vs DOY ({})'.format(year))
        plt.ylabel('Dobson Unit (DU)')
        plt.xlabel('Day of Year (DOY)')
        plt.show()
    else:
        PandoraUAF=plt.scatter(gooddf['doy'][(gooddf['year']==year)],gooddf[7][(gooddf['year']==year)],color='r')
        plt.legend((MAXDOASUAF,PandoraUAF),('MAXDOASUAF','PandoraUAF'),scatterpoints=1,loc='upper right')
        plt.title('VCD HCHO vs DOY ({})'.format(year))
        plt.ylabel('Dobson Unit (DU)')
        plt.xlabel('Day of Year (DOY)')
        plt.show()


if True:
        PandoraUAF=plt.scatter(gooddf['doy'],gooddf[7],color='r')
        PandoraUAFAvg=plt.scatter(gooddf2['doy'],gooddf2[7],color='b')
        plt.legend((PandoraUAF,PandoraUAFAvg),('PandoraUAF','UAF1HrAvg'),scatterpoints=1,loc='upper right')
        plt.title('VCD HCHO vs DOY ({})'.format(year))
        plt.ylabel('Dobson Unit (DU)')
        plt.xlabel('Day of Year (DOY)')
        plt.show()

hist,bins=np.histogram(masterdf[7],bins=np.arange(-10.0,10.0,0.1))
center=(bins[:-1]+bins[1:])/2
plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
plt.title('Hist of VCD (master)')
plt.show()

hist,bins=np.histogram(simpsondf['VCD'].dropna(),bins=np.arange(-10.0,10.0,0.1))
center=(bins[:-1]+bins[1:])/2
plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
plt.title('Hist of VCD (simpsondf)')
plt.show()

hist,bins=np.histogram(masterdf[7],bins=np.arange(-2.0,2.0,0.1))
center=(bins[:-1]+bins[1:])/2
plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
plt.title('Hist of VCD (master)')
plt.show()

hist,bins=np.histogram(simpsondf['VCD'].dropna(),bins=np.arange(-10.0,10.0,0.1))
center=(bins[:-1]+bins[1:])/2
plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
plt.title('Hist of VCD (simpsondf)')
plt.show()

for monthstr in ['03', '04', '05', '06', '07']:
    sliceddf=getMonthSlice(masterdf,monthstr)
    hist,bins=np.histogram(sliceddf[7],bins=np.arange(-2.0,2.0,0.1))
    center=(bins[:-1]+bins[1:])/2
    plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
    plt.title('Hist of VCD (master) for month {}'.format(monthstr))
    plt.show()

    sliceddf_simp=getMonthSlice(simpsondf.loc[(simpsondf['year']=='2018')],monthstr)
    hist,bins=np.histogram(sliceddf_simp['VCD'].dropna(),bins=np.arange(-10.0,10.0,0.1))
    center=(bins[:-1]+bins[1:])/2
    plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
    plt.title('Hist of VCD (simpson) for month {}'.format(monthstr))
    plt.show()

    sliceddf_simp=getMonthSlice(simpsondf.loc[(simpsondf['year']=='2018')],monthstr)
    hist,bins=np.histogram(sliceddf_simp['VCD'].dropna(),bins=np.arange(-2.0,2.0,0.1))
    center=(bins[:-1]+bins[1:])/2
    plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
    plt.title('Hist of VCD (simpson) for month {}'.format(monthstr))
    plt.show()

for monthstr in ['03', '04', '05', '06', '07']:
    sliceddf=hourlyAverage(getMonthSlice(masterdf,monthstr))
    hist,bins=np.histogram(sliceddf[7],bins=np.arange(-2.0,2.0,0.1))
    center=(bins[:-1]+bins[1:])/2
    plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
    plt.title('Hist of VCD (master) for month {}'.format(monthstr))
    plt.show()

    sliceddf_simp=hourlyAverage(getMonthSlice(simpsondf.loc[(simpsondf['year']=='2018')],monthstr))
    hist,bins=np.histogram(sliceddf_simp['VCD'].dropna(),bins=np.arange(-10.0,10.0,0.1))
    center=(bins[:-1]+bins[1:])/2
    plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
    plt.title('Hist of VCD (simpson) for month {}'.format(monthstr))
    plt.show()

    sliceddf_simp=hourlyAverage(getMonthSlice(simpsondf.loc[(simpsondf['year']=='2018')],monthstr))
    hist,bins=np.histogram(sliceddf_simp['VCD'].dropna(),bins=np.arange(-2.0,2.0,0.1))
    center=(bins[:-1]+bins[1:])/2
    plt.bar(center,hist,align='center',width=(0.7*(bins[1]-bins[0])))
    plt.title('Hist of VCD (simpson) for month {}'.format(monthstr))
    plt.show()

print(masterdf.describe())
print(simpsondf.describe())
print('VCD on real sunny dates')
masterdf['date']=masterdf['datetime_AKDT'].apply(lambda s:s[:8])
masterdf.index=masterdf['date']
sunnydf=masterdf.loc[realsunnydates]
sunnydf.index=range(len(sunnydf.index))
masterdf.index=range(len(masterdf.index))
sunnydf=sunnydf.dropna()
sunnydf=sunnydf[sunnydf['datetime_AKDT'].apply(lambda s:int(s[9:11]))>13]
sunnydf=sunnydf[sunnydf['datetime_AKDT'].apply(lambda s:int(s[9:11]))<15]


simpsondf['date']=simpsondf['datetime_AKDT'].apply(lambda s:s[:8])
simpsondf.index=simpsondf['date']
sunnysimpsondf=simpsondf.loc[realsunnydates]
sunnysimpsondf.index=range(len(sunnysimpsondf.index))
simpsondf.index=range(len(simpsondf.index))


PandoraUAF=plt.scatter(sunnydf['doy'],sunnydf[7],color='r')
z = np.polyfit(sunnydf['doy'],sunnydf[7], 1)
p = np.poly1d(z)
x = sunnysimpsondf['doy'][(sunnysimpsondf['year']=='2018')]
plt.plot(x,p(x),"r--")
MAXDOASUAF=plt.scatter(sunnysimpsondf['doy'][(sunnysimpsondf['year']=='2018')],sunnysimpsondf['VCD'][(sunnysimpsondf['year']=='2018')],color='b')
z = np.polyfit(sunnysimpsondf.dropna()['doy'][(sunnysimpsondf['year']=='2018')],sunnysimpsondf.dropna()['VCD'][(sunnysimpsondf['year']=='2018')], 1)
p = np.poly1d(z)
plt.plot(x,p(x),"b--")

plt.legend((MAXDOASUAF,PandoraUAF),('MAXDOASUAF','PandoraUAF'),scatterpoints=1,loc='best')
plt.title('VCD HCHO (Sunny only) vs DOY ({})'.format(year))
plt.ylabel('Dobson Unit (DU)')
plt.xlabel('Day of Year (DOY)')
plt.show()

mao=skiprows('mao_hcho.csv', 57)
#High quality data - I want to plot the VCDs of those rows only.
mao=mao[mao[12]==0]
mao['datetime_AKDT']=mao[0].apply(akdt)
mao['doy']=mao['datetime_AKDT'].apply(doy)
maohr=hourlyAverage(mao)
maohr['date']=maohr['datetime_AKDT'].apply(lambda s:s[:8])
avgsimpson['date']=avgsimpson['datetime_AKDT'].apply(lambda s:s[:8])


maohr.index=maohr['date']
sunnymao=maohr.loc[realsunnydates]
sunnymao=sunnymao[notnull(sunnymao['datetime_AKDT'])]
sunnymao.index=range(len(sunnymao.index))
maohr.index=range(len(maohr.index))

avgsimpson.index=avgsimpson['date']
sunnysimpson=avgsimpson.loc[realsunnydates]
sunnysimpson=sunnysimpson[notnull(sunnysimpson['datetime_AKDT'])]
sunnysimpson.index=range(len(sunnysimpson.index))
avgsimpson.index=range(len(avgsimpson.index))

avgs=[]
times=[]
avgs2=[]
times2=[]
for i in range(24):
    sunnymao2=sunnymao[sunnymao['datetime_AKDT'].apply(lambda s:int(s[9:11]))==i]
    if len(sunnymao2):
        avgs.append(sunnymao2[7].mean())
        times.append(i)
    sunnysimpson2=sunnysimpson[sunnysimpson['datetime_AKDT'].apply(lambda s:int(s[9:11]))==i]
    if len(sunnysimpson2):
        avgs2.append(sunnysimpson2['VCD'].mean())
        times2.append(i)
sp1=plt.scatter(times,avgs,color='b')
sp2=plt.scatter(times2,avgs2,color='r')
plt.legend((sp1,sp2),('Mao','Simpson'),scatterpoints=1,loc='best')
plt.title('VCD HCHO (Sunny only) vs Time of day')
plt.ylabel('VCD')
plt.xlabel('Time of day')
plt.show()


for yyyymm in ['201805','201806','201807']:
    realsunnydates2=[d for d in realsunnydates if d[0:6]==yyyymm]
    if (len(realsunnydates2)>8):
        todrop=(len(realsunnydates2)-7)//2
        realsunnydates2=realsunnydates2[todrop:-todrop]
    maohr.index=maohr['date']
    sunnymao=maohr.loc[realsunnydates2]
    sunnymao=sunnymao[notnull(sunnymao['datetime_AKDT'])]
    sunnymao.index=range(len(sunnymao.index))
    maohr.index=range(len(maohr.index))

    avgsimpson.index=avgsimpson['date']
    sunnysimpson=avgsimpson.loc[realsunnydates2]
    sunnysimpson=sunnysimpson[notnull(sunnysimpson['datetime_AKDT'])]
    sunnysimpson.index=range(len(sunnysimpson.index))
    avgsimpson.index=range(len(avgsimpson.index))


    avgs=[]
    times=[]
    avgs2=[]
    times2=[]
    for i in range(24):
        sunnymao2=sunnymao[sunnymao['datetime_AKDT'].apply(lambda s:int(s[9:11]))==i]
        if len(sunnymao2):
            avgs.append(sunnymao2[7].mean())
            times.append(i)
        sunnysimpson2=sunnysimpson[sunnysimpson['datetime_AKDT'].apply(lambda s:int(s[9:11]))==i]
        if len(sunnysimpson2):
            avgs2.append(sunnysimpson2['VCD'].mean())
            times2.append(i)
    sp1=plt.scatter(times,avgs,color='b')
    sp2=plt.scatter(times2,avgs2,color='r')
    plt.legend((sp1,sp2),('Mao','Simpson'),scatterpoints=1,loc='best')
    plt.title('VCD HCHO (Sunny only for {}) vs Time of day'.format(yyyymm))
    plt.ylabel('VCD')
    plt.xlabel('Time of day')
    plt.show()